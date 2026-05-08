"""Single-checkpoint beam-eval CLI driven by a YAML config.

Common workflow::

    uv run python scripts/beam_eval_model.py <model.pt> --config fast

Where ``fast`` resolves to ``scripts/eval-configs/fast.yaml`` and bundles
``n_per_depth`` (per-depth sample schedule), ``beam_width``, ``precision``,
and ``seed``. CLI flags override individual YAML fields.

Output JSON schema (flat, single-width)::

    {
      "model": "<absolute path>",
      "config_name": "fast",
      "config_description": "...",
      "device": "mps",
      "precision": "bf16",
      "max_depth": 14,
      "beam_width": 256,
      "seed": 0,
      "n_per_depth": [12, 24, ...],
      "wall_time_seconds": 25.4,
      "per_walk_depth": [
          {"d": 1, "n": 12, "solve_rate": 1.0, "avg_solve_len": 1.0},
          ...
      ],
      "states_scored": 532
    }

When ``--include-v-star`` is set, a sibling top-level ``v_star_results``
key is added with ``per_v_star`` + ``wall_time_seconds`` (same shape as
the legacy ``post_run_beam_eval.json``'s per-V* dict).

This is a breaking schema change vs. the prior width-keyed
``results.<width>`` shape — sweeps now produce one JSON per swept value
and rely on filename suffixes to encode the swept axis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "davi-3x3"
ORACLE_CACHE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"

# Make ``experiments/davi-3x3/eval.py`` importable without a package marker.
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

# Add scripts/ to sys.path for the _eval_config helper.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_config import EvalConfig  # noqa: E402
from eval import beam_eval_v_star, beam_eval_walk  # noqa: E402

from rubik.cube.spec import CUBE_3X3  # noqa: E402
from rubik.model.network import ValueNet  # noqa: E402
from rubik.oracle.v_star_bounded_3x3 import (  # noqa: E402
    load_v_star_bounded_3x3_arrays,
)
from rubik.training.config import DAVIConfig  # noqa: E402

PRECISION_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _resolve_training_config(model_path: Path) -> Path:
    """Auto-discover the training-time ``config.yaml`` (architecture + device)."""
    sibling = model_path.parent / "config.yaml"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        f"could not auto-discover training config.yaml; expected at {sibling}. "
        "(This is the run's training config — separate from --config which "
        "selects the eval-schedule YAML under scripts/eval-configs/.)"
    )


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict, bool]:
    """Load a checkpoint file. Returns ``(raw, is_bundle)``.

    If the file contains a bundle dict with ``net_state`` (the new
    training-bundle format), returns ``(dict, True)``. Otherwise treats
    the contents as a bare state_dict (legacy format) and returns
    ``(state_dict, False)``. Callers extract the model state via
    ``ckpt["net_state"] if is_bundle else ckpt``.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "net_state" in raw:
        return raw, True
    return raw, False


def _build_net(
    ckpt: dict,
    is_bundle: bool,
    training_config: DAVIConfig,
    device: torch.device,
) -> ValueNet:
    """Construct a ValueNet from architecture in ``training_config`` and load
    weights from the loaded ``ckpt``. ``is_bundle`` flags the new bundle
    format vs legacy bare-state-dict.
    """
    net = ValueNet(
        CUBE_3X3,
        body_widths=training_config.body_widths,
        n_residual_blocks=training_config.n_residual_blocks,
        normalization=training_config.normalization,
    ).to(device)
    state_dict = ckpt["net_state"] if is_bundle else ckpt
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _load_net(
    checkpoint: Path, training_config: DAVIConfig, device: torch.device
) -> ValueNet:
    """Backwards-compatible thin wrapper around ``_load_checkpoint`` +
    ``_build_net``. Discards the bundle metadata. Kept in case any out-of-
    tree script imports this name directly; new code should call the two
    primitives separately so it can read the bundle metadata fields.
    """
    ckpt, is_bundle = _load_checkpoint(checkpoint, device)
    return _build_net(ckpt, is_bundle, training_config, device)


def _walk_eval(
    net: torch.nn.Module,
    eval_cfg: EvalConfig,
    seed: int,
) -> tuple[list[dict], float]:
    """Run the per-depth beam_eval_walk, returning per_walk_depth + wall."""
    per_walk_depth: list[dict] = []
    t0 = time.perf_counter()
    # Each depth bin gets its own n_per_depth from the config schedule.
    for i, d in enumerate(eval_cfg.walk_depths):
        n_d = int(eval_cfg.n_per_depth[i])
        # Per-depth seeding (rather than one shared generator) means changing
        # the schedule at one depth doesn't perturb the others' walk states.
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + i)
        metrics = beam_eval_walk(
            net,
            CUBE_3X3,
            n_per_depth=n_d,
            walk_depths=(int(d),),
            beam_width=eval_cfg.beam_width,
            generator=gen,
        )
        rate = float(metrics[f"solve_rate_d{int(d)}"])
        avg = metrics[f"avg_solve_len_d{int(d)}"]
        per_walk_depth.append(
            {
                "d": int(d),
                "n": n_d,
                "solve_rate": rate,
                "avg_solve_len": float(avg) if avg is not None else None,
            }
        )
    wall = time.perf_counter() - t0
    return per_walk_depth, wall


def _v_star_eval(
    net: torch.nn.Module,
    eval_cfg: EvalConfig,
    oracle_arrays: tuple,
    seed: int,
) -> tuple[list[dict], float]:
    """Run beam_eval_v_star and reshape its flat dict to a per-V* list."""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    metrics = beam_eval_v_star(
        net,
        CUBE_3X3,
        oracle_arrays,
        n_per_layer=eval_cfg.n_per_v_star_layer,
        beam_width=eval_cfg.beam_width,
        rng=rng,
    )
    wall = time.perf_counter() - t0
    layers: set[int] = set()
    for key in metrics:
        if key.startswith("solve_rate_v"):
            layers.add(int(key.removeprefix("solve_rate_v")))
    per_v_star: list[dict] = []
    for v in sorted(layers):
        rate = float(metrics[f"solve_rate_v{v}"])
        avg = metrics[f"avg_solve_len_v{v}"]
        mae = float(metrics[f"mae_v{v}"])
        per_v_star.append(
            {
                "v": v,
                "n": int(eval_cfg.n_per_v_star_layer),
                "solve_rate": rate,
                "avg_solve_len": float(avg) if avg is not None else None,
                "mae": mae,
            }
        )
    return per_v_star, wall


def evaluate_checkpoint(
    *,
    model_path: Path,
    eval_cfg: EvalConfig,
    device: torch.device,
    training_config: DAVIConfig,
    oracle_arrays: tuple | None = None,
) -> dict:
    """Core eval routine, shared with ``beam_eval_run.py`` and ``beam_eval_sweep.py``.

    Returns the JSON payload (flat schema, no on-disk write).
    """
    ckpt, is_bundle = _load_checkpoint(model_path, device)
    net = _build_net(ckpt, is_bundle, training_config, device)
    dtype = PRECISION_DTYPES[eval_cfg.precision]
    if dtype is not torch.float32:
        net = net.to(dtype)

    # Bundle-metadata pass-through. Each field is None when the
    # checkpoint is legacy (bare state_dict) or when training didn't
    # populate that particular optional field. Surfacing them in the
    # payload lets downstream consumers (beam_eval_run.py, plots) prefer
    # dict-step over filename-extracted step and tag results with the
    # originating wandb run.
    checkpoint_step: int | None = (
        int(ckpt["step"]) if is_bundle and "step" in ckpt else None
    )
    checkpoint_wall_time_seconds: float | None = (
        float(ckpt["wall_time_seconds"])
        if is_bundle and "wall_time_seconds" in ckpt
        else None
    )
    checkpoint_wandb_run_id: str | None = (
        str(ckpt["wandb_run_id"]) if is_bundle and "wandb_run_id" in ckpt else None
    )

    per_walk_depth, wall = _walk_eval(net, eval_cfg, eval_cfg.seed)
    states_scored = sum(int(item["n"]) for item in per_walk_depth)

    out: dict = {
        "model": str(model_path.resolve()),
        "checkpoint_step": checkpoint_step,
        "checkpoint_wall_time_seconds": checkpoint_wall_time_seconds,
        "checkpoint_wandb_run_id": checkpoint_wandb_run_id,
        "config_name": eval_cfg.source_name,
        "config_description": eval_cfg.description,
        "device": str(device),
        "precision": eval_cfg.precision,
        "max_depth": eval_cfg.max_depth,
        "beam_width": eval_cfg.beam_width,
        "seed": eval_cfg.seed,
        "n_per_depth": list(eval_cfg.n_per_depth),
        "wall_time_seconds": float(wall),
        "per_walk_depth": per_walk_depth,
        "states_scored": int(states_scored),
    }
    if eval_cfg.include_v_star:
        if oracle_arrays is None:
            if not ORACLE_CACHE_PATH.exists():
                raise FileNotFoundError(
                    f"--include-v-star requires {ORACLE_CACHE_PATH} (run "
                    "scripts/build_v_star_bounded_3x3.py first)."
                )
            oracle_arrays = load_v_star_bounded_3x3_arrays(ORACLE_CACHE_PATH)
        per_v_star, v_wall = _v_star_eval(net, eval_cfg, oracle_arrays, eval_cfg.seed)
        out["v_star_results"] = {
            "per_v_star": per_v_star,
            "wall_time_seconds": float(v_wall),
            "n_per_layer": int(eval_cfg.n_per_v_star_layer),
        }
    return out


def build_default_output_path(model_path: Path, eval_cfg: EvalConfig) -> Path:
    """Default JSON path: ``<model-dir>/<model-stem>_eval_<config-name>.json``."""
    return model_path.parent / f"{model_path.stem}_eval_{eval_cfg.source_name}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="Path to a .pt checkpoint (e.g. net_step_*.pt or net_final.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help=(
            "Eval config name (resolves to scripts/eval-configs/<name>.yaml) "
            "or a literal filesystem path to a YAML file. Default: 'default'."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help=(
            "Slice n_per_depth[:N] from the config. Errors if N exceeds the "
            "schedule length. Default: full schedule from the config."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override config.beam_width.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=("fp32", "bf16", "fp16"),
        help="Override config.precision.",
    )
    parser.add_argument(
        "--include-v-star",
        action="store_true",
        help=(
            "Force include_v_star=True (overrides config). Requires "
            "data/v_star_bounded_3x3_k6.npz. Boolean flag — there's no "
            "matching --no-include-v-star; if the YAML enables it and you "
            "want to disable, edit the YAML."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config.seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Override the inference device (default: training config.yaml's "
            "device). Use 'cpu' for tiny-net unit tests."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: "
            "<model-dir>/<model-stem>_eval_<config-name>.json"
        ),
    )
    parser.add_argument(
        "--render-html",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override config.render_html. Pass --render-html to force HTML "
            "output, or --no-render-html to suppress. When omitted, the "
            "YAML config's render_html field decides."
        ),
    )
    args = parser.parse_args(argv)

    model_path: Path = args.model
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    eval_cfg = EvalConfig.resolve(args.config)
    eval_cfg = eval_cfg.with_overrides(
        beam_width=args.width,
        precision=args.precision,
        seed=args.seed,
        include_v_star=True if args.include_v_star else None,
        max_depth=args.max_depth,
        render_html=args.render_html,
    )

    training_config_path = _resolve_training_config(model_path)
    training_config = DAVIConfig.from_yaml(training_config_path)
    device = torch.device(args.device or training_config.device)

    payload = evaluate_checkpoint(
        model_path=model_path,
        eval_cfg=eval_cfg,
        device=device,
        training_config=training_config,
    )

    out_path = args.output_json or build_default_output_path(model_path, eval_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    # Headline progress line — solve rate at the deepest walk.
    deep_rate = (
        payload["per_walk_depth"][-1]["solve_rate"]
        if payload["per_walk_depth"]
        else float("nan")
    )
    print(
        f"[beam-eval] config={eval_cfg.source_name} "
        f"width={eval_cfg.beam_width} precision={eval_cfg.precision} "
        f"wall={payload['wall_time_seconds']:.1f}s "
        f"solve_rate@d={eval_cfg.max_depth}={deep_rate:.3f}",
        file=sys.stderr,
        flush=True,
    )
    print(f"wrote {out_path}")

    if eval_cfg.render_html:
        # Chain the renderer. Same Python interpreter to keep imports cheap.
        import subprocess

        renderer = REPO_ROOT / "scripts" / "render_beam_eval_report.py"
        html_path = out_path.with_suffix(".html")
        subprocess.run(
            [
                sys.executable,
                str(renderer),
                "--input",
                str(out_path),
                "--output",
                str(html_path),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
