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


def _load_net(
    checkpoint: Path, training_config: DAVIConfig, device: torch.device
) -> ValueNet:
    """Reconstruct + load a ValueNet from a training checkpoint."""
    net = ValueNet(
        CUBE_3X3,
        body_widths=training_config.body_widths,
        n_residual_blocks=training_config.n_residual_blocks,
        normalization=training_config.normalization,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "net_state" in ckpt:
        net.load_state_dict(ckpt["net_state"])
    else:
        # Legacy bare state-dict format.
        net.load_state_dict(ckpt)
    net.eval()
    return net


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
    net = _load_net(model_path, training_config, device)
    dtype = PRECISION_DTYPES[eval_cfg.precision]
    if dtype is not torch.float32:
        net = net.to(dtype)

    per_walk_depth, wall = _walk_eval(net, eval_cfg, eval_cfg.seed)
    states_scored = sum(int(item["n"]) for item in per_walk_depth)

    out: dict = {
        "model": str(model_path.resolve()),
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
        action="store_true",
        help=(
            "Also render the HTML report next to the JSON via "
            "render_beam_eval_report.py."
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

    if args.render_html:
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
