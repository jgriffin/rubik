"""Parameterized beam-eval CLI with width sweep.

Sibling to ``scripts/post_run_beam_eval.py``. The post-run version is
trajectory-style: takes a run dir, iterates its checkpoints, fixed
``beam_width=256``. This version is **width-sweep style**: takes a
single checkpoint and runs ``beam_eval_walk`` (and optionally
``beam_eval_v_star``) at each width in a comma-separated list, recording
per-width wall time. Used to find the speed/quality tradeoff and pick
production beam-width defaults.

Usage::

    uv run python scripts/beam_eval_run.py \\
        --checkpoint <path> \\
        [--config <path>] \\
        --widths "8,16,32,64,128,256,512" \\
        [--max-walk-depth 14] \\
        [--n-per-depth 100] \\
        [--device mps] \\
        --output-json <path> \\
        [--include-v-star] \\
        [--seed 0]

If ``--config`` is omitted, the script auto-discovers
``<checkpoint>.parent / "config.yaml"``.

Output JSON schema::

    {
      "checkpoint": "<absolute path>",
      "config_path": "<absolute path>",
      "device": "mps",
      "precision": "bf16",
      "max_walk_depth": 14,
      "n_per_depth": 100,
      "seed": 0,
      "results": {
        "8": {
          "per_walk_depth": [
            {"d": 1, "solve_rate": 1.0, "avg_solve_len": 1.0, "n": 100},
            ...
          ],
          "wall_time_seconds": 12.34,
          "states_scored": 1400
        },
        ...
      }
    }

When ``--include-v-star`` is set, a sibling top-level key ``v_star_results``
is added with per-width ``per_v_star`` lists and ``wall_time_seconds``.
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

# Make ``experiments/davi-3x3/eval.py`` importable without a package marker
# (mirrors scripts/post_run_beam_eval.py).
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from eval import beam_eval_v_star, beam_eval_walk  # noqa: E402

from rubik.cube.spec import CUBE_3X3  # noqa: E402
from rubik.model.network import ValueNet  # noqa: E402
from rubik.oracle.v_star_bounded_3x3 import (  # noqa: E402
    load_v_star_bounded_3x3_arrays,
)
from rubik.training.config import DAVIConfig  # noqa: E402


def _parse_widths(arg: str) -> list[int]:
    widths: list[int] = []
    for part in arg.split(","):
        s = part.strip()
        if not s:
            continue
        w = int(s)
        if w <= 0:
            raise ValueError(f"widths must be positive ints; got {w}")
        widths.append(w)
    if not widths:
        raise ValueError("--widths must contain at least one positive int")
    return widths


def _resolve_config_path(checkpoint: Path, config_arg: Path | None) -> Path:
    if config_arg is not None:
        if not config_arg.exists():
            raise FileNotFoundError(f"--config path not found: {config_arg}")
        return config_arg
    sibling = checkpoint.parent / "config.yaml"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        f"could not auto-discover config.yaml; pass --config explicitly. "
        f"Looked at: {sibling}"
    )


def _load_net(checkpoint: Path, config: DAVIConfig, device: torch.device) -> ValueNet:
    """Reconstruct + load a ValueNet for ``checkpoint`` (mirrors post_run_beam_eval)."""
    net = ValueNet(
        CUBE_3X3,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
        normalization=config.normalization,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "net_state" in ckpt:
        net.load_state_dict(ckpt["net_state"])
    else:
        # Legacy bare state-dict format.
        net.load_state_dict(ckpt)
    net.eval()
    return net


def _walk_metrics_to_per_walk_depth(
    metrics: dict, walk_depths: tuple[int, ...], n_per_depth: int
) -> list[dict]:
    """Reshape ``beam_eval_walk``'s flat dict into a per-depth list."""
    per_walk_depth: list[dict] = []
    for d in walk_depths:
        rate = float(metrics[f"solve_rate_d{int(d)}"])
        avg = metrics[f"avg_solve_len_d{int(d)}"]
        per_walk_depth.append(
            {
                "d": int(d),
                "solve_rate": rate,
                "avg_solve_len": float(avg) if avg is not None else None,
                "n": int(n_per_depth),
            }
        )
    return per_walk_depth


def _v_star_metrics_to_per_v_star(metrics: dict, n_per_layer: int) -> list[dict]:
    """Reshape ``beam_eval_v_star``'s flat dict into a per-V* list."""
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
                "solve_rate": rate,
                "avg_solve_len": float(avg) if avg is not None else None,
                "mae": mae,
                "n": int(n_per_layer),
            }
        )
    return per_v_star


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a .pt checkpoint (e.g. net_step_*.pt or net_final.pt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to the run's config.yaml. If omitted, auto-discovers from "
            "<checkpoint>.parent / 'config.yaml'."
        ),
    )
    parser.add_argument(
        "--widths",
        type=str,
        required=True,
        help=("Comma-separated beam widths to sweep, e.g. '8,16,32,64,128,256,512'."),
    )
    parser.add_argument(
        "--max-walk-depth",
        type=int,
        default=14,
        help="Walk-depth grid is range(1, max_walk_depth + 1). Default 14.",
    )
    parser.add_argument(
        "--n-per-depth",
        type=int,
        default=100,
        help="Number of random-walk scrambles per walk-depth bin.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (default: read from config.yaml).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--include-v-star",
        action="store_true",
        help="Also run beam_eval_v_star per width (V*-stratified eval).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the per-width torch.Generator (deterministic across widths).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=("fp32", "bf16", "fp16"),
        help=(
            "Inference precision for the loaded checkpoint. bf16 (default) "
            "is ~1.3-5× faster than fp32 on M4 Max with bit-identical solve "
            "rates on the trained net (precision-sweep experiment, 2026-05-07). "
            "Pass fp32 for bit-exact reproduction of pre-flip runs. Training is "
            "untouched — checkpoint weights are stored in fp32 on disk."
        ),
    )
    args = parser.parse_args(argv)

    checkpoint: Path = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint}")

    config_path = _resolve_config_path(checkpoint, args.config)
    config = DAVIConfig.from_yaml(config_path)
    device = torch.device(args.device or config.device)
    widths = _parse_widths(args.widths)
    max_walk_depth = int(args.max_walk_depth)
    if max_walk_depth < 1:
        raise ValueError(f"--max-walk-depth must be >= 1; got {max_walk_depth}")
    walk_depths = tuple(range(1, max_walk_depth + 1))
    n_per_depth = int(args.n_per_depth)

    net = _load_net(checkpoint, config, device)

    # Precision cast: weights load in fp32, then cast to the requested dtype
    # for inference. Input one-hot follows ``self.fc_in.weight.dtype`` so
    # the whole forward runs in the chosen precision.
    precision_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.precision]
    if precision_dtype is not torch.float32:
        net = net.to(precision_dtype)

    # Optional V*-stratified eval needs the bounded oracle arrays.
    oracle_arrays = None
    n_per_v_star_layer = 200  # match post_run_beam_eval default
    if args.include_v_star:
        if not ORACLE_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"--include-v-star requires {ORACLE_CACHE_PATH} (run "
                "scripts/build_v_star_bounded_3x3.py first)."
            )
        oracle_arrays = load_v_star_bounded_3x3_arrays(ORACLE_CACHE_PATH)

    results: dict[str, dict] = {}
    v_star_results: dict[str, dict] = {}

    for width in widths:
        # Fresh per-width generator seeded the same way → walk states are
        # identical across widths within a single run, and bit-identical
        # across runs at the same seed. The network's beam-search outputs
        # vary with width; the input distribution does not.
        gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
        t0 = time.perf_counter()
        walk_metrics = beam_eval_walk(
            net,
            CUBE_3X3,
            n_per_depth=n_per_depth,
            walk_depths=walk_depths,
            beam_width=width,
            generator=gen,
        )
        dt = time.perf_counter() - t0
        per_walk_depth = _walk_metrics_to_per_walk_depth(
            walk_metrics, walk_depths, n_per_depth
        )
        states_scored = n_per_depth * len(walk_depths)
        results[str(width)] = {
            "per_walk_depth": per_walk_depth,
            "wall_time_seconds": float(dt),
            "states_scored": int(states_scored),
        }
        # Progress line — solve_rate at the deepest walk is the headline number.
        deep_rate = per_walk_depth[-1]["solve_rate"] if per_walk_depth else float("nan")
        print(
            f"[beam-eval] width={width} wall={dt:.1f}s "
            f"solve_rate@d={max_walk_depth}={deep_rate:.3f}",
            file=sys.stderr,
            flush=True,
        )

        if args.include_v_star:
            import numpy as np  # local: heavy import

            rng = np.random.default_rng(int(args.seed))
            t0v = time.perf_counter()
            v_metrics = beam_eval_v_star(
                net,
                CUBE_3X3,
                oracle_arrays,
                n_per_layer=n_per_v_star_layer,
                beam_width=width,
                rng=rng,
            )
            dtv = time.perf_counter() - t0v
            per_v_star = _v_star_metrics_to_per_v_star(v_metrics, n_per_v_star_layer)
            v_star_results[str(width)] = {
                "per_v_star": per_v_star,
                "wall_time_seconds": float(dtv),
            }

    out: dict = {
        "checkpoint": str(checkpoint.resolve()),
        "config_path": str(config_path.resolve()),
        "device": str(device),
        "precision": args.precision,
        "max_walk_depth": max_walk_depth,
        "n_per_depth": n_per_depth,
        "seed": int(args.seed),
        "results": results,
    }
    if args.include_v_star:
        out["v_star_results"] = v_star_results

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))

    # Summary table to stdout.
    print(f"wrote {args.output_json}")
    print()
    print(f"{'width':>8s}  {'wall (s)':>10s}  {'solve@d=' + str(max_walk_depth):>14s}")
    for width in widths:
        entry = results[str(width)]
        deep_rate = entry["per_walk_depth"][-1]["solve_rate"]
        print(f"{width:>8d}  {entry['wall_time_seconds']:>10.2f}  {deep_rate:>14.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
