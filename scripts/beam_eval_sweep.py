"""Single-checkpoint beam-eval sweep across one parameter axis.

Wraps ``beam_eval_model.py``'s ``evaluate_checkpoint`` to produce one
JSON per swept value. The sweep axis is selected with ``--over PARAM``:

- ``--over width``      ``--values "8,16,32,64,128,256,512"``
- ``--over precision``  ``--values "fp32,bf16,fp16"``
- ``--over seed``       ``--values "0,1,2,3,4"``  (multi-seed CI)
- ``--over max-depth``  ``--values "8,10,12,14"``

For each swept value:

1. Load the base config (``--config NAME_OR_PATH``).
2. Apply CLI overrides for the *non-swept* params.
3. Apply the swept value LAST.
4. Run ``evaluate_checkpoint`` and write the result to
   ``<model-dir>/<model-stem>_eval_<config-name>_<param>=<value>.json``.

When ``--render-html`` is set, after all sweeps invoke
``render_beam_eval_report.py`` with ``--input`` set to every output JSON
to produce a single overlay HTML at
``<model-dir>/<model-stem>_eval_<config-name>_<param>_sweep.html``.

The renderer auto-detects the new flat-schema inputs and labels each
overlay legend entry from the filename's ``_<param>=<value>`` suffix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "davi-3x3"
ORACLE_CACHE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_config import EvalConfig  # noqa: E402
from beam_eval_model import _resolve_training_config, evaluate_checkpoint  # noqa: E402

from rubik.oracle.v_star_bounded_3x3 import (  # noqa: E402
    load_v_star_bounded_3x3_arrays,
)
from rubik.training.config import DAVIConfig  # noqa: E402

# Map ``--over`` CLI name → (parser callable, EvalConfig field name used
# in the filename, with_overrides kwarg).
SWEEP_AXES: dict[str, dict] = {
    "width": {
        "parse": int,
        "kwarg": "beam_width",
    },
    "precision": {
        "parse": str,
        "kwarg": "precision",
    },
    "seed": {
        "parse": int,
        "kwarg": "seed",
    },
    "max-depth": {
        "parse": int,
        "kwarg": "max_depth",
    },
}


def _parse_values(over: str, raw: str) -> list:
    parser = SWEEP_AXES[over]["parse"]
    out: list = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(parser(s))
    if not out:
        raise ValueError("--values must contain at least one entry")
    return out


def _apply_swept(eval_cfg: EvalConfig, over: str, value) -> EvalConfig:
    """Apply the swept value LAST so it overrides the CLI overrides."""
    kwarg = SWEEP_AXES[over]["kwarg"]
    return eval_cfg.with_overrides(**{kwarg: value})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="Path to a .pt checkpoint.",
    )
    parser.add_argument(
        "--over",
        choices=list(SWEEP_AXES.keys()),
        required=True,
        help="Parameter to sweep over.",
    )
    parser.add_argument(
        "--values",
        type=str,
        required=True,
        help=(
            "Comma-separated values to sweep over. Type matches the axis: "
            "ints for width/seed/max-depth, strings for precision."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help=("Base eval config name or path (defaults applied to all sweep runs)."),
    )
    # Non-swept-param overrides applied to every run.
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=("fp32", "bf16", "fp16"),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-v-star", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--render-html",
        action="store_true",
        help=(
            "After all sweep runs, invoke render_beam_eval_report.py with "
            "--input set to every output JSON."
        ),
    )
    args = parser.parse_args(argv)

    model_path: Path = args.model
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    values = _parse_values(args.over, args.values)
    base_cfg = EvalConfig.resolve(args.config)

    # Apply non-swept overrides (the swept axis's override is ignored even
    # if the user passes it; the per-run sweep value takes priority below).
    overrides_for_all = {
        "beam_width": args.width,
        "precision": args.precision,
        "seed": args.seed,
        "include_v_star": True if args.include_v_star else None,
        "max_depth": args.max_depth,
    }
    # Strip the swept axis's kwarg from the all-runs overrides so it
    # doesn't double-apply — the sweep value goes in last via
    # _apply_swept.
    swept_kwarg = SWEEP_AXES[args.over]["kwarg"]
    overrides_for_all[swept_kwarg] = None
    base_cfg = base_cfg.with_overrides(**overrides_for_all)

    training_config_path = _resolve_training_config(model_path)
    training_config = DAVIConfig.from_yaml(training_config_path)
    device = torch.device(args.device or training_config.device)

    # Pre-load oracle once if any sweep run needs V*.
    oracle_arrays = None
    if base_cfg.include_v_star:
        if not ORACLE_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"--include-v-star requires {ORACLE_CACHE_PATH} (run "
                "scripts/build_v_star_bounded_3x3.py first)."
            )
        oracle_arrays = load_v_star_bounded_3x3_arrays(ORACLE_CACHE_PATH)

    written: list[Path] = []
    for value in values:
        cfg = _apply_swept(base_cfg, args.over, value)
        out_name = (
            f"{model_path.stem}_eval_{base_cfg.source_name}_{args.over}={value}.json"
        )
        out_path = model_path.parent / out_name
        print(
            f"[sweep] {args.over}={value} → {out_path.name}",
            file=sys.stderr,
            flush=True,
        )
        payload = evaluate_checkpoint(
            model_path=model_path,
            eval_cfg=cfg,
            device=device,
            training_config=training_config,
            oracle_arrays=oracle_arrays,
        )
        out_path.write_text(json.dumps(payload, indent=2))
        written.append(out_path)
        deep_rate = (
            payload["per_walk_depth"][-1]["solve_rate"]
            if payload["per_walk_depth"]
            else float("nan")
        )
        print(
            f"  wall={payload['wall_time_seconds']:.1f}s "
            f"solve_rate@d={cfg.max_depth}={deep_rate:.3f}",
            file=sys.stderr,
            flush=True,
        )

    print(f"\n{len(written)} sweep JSON(s) written.")
    for p in written:
        print(f"  {p}")

    if args.render_html:
        renderer = REPO_ROOT / "scripts" / "render_beam_eval_report.py"
        html_name = (
            f"{model_path.stem}_eval_{base_cfg.source_name}_{args.over}_sweep.html"
        )
        html_path = model_path.parent / html_name
        cmd = [
            sys.executable,
            str(renderer),
            "--input",
            *(str(p) for p in written),
            "--output",
            str(html_path),
        ]
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
