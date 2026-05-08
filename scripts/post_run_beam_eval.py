"""Post-hoc beam-eval trajectory for a 3x3 DAVI run directory.

Iterates checkpoints in ``<run-dir>/`` (matching ``net_step_*.pt`` and
``net_final.pt``), runs ``beam_eval_walk`` + ``beam_eval_v_star`` on each,
and writes a trajectory JSON keyed by checkpoint step → eval dict to
``<run-dir>/results/post_run_beam_eval.json``.

Means we can save intermediate checkpoints during training (cheap) and
post-process them later without slowing training. See plan
``plans/m8-3x3-davi.md`` §P1c.

**Empty-checkpoint behavior.** If the run dir contains no checkpoints
matching ``net_step_*.pt`` or ``net_final.pt``, this script writes an
empty JSON object (``{}``) to the output path and exits 0. Tests rely
on this contract — a freshly-created run dir with an empty
``checkpoints/`` is a normal pre-training state, not an error.

Usage::

    uv run python scripts/post_run_beam_eval.py --run-dir <path>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "davi-3x3"
CACHE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"

# Make ``experiments/davi-3x3/eval.py`` importable without a package marker.
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from eval import beam_eval_v_star, beam_eval_walk  # noqa: E402

from rubik.cube.spec import CUBE_3X3  # noqa: E402
from rubik.model.network import ValueNet  # noqa: E402
from rubik.oracle.v_star_bounded_3x3 import (  # noqa: E402
    load_v_star_bounded_3x3_arrays,
)
from rubik.training.config import DAVIConfig  # noqa: E402

# Match ``net_step_<int>.pt`` and ``net_final.pt``. Final maps to the
# actual final step (early-stop-aware: parsed from metrics.jsonl) when
# available, falling back to the config's n_steps.
_STEP_RE = re.compile(r"^net_step_(\d+)\.pt$")


def _final_step_from_metrics(run_dir: Path, fallback: int) -> int:
    """Return the highest step recorded in metrics.jsonl, or ``fallback``.

    With early-stop, the run can terminate before ``config.n_steps``; in
    that case ``net_final.pt`` corresponds to the early-stop step, not
    ``n_steps``. Reading metrics.jsonl gives the truthful step label so
    legends and trajectories don't compress the final checkpoint to the
    far right of the x-axis.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return int(fallback)
    max_step = 0
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = rec.get("step")
        if isinstance(s, int) and s > max_step:
            max_step = s
    return max_step or int(fallback)


def _list_checkpoints(run_dir: Path, n_steps: int) -> list[tuple[int, Path]]:
    """Return ``[(step, path), ...]`` sorted ascending by step."""
    out: list[tuple[int, Path]] = []
    final_step = _final_step_from_metrics(run_dir, fallback=n_steps)
    for p in sorted(run_dir.iterdir() if run_dir.exists() else []):
        if p.name == "net_final.pt":
            out.append((int(final_step), p))
            continue
        m = _STEP_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def _load_net(ckpt_path: Path, config: DAVIConfig, device: torch.device) -> ValueNet:
    """Reconstruct + load a ValueNet for a given checkpoint."""
    net = ValueNet(
        CUBE_3X3,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
        normalization=config.normalization,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "net_state" in ckpt:
        net.load_state_dict(ckpt["net_state"])
    else:
        # Legacy bare state-dict format.
        net.load_state_dict(ckpt)
    net.eval()
    return net


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory with config.yaml + checkpoints",
    )
    parser.add_argument(
        "--n-per-walk-depth",
        type=int,
        default=100,
        help="Number of random-walk scrambles per walk-depth bin in beam_eval_walk.",
    )
    parser.add_argument(
        "--n-per-v-star-layer",
        type=int,
        default=200,
        help="Number of states sampled per V* layer in beam_eval_v_star.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=256,
        help="Beam width for both beam evals.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (default: read from run config.yaml).",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of step labels to evaluate "
            "(e.g. '20000,30000,120000,final'). The literal 'final' selects "
            "net_final.pt (label resolved from metrics.jsonl). When omitted, "
            "every net_step_*.pt and net_final.pt is evaluated."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=("fp32", "bf16", "fp16"),
        help=(
            "Inference precision for the loaded checkpoint. bf16 (default) "
            "is ~1.3-5× faster than fp32 on M4 Max with bit-identical solve "
            "rates on the trained net. Pass fp32 for bit-exact reproduction. "
            "Training is untouched — checkpoint weights stay fp32 on disk."
        ),
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    config_path = run_dir / "config.yaml"
    results_dir = run_dir / "results"
    out_path = results_dir / "post_run_beam_eval.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    # If config.yaml is absent, behavior depends on whether checkpoints
    # exist: a missing config + no checkpoints is "fresh run dir, nothing
    # to eval" → write empty JSON, exit 0. A missing config + present
    # checkpoints is a corrupt state → error out.
    config: DAVIConfig | None = None
    if config_path.exists():
        config = DAVIConfig.from_yaml(config_path)

    # Treat n_steps as 0 when config is missing — `_list_checkpoints` only
    # uses it as the "final" step label, and an empty dir won't surface one.
    n_steps = config.n_steps if config is not None else 0
    checkpoints = _list_checkpoints(run_dir, n_steps)

    if args.checkpoint_steps is not None:
        # Build the requested-step set. 'final' selects whatever step
        # net_final.pt resolves to via metrics.jsonl.
        requested_raw = [
            s.strip() for s in args.checkpoint_steps.split(",") if s.strip()
        ]
        final_step = _final_step_from_metrics(run_dir, fallback=n_steps)
        requested: set[int] = set()
        for s in requested_raw:
            if s == "final":
                requested.add(int(final_step))
            else:
                requested.add(int(s))
        filtered = [(step, path) for step, path in checkpoints if step in requested]
        missing = requested - {step for step, _ in filtered}
        if missing:
            raise ValueError(
                f"requested checkpoint steps not found in {run_dir}: {sorted(missing)}"
            )
        checkpoints = filtered

    if not checkpoints:
        out_path.write_text(json.dumps({}, indent=2))
        out_display = (
            out_path.relative_to(REPO_ROOT)
            if out_path.is_relative_to(REPO_ROOT)
            else out_path
        )
        print(
            f"no checkpoints in {run_dir}; wrote empty {out_display}",
            flush=True,
        )
        return 0

    if config is None:
        raise FileNotFoundError(
            f"{config_path} not found but checkpoints exist in {run_dir}; "
            "cannot reconstruct the value net without it"
        )

    device = torch.device(args.device or config.device)
    states_arr, depths_arr = load_v_star_bounded_3x3_arrays(CACHE_PATH)

    # When subsampling via --checkpoint-steps, merge into any existing
    # trajectory so successive subsample passes accumulate without
    # clobbering prior results.
    trajectory: dict[str, dict] = {}
    if args.checkpoint_steps is not None and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if isinstance(existing, dict):
                trajectory.update(existing)
        except json.JSONDecodeError:
            pass

    precision_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.precision]

    for step, ckpt_path in checkpoints:
        print(f"evaluating {ckpt_path.name} (step={step}) ...", flush=True)
        net = _load_net(ckpt_path, config, device)
        if precision_dtype is not torch.float32:
            net = net.to(precision_dtype)

        walk_metrics = beam_eval_walk(
            net,
            CUBE_3X3,
            n_per_depth=args.n_per_walk_depth,
            beam_width=args.beam_width,
        )
        v_star_metrics = beam_eval_v_star(
            net,
            CUBE_3X3,
            (states_arr, depths_arr),
            n_per_layer=args.n_per_v_star_layer,
            beam_width=args.beam_width,
        )
        trajectory[str(step)] = {
            "checkpoint": ckpt_path.name,
            "beam_walk": walk_metrics,
            "beam_v_star": v_star_metrics,
        }

    out_path.write_text(json.dumps(trajectory, indent=2))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
