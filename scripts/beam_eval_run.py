"""Run-dir wrapper for the beam-eval CLI.

Walks a training-run directory and invokes the same eval logic as
``beam_eval_model.py`` for each ``net_step_*.pt`` and ``net_final.pt``
checkpoint. Per-checkpoint JSON lands at
``<run-dir>/results/<step>_eval_<config-name>.json`` in the same flat
schema ``beam_eval_model.py`` emits — there is **no** trajectory rollup
JSON. Each checkpoint gets one independent file.

Common workflow::

    uv run python scripts/beam_eval_run.py <run-dir> --config fast

Filtering (mutually exclusive):

- ``--every-steps N`` — only checkpoints at multiples of N (plus
  net_final.pt always).
- ``--steps "10000,20000,final"`` — explicit step labels; literal
  ``"final"`` resolves to net_final.pt's step (read from metrics.jsonl
  or the training config's n_steps).

If neither is set, every checkpoint is evaluated.

This script is the post-refactor sibling of ``beam_eval_model.py`` —
the same shared core (``evaluate_checkpoint``) runs for every
checkpoint; this layer just walks the directory and resolves filtering.

**Empty-checkpoint behavior.** If the run dir contains no checkpoints
matching ``net_step_*.pt`` or ``net_final.pt``, this script writes an
empty JSON object (``{}``) to ``<run-dir>/results/post_run_beam_eval.json``
and exits 0. Tests rely on this contract — a freshly-created run dir
is a normal pre-training state, not an error. (Empty-state file path
deliberately keeps the old name so test fixtures stay valid; the
post-rename per-checkpoint path is only used when checkpoints exist.)
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
ORACLE_CACHE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"

# Make ``experiments/davi-3x3/eval.py`` and the scripts/ dir importable.
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_config import EvalConfig  # noqa: E402
from beam_eval_model import evaluate_checkpoint  # noqa: E402

from rubik.oracle.v_star_bounded_3x3 import (  # noqa: E402
    load_v_star_bounded_3x3_arrays,
)
from rubik.training.config import DAVIConfig  # noqa: E402

# Match ``net_step_<int>.pt``. ``net_final.pt`` is handled specially.
_STEP_RE = re.compile(r"^net_step_(\d+)\.pt$")


def _final_step_from_metrics(run_dir: Path, fallback: int) -> int:
    """Return the highest step recorded in metrics.jsonl, or ``fallback``.

    With early-stop, the run can terminate before ``config.n_steps``; in
    that case ``net_final.pt`` corresponds to the early-stop step, not
    ``n_steps``. Reading metrics.jsonl gives the truthful step label.
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


def _filter_checkpoints(
    checkpoints: list[tuple[int, Path]],
    *,
    every_steps: int | None,
    steps_csv: str | None,
    final_step: int,
) -> list[tuple[int, Path]]:
    """Apply --every-steps or --steps filter. Mutually exclusive upstream."""
    if every_steps is not None:
        # Multiples of N, plus net_final.pt always.
        kept: list[tuple[int, Path]] = []
        for step, path in checkpoints:
            if path.name == "net_final.pt" or step % every_steps == 0:
                kept.append((step, path))
        return kept
    if steps_csv is not None:
        requested_raw = [s.strip() for s in steps_csv.split(",") if s.strip()]
        requested: set[int] = set()
        for s in requested_raw:
            if s == "final":
                requested.add(int(final_step))
            else:
                requested.add(int(s))
        filtered = [(step, path) for step, path in checkpoints if step in requested]
        missing = requested - {step for step, _ in filtered}
        if missing:
            raise ValueError(f"requested checkpoint steps not found: {sorted(missing)}")
        return filtered
    return checkpoints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training run directory (contains config.yaml + checkpoints).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help=(
            "Eval config name (resolves to scripts/eval-configs/<name>.yaml) "
            "or a literal filesystem path. Default: 'default'."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Slice n_per_depth[:N] from the config.",
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
        help="Force include_v_star=True (overrides config).",
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
        help="Override training config.device.",
    )

    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--every-steps",
        type=int,
        default=None,
        help=(
            "Only evaluate checkpoints at multiples of this step (plus "
            "net_final.pt always)."
        ),
    )
    selector.add_argument(
        "--steps",
        type=str,
        default=None,
        help=(
            "Comma-separated step labels (e.g. '20000,30000,final'). "
            "'final' resolves to net_final.pt's step."
        ),
    )
    parser.add_argument(
        "--render-html",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override config.render_html. When enabled, after all evals "
            "complete render a SINGLE overlay HTML across the per-checkpoint "
            "JSONs at <run-dir>/results/trajectory_<config>.html. (Per-model "
            "renders are suppressed in run mode — one overlay is the point.) "
            "When omitted, the YAML config's render_html field decides."
        ),
    )

    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    config_path = run_dir / "config.yaml"
    results_dir = run_dir / "results"
    empty_state_path = results_dir / "post_run_beam_eval.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    # If config.yaml is absent, behavior depends on whether checkpoints
    # exist: a missing config + no checkpoints is "fresh run dir, nothing
    # to eval" → write empty JSON, exit 0. A missing config + present
    # checkpoints is a corrupt state → error out.
    training_config: DAVIConfig | None = None
    if config_path.exists():
        training_config = DAVIConfig.from_yaml(config_path)

    n_steps = training_config.n_steps if training_config is not None else 0
    checkpoints = _list_checkpoints(run_dir, n_steps)
    final_step = _final_step_from_metrics(run_dir, fallback=n_steps)

    if not checkpoints:
        empty_state_path.write_text(json.dumps({}, indent=2))
        out_display = (
            empty_state_path.relative_to(REPO_ROOT)
            if empty_state_path.is_relative_to(REPO_ROOT)
            else empty_state_path
        )
        print(
            f"no checkpoints in {run_dir}; wrote empty {out_display}",
            flush=True,
        )
        return 0

    if training_config is None:
        raise FileNotFoundError(
            f"{config_path} not found but checkpoints exist in {run_dir}; "
            "cannot reconstruct the value net without it"
        )

    checkpoints = _filter_checkpoints(
        checkpoints,
        every_steps=args.every_steps,
        steps_csv=args.steps,
        final_step=final_step,
    )

    # Build the EvalConfig once; CLI overrides apply to all checkpoints.
    eval_cfg = EvalConfig.resolve(args.config).with_overrides(
        beam_width=args.width,
        precision=args.precision,
        seed=args.seed,
        include_v_star=True if args.include_v_star else None,
        max_depth=args.max_depth,
        render_html=args.render_html,
    )

    device = torch.device(args.device or training_config.device)

    # Pre-load oracle once if needed (avoids re-reading the .npz per checkpoint).
    oracle_arrays = None
    if eval_cfg.include_v_star:
        if not ORACLE_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"--include-v-star requires {ORACLE_CACHE_PATH} (run "
                "scripts/build_v_star_bounded_3x3.py first)."
            )
        oracle_arrays = load_v_star_bounded_3x3_arrays(ORACLE_CACHE_PATH)

    written: list[Path] = []
    for step, ckpt_path in checkpoints:
        # Output filename uses the step label, not the checkpoint stem, so
        # net_final.pt resolves to its early-stop-aware step in the name.
        out_path = results_dir / f"step_{step}_eval_{eval_cfg.source_name}.json"
        print(
            f"evaluating {ckpt_path.name} (step={step}) ...",
            flush=True,
        )
        payload = evaluate_checkpoint(
            model_path=ckpt_path,
            eval_cfg=eval_cfg,
            device=device,
            training_config=training_config,
            oracle_arrays=oracle_arrays,
        )
        # Annotate with the step label so per-checkpoint JSONs carry the
        # context they came from without needing a sibling rollup.
        payload["step"] = int(step)
        out_path.write_text(json.dumps(payload, indent=2))
        written.append(out_path)
        deep_rate = (
            payload["per_walk_depth"][-1]["solve_rate"]
            if payload["per_walk_depth"]
            else float("nan")
        )
        print(
            f"  wrote {out_path.name} "
            f"(wall={payload['wall_time_seconds']:.1f}s "
            f"solve_rate@d={eval_cfg.max_depth}={deep_rate:.3f})",
            flush=True,
        )

    print(f"\n{len(written)} checkpoint JSON(s) written under {results_dir}")

    if eval_cfg.render_html and written:
        import subprocess

        renderer = REPO_ROOT / "scripts" / "render_beam_eval_report.py"
        html_path = results_dir / f"trajectory_{eval_cfg.source_name}.html"
        subprocess.run(
            [
                sys.executable,
                str(renderer),
                "--input",
                *(str(p) for p in written),
                "--output",
                str(html_path),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        print(f"rendered {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
