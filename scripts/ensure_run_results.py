"""Idempotent wrapper for per-run beam-eval reporting.

Checks which checkpoints in a training run already have entries in
``<run-dir>/results/beam_eval_<config>.jsonl`` and only evaluates the
missing ones. Always regenerates the eval-trajectory HTML so it stays
in sync with the JSONL.

Use this when you want "make sure my reports are up to date" without
forcing a full re-eval. Use ``beam_eval_run.py`` directly when you do
want a full re-eval (e.g. after changing eval-config schema or
debugging the eval itself). The two scripts share an output format and
co-exist cleanly — re-running ``beam_eval_run.py`` over a populated
JSONL just merges (replaces existing entries by step).

Common workflow::

    uv run python scripts/ensure_run_results.py <run-dir>

Rendering scope is **per-run beam-eval only**. The multi-run
``error_trajectories.html`` report is global (driven by the ``RUNS``
table inside ``experiments/davi-3x3/analysis/render_error_trajectories.py``)
and is not touched here — append a tuple there and re-run that script
when you want a new run to appear in it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Match ``net_step_<int>.pt``. ``net_final.pt`` is handled specially
# (its step isn't in the filename — beam_eval_run.py resolves it via
# metrics.jsonl when --steps "final" is requested).
_STEP_RE = re.compile(r"^net_step_(\d+)\.pt$")


def _existing_steps_in_jsonl(jsonl_path: Path) -> set[int]:
    """Return the set of ``step`` values already present as JSONL lines."""
    if not jsonl_path.exists():
        return set()
    out: set[int] = set()
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = rec.get("step")
        if isinstance(s, int):
            out.add(s)
    return out


def _checkpoint_steps_in_dir(run_dir: Path) -> set[int]:
    """Return ``{step}`` for every ``net_step_<step>.pt`` in ``run_dir``.

    ``net_final.pt`` is NOT included here — the caller checks for it
    separately because its step isn't recoverable from the filename
    without loading metrics.jsonl or the bundle itself.
    """
    out: set[int] = set()
    if not run_dir.exists():
        return out
    for p in run_dir.iterdir():
        m = _STEP_RE.match(p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def _resolve_missing(
    run_dir: Path, jsonl_path: Path
) -> tuple[set[int], bool, set[int]]:
    """Return ``(missing_steps, final_pending, all_existing_steps)``.

    ``missing_steps``: ``net_step_<N>.pt`` present in dir but absent
    from the JSONL. ``final_pending``: ``net_final.pt`` exists in dir
    (we always re-eval it because we can't cheaply tell its step
    without loading the bundle; the JSONL merge dedups by step
    afterward at modest cost — at most one redundant eval per call).
    ``all_existing_steps``: the JSONL's current step set, surfaced for
    the summary print.
    """
    existing = _existing_steps_in_jsonl(jsonl_path)
    ckpt_steps = _checkpoint_steps_in_dir(run_dir)
    missing = ckpt_steps - existing
    final_pending = (run_dir / "net_final.pt").exists()
    return missing, final_pending, existing


def _invoke_beam_eval_run(
    run_dir: Path,
    eval_config: str,
    steps_csv: str | None,
    render_html: bool,
) -> int:
    """Shell out to beam_eval_run.py. Returns the subprocess return code."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "beam_eval_run.py"),
        str(run_dir),
        "--config",
        eval_config,
    ]
    if steps_csv is not None:
        cmd.extend(["--steps", steps_csv])
    if render_html:
        cmd.append("--render-html")
    print(f"→ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def _invoke_render_only(run_dir: Path, eval_config: str, jsonl_path: Path) -> int:
    """Re-render eval_trajectory_<config>.html from an existing JSONL.

    Used when nothing is missing — we still regenerate the HTML so it
    reflects any renderer-code changes since it was last produced.
    """
    out_html = run_dir / "results" / f"eval_trajectory_{eval_config}.html"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_beam_eval_report.py"),
        "--input",
        str(jsonl_path),
        "--output",
        str(out_html),
    ]
    print(f"→ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Training run directory.")
    parser.add_argument(
        "--eval-config",
        default="fast",
        help=(
            "Beam-eval config name (resolves to "
            "scripts/eval-configs/<name>.yaml). Default: fast."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the staleness check; re-eval every checkpoint.",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory", file=sys.stderr)
        return 1

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / f"beam_eval_{args.eval_config}.jsonl"

    if args.force:
        rc = _invoke_beam_eval_run(
            run_dir, args.eval_config, steps_csv=None, render_html=True
        )
        if rc != 0:
            return rc
    else:
        missing, final_pending, existing = _resolve_missing(run_dir, jsonl_path)
        if not missing and not final_pending:
            print(
                f"all {len(existing)} checkpoint(s) already evaluated; "
                "re-rendering HTML from existing JSONL"
            )
            if jsonl_path.exists():
                rc = _invoke_render_only(run_dir, args.eval_config, jsonl_path)
                if rc != 0:
                    return rc
        else:
            steps_parts: list[str] = sorted(str(s) for s in missing)
            if final_pending:
                steps_parts.append("final")
            steps_csv = ",".join(steps_parts)
            n_to_eval = len(missing) + (1 if final_pending else 0)
            print(
                f"{len(existing)} already evaluated; "
                f"evaluating {n_to_eval} new checkpoint(s): {steps_csv}"
            )
            rc = _invoke_beam_eval_run(
                run_dir, args.eval_config, steps_csv=steps_csv, render_html=True
            )
            if rc != 0:
                return rc

    def _display(p: Path) -> Path | str:
        return p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p

    # Summary: report what now exists.
    print()
    print(f"=== run results for {_display(run_dir)} ===")
    if jsonl_path.exists():
        final_count = len(_existing_steps_in_jsonl(jsonl_path))
        print(f"  {_display(jsonl_path)} ({final_count} record(s))")
    eval_html = results_dir / f"eval_trajectory_{args.eval_config}.html"
    if eval_html.exists():
        print(f"  {_display(eval_html)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
