"""Analyze each DAVI 3x3 run's metrics.jsonl and write a results.md skeleton.

Iterates over the canonical ``RUNS`` list and produces, per run:

- A train-loss curve summary (start / end / min).
- An eval trajectory: every eval step, corrected_macro and
  pred_mean/pred_std values.
- A per-V*-layer MAE table at start (first eval), middle, and end
  (final), drawn from the ``v_star_mae/d{1..6}`` keys emitted by the
  bounded BFS oracle (K=6).

Note: the 3x3 ``value_eval`` schema does NOT emit live
``solve_rate_dN`` / ``avg_solve_len_dN`` keys (per plan §P1c — beam cost
would slow training). The capability picture lives in
``<run>/results/beam_eval_focused.json`` and is rendered by
``render_error_trajectories.py``, not summarized here.

Per project convention, we do *not* auto-generate ``intuition.md`` —
that's hand-written, then appended at the end of ``results.md`` so the
hypothesis-and-verification reasoning persists across re-runs of this
analyzer.

Usage::

    uv run python experiments/davi-3x3/analysis/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = EXPERIMENT_DIR / "runs"
RESULTS_PATH = EXPERIMENT_DIR / "results" / "results.md"
INTUITION_PATH = EXPERIMENT_DIR / "results" / "intuition.md"

# (label, run_subdir) — keep in sync with the renderer.
RUNS: list[tuple[str, str]] = [
    ("smoke", "20260506T203408Z_smoke"),
]

# V* layer range emitted by the bounded oracle (K=6). Layer 0 is dropped
# from corrected_macro because some evals in earlier runs erroneously
# included v_star_mae/d0 in the recorded macro_v_star_mae.
V_STAR_DEPTHS = list(range(1, 7))


def _load_records(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found at {metrics_path}")
    records: list[dict] = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _summarize_loss(records: list[dict]) -> dict:
    losses = [(r["step"], r["loss"]) for r in records if r.get("event") == "step"]
    if not losses:
        return {"start": None, "end": None, "min": None, "min_step": None}
    start_loss = losses[0][1]
    end_loss = losses[-1][1]
    min_step, min_loss = min(losses, key=lambda t: t[1])
    return {
        "start": start_loss,
        "end": end_loss,
        "min": min_loss,
        "min_step": min_step,
    }


def _eval_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("event") == "eval"]


def _corrected_macro(rec: dict) -> float | None:
    """Mean of v_star_mae/dN for N>=1 (drops contaminated d=0)."""
    vals = []
    for d in V_STAR_DEPTHS:
        v = rec.get(f"v_star_mae/d{d}")
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _format_macro_trajectory(eval_recs: list[dict]) -> str:
    lines = [
        "| step | corrected_macro | pred_mean | pred_std |",
        "|-----:|-----:|-----:|-----:|",
    ]
    for rec in eval_recs:
        step = rec["step"]
        cm = _corrected_macro(rec)
        cm_s = f"{cm:.4f}" if cm is not None else "—"
        pm = rec.get("pred_mean")
        ps = rec.get("pred_std")
        pm_s = f"{pm:.3f}" if pm is not None else "—"
        ps_s = f"{ps:.3f}" if ps is not None else "—"
        lines.append(f"| {step} | {cm_s} | {pm_s} | {ps_s} |")
    return "\n".join(lines)


def _format_per_v_star_table(eval_recs: list[dict]) -> str:
    if not eval_recs:
        return "_(no eval records)_"
    pick_recs = []
    if eval_recs:
        pick_recs.append(("start", eval_recs[0]))
    if len(eval_recs) >= 3:
        pick_recs.append(("middle", eval_recs[len(eval_recs) // 2]))
    if len(eval_recs) >= 2:
        pick_recs.append(("end", eval_recs[-1]))

    header = (
        "| V* depth | "
        + " | ".join(f"{label} (step {rec['step']})" for label, rec in pick_recs)
        + " |"
    )
    sep = "|------:|" + "|".join(["------:"] * len(pick_recs)) + "|"
    lines = [header, sep]
    for d in V_STAR_DEPTHS:
        cells = []
        for _label, rec in pick_recs:
            v = rec.get(f"v_star_mae/d{d}")
            cells.append(f"{v:.3f}" if v is not None else "—")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _section_for_run(label: str, run_subdir: str) -> list[str]:
    """Render the per-run subsection. Returns a list of markdown lines."""
    run_dir = RUNS_DIR / run_subdir
    metrics_path = run_dir / "metrics.jsonl"

    md: list[str] = []
    md.append(f"## {label}")
    md.append("")

    if not metrics_path.exists():
        rel = metrics_path.relative_to(REPO_ROOT)
        md.append(f"_(metrics.jsonl not found at `{rel}`)_")
        md.append("")
        return md

    records = _load_records(metrics_path)
    eval_recs = _eval_records(records)
    loss_summary = _summarize_loss(records)

    md.append(
        f"_Run dir: `{run_dir.relative_to(REPO_ROOT)}`. "
        f"Records: {len(records)} total, {len(eval_recs)} eval cycles._"
    )
    md.append("")

    md.append("### Train loss")
    md.append("")
    if loss_summary["start"] is None:
        md.append("_(no step records)_")
    else:
        md.append(
            f"- start: {loss_summary['start']:.4f}\n"
            f"- end:   {loss_summary['end']:.4f}\n"
            f"- min:   {loss_summary['min']:.4f} "
            f"(step {loss_summary['min_step']})"
        )
    md.append("")

    md.append("### Eval corrected-macro trajectory")
    md.append("")
    md.append(_format_macro_trajectory(eval_recs))
    md.append("")

    md.append("### Per-V*-layer MAE — start / middle / end (bounded oracle, d=1..6)")
    md.append("")
    md.append(_format_per_v_star_table(eval_recs))
    md.append("")

    md.append("### Beam capability")
    md.append("")
    md.append(
        "_(Live solve_rate / avg_solve_len are not captured during 3x3 "
        "training. See `<run>/results/beam_eval_focused.json` and the "
        "Beam capability section of `error_trajectories.html`.)_"
    )
    md.append("")

    return md


def main() -> None:
    md_lines: list[str] = []
    md_lines.append("# DAVI 3x3 — per-run results")
    md_lines.append("")
    md_lines.append(
        "_One section per run: train loss, eval corrected-macro trajectory, and "
        "per-V*-layer MAE at start / middle / end. Cross-run views (overlaid "
        "charts, beam capability bars, heatmap) live in "
        "`error_trajectories.html`._"
    )
    md_lines.append("")

    if not RUNS:
        md_lines.append(
            "_(No runs registered yet. Append to `RUNS` in this script when a "
            "new run lands.)_"
        )
        md_lines.append("")
    else:
        for label, run_subdir in RUNS:
            md_lines.extend(_section_for_run(label, run_subdir))

    md = "\n".join(md_lines)

    if INTUITION_PATH.exists():
        md += "\n\n---\n\n"
        md += INTUITION_PATH.read_text()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(md)
    print(f"results.md written to {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
