"""Analyze each DAVI run's metrics.jsonl and write a results.md skeleton.

Iterates over the canonical ``RUNS`` list (cycle-1 baseline plus the
cycle-3/cycle-4 follow-ups) and produces, per run:

- A train-loss curve summary (start / end / min).
- A V* macro-MAE trajectory: every eval step, macro_mae and val_mae values.
- A per-depth MAE table at start (first eval), middle, and end (final).
- A greedy-solve rate trajectory per test depth.

The same shape that the original single-run analyzer produced — just
emitted once per run, under an H2 section keyed on the run's label. The
file as a whole stays cold-readable; flipping between runs is a scroll
rather than a re-run.

Per project convention, we do *not* auto-generate ``intuition.md`` —
that's hand-written, then appended at the end of ``results.md`` so the
hypothesis-and-verification reasoning persists across re-runs of this
analyzer.

Usage::

    uv run python experiments/davi-2x2/analysis/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = EXPERIMENT_DIR / "runs"
RESULTS_PATH = EXPERIMENT_DIR / "results" / "results.md"
INTUITION_PATH = EXPERIMENT_DIR / "results" / "intuition.md"

# (label, run_subdir) — keep in sync with the renderer + capture script.
RUNS: list[tuple[str, str]] = [
    ("cycle-1 baseline-30k  (K=18, sync=500)", "baseline-30k"),
    ("cycle-3 sync500_kmax20-30k  (K=20)", "sync500_kmax20-30k"),
    ("cycle-3 sync1000_kmax20-30k  (K=20)", "sync1000_kmax20-30k"),
    ("cycle-4 kmax28_warm-30k  (K=28, warm-start)", "kmax28_warm-30k"),
]


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


def _solve_depth_keys(eval_recs: list[dict]) -> list[int]:
    if not eval_recs:
        return []
    rec = eval_recs[0]
    depths: list[int] = []
    for k in rec:
        if k.startswith("solve_rate_d"):
            depths.append(int(k[len("solve_rate_d") :]))
    return sorted(depths)


def _format_macro_mae_trajectory(eval_recs: list[dict]) -> str:
    lines = [
        "| step | train_loss_recent | macro_mae | val_mae | pred_mean | pred_std |",
        "|-----:|-----:|-----:|-----:|-----:|-----:|",
    ]
    for rec in eval_recs:
        step = rec["step"]
        # train_loss_recent intentionally not joined here — the curve
        # summary sits in its own section. Just dash-fill to keep table
        # mechanically simple.
        lines.append(
            f"| {step} | — | {rec['macro_mae']:.4f} | "
            f"{rec['val_mae']:.4f} | {rec['pred_mean']:.3f} | {rec['pred_std']:.3f} |"
        )
    return "\n".join(lines)


def _format_per_depth_table(eval_recs: list[dict]) -> str:
    if not eval_recs:
        return "_(no eval records)_"
    pick_recs = []
    if eval_recs:
        pick_recs.append(("start", eval_recs[0]))
    if len(eval_recs) >= 3:
        pick_recs.append(("middle", eval_recs[len(eval_recs) // 2]))
    if len(eval_recs) >= 2:
        pick_recs.append(("end", eval_recs[-1]))

    # Collect all depths seen across the picked records.
    all_depths = set()
    for _label, rec in pick_recs:
        all_depths.update(int(d) for d in rec["per_depth_mae"])
    sorted_depths = sorted(all_depths)

    header = (
        "| depth | "
        + " | ".join(f"{label} (step {rec['step']})" for label, rec in pick_recs)
        + " |"
    )
    sep = "|------:|" + "|".join(["------:"] * len(pick_recs)) + "|"
    lines = [header, sep]
    for d in sorted_depths:
        cells = []
        for _label, rec in pick_recs:
            v = rec["per_depth_mae"].get(str(d))
            if v is None:
                # JSON keys can come back as int when round-tripped via
                # python; per_depth_mae was built with int() keys, but
                # json serializes them as strings.
                v = rec["per_depth_mae"].get(d)
            cells.append(f"{v:.3f}" if v is not None else "—")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_solve_trajectory(eval_recs: list[dict]) -> str:
    depths = _solve_depth_keys(eval_recs)
    if not depths:
        return "_(no greedy-solve records)_"
    header = (
        "| step | "
        + " | ".join(f"d{d} rate" for d in depths)
        + " | "
        + " | ".join(f"d{d} avg_len" for d in depths)
        + " |"
    )
    sep = "|-----:|" + "|".join(["-----:"] * (len(depths) * 2)) + "|"
    lines = [header, sep]
    for rec in eval_recs:
        rates = [f"{rec.get(f'solve_rate_d{d}', float('nan')):.2f}" for d in depths]
        lens = []
        for d in depths:
            v = rec.get(f"avg_solve_len_d{d}")
            lens.append(f"{v:.2f}" if v is not None else "—")
        lines.append(
            f"| {rec['step']} | " + " | ".join(rates) + " | " + " | ".join(lens) + " |"
        )
    return "\n".join(lines)


def _section_for_run(label: str, run_subdir: str) -> list[str]:
    """Render the per-run subsection. Returns a list of markdown lines."""
    run_dir = RUNS_DIR / run_subdir
    metrics_path = run_dir / "metrics.jsonl"

    md: list[str] = []
    md.append(f"## {label}")
    md.append("")

    if not metrics_path.exists():
        md.append(f"_(metrics.jsonl not found at `{metrics_path.relative_to(REPO_ROOT)}`)_")
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

    md.append("### V* macro-MAE trajectory")
    md.append("")
    md.append(_format_macro_mae_trajectory(eval_recs))
    md.append("")

    md.append("### Per-depth MAE — start / middle / end")
    md.append("")
    md.append(_format_per_depth_table(eval_recs))
    md.append("")

    md.append("### Greedy-policy solve rate trajectory")
    md.append("")
    md.append(_format_solve_trajectory(eval_recs))
    md.append("")

    return md


def main() -> None:
    md_lines: list[str] = []
    md_lines.append("# DAVI 2x2 — per-run results")
    md_lines.append("")
    md_lines.append(
        "_One section per run: train loss, V* macro-MAE trajectory, per-depth "
        "MAE at start / middle / end, and greedy-solve trajectory. The "
        "side-by-side cross-run views (overlaid charts, histograms, wavefront "
        "heatmap) live in `error_trajectories.html`._"
    )
    md_lines.append("")

    for label, run_subdir in RUNS:
        md_lines.extend(_section_for_run(label, run_subdir))

    md = "\n".join(md_lines)

    if INTUITION_PATH.exists():
        md += "\n\n---\n\n"
        md += INTUITION_PATH.read_text()

    RESULTS_PATH.write_text(md)
    print(f"results.md written to {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
