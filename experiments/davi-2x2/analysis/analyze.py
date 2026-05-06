"""Analyze the davi-baseline 30k run and write a results.md skeleton.

Reads ``runs/baseline-30k/metrics.jsonl`` and produces:

- A train-loss curve summary (start / end / min).
- macro-MAE trajectory: every eval step, macro_mae and val_mae values.
- A per-depth MAE table at start (first eval), middle, and end (final).
- A greedy-solve rate trajectory per test depth.

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
RUN_DIR = EXPERIMENT_DIR / "runs" / "baseline-30k"
METRICS_PATH = RUN_DIR / "metrics.jsonl"
RESULTS_PATH = EXPERIMENT_DIR / "results" / "results.md"
INTUITION_PATH = EXPERIMENT_DIR / "results" / "intuition.md"


def _load_records() -> list[dict]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics.jsonl not found at {METRICS_PATH}")
    records: list[dict] = []
    for line in METRICS_PATH.read_text().splitlines():
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


def main() -> None:
    records = _load_records()
    eval_recs = _eval_records(records)
    loss_summary = _summarize_loss(records)

    md_lines: list[str] = []
    md_lines.append("# davi-baseline — 30k DAVI run results")
    md_lines.append("")
    md_lines.append(
        f"_Run dir: `{RUN_DIR.relative_to(REPO_ROOT)}`. "
        f"Records: {len(records)} total, {len(eval_recs)} eval cycles._"
    )
    md_lines.append("")

    md_lines.append("## Train loss")
    md_lines.append("")
    if loss_summary["start"] is None:
        md_lines.append("_(no step records)_")
    else:
        md_lines.append(
            f"- start: {loss_summary['start']:.4f}\n"
            f"- end:   {loss_summary['end']:.4f}\n"
            f"- min:   {loss_summary['min']:.4f} "
            f"(step {loss_summary['min_step']})"
        )
    md_lines.append("")

    md_lines.append("## V* macro-MAE trajectory")
    md_lines.append("")
    md_lines.append(_format_macro_mae_trajectory(eval_recs))
    md_lines.append("")

    md_lines.append("## Per-depth MAE — start / middle / end")
    md_lines.append("")
    md_lines.append(_format_per_depth_table(eval_recs))
    md_lines.append("")

    md_lines.append("## Greedy-policy solve rate trajectory")
    md_lines.append("")
    md_lines.append(_format_solve_trajectory(eval_recs))
    md_lines.append("")

    md = "\n".join(md_lines)

    if INTUITION_PATH.exists():
        md += "\n\n---\n\n"
        md += INTUITION_PATH.read_text()

    RESULTS_PATH.write_text(md)
    print(f"results.md written to {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
