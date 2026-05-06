"""Analyze beam-search-2x2 sweep + acceptance runs and write results.md.

Reads ``runs/sync500_kmax20-baseline/metrics.jsonl`` (and optionally a
second acceptance run dir) and produces:

- A run header (checkpoint, n_params, device, n_cells, wall time).
- Per-(depth, beam_width) pivot tables: solve_rate, avg_solve_len,
  mean_v_star_excess, n_expansions, elapsed_seconds.
- An "acceptance shape" summary: per-depth solve_rate at the max width and
  per-depth mean V*-excess at the max width — the cells that the C5
  acceptance run scales up.
- (If ``--acceptance-dir`` is passed) a dedicated "Acceptance gate"
  section with per-depth pass/fail vs the SPEC's ≥0.999 solve-rate gate
  and ≤1.0 mean V*-excess gate.

Per project convention, ``intuition.md`` is hand-written and appended at
the end of ``results.md`` so the hypothesis-and-verification reasoning
persists across re-runs of this analyzer.

Usage::

    uv run python experiments/beam-search-2x2/analyze.py \\
        [--run-dir runs/<sweep-name>] \\
        [--acceptance-dir runs/<acceptance-name>]

If ``--run-dir`` is omitted, defaults to ``runs/sync500_kmax20-baseline``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "sync500_kmax20-baseline"
DEFAULT_ACCEPTANCE_DIR = EXPERIMENT_DIR / "runs" / "sync500_kmax20-acceptance"
RESULTS_PATH = EXPERIMENT_DIR / "results.md"
INTUITION_PATH = EXPERIMENT_DIR / "intuition.md"

# SPEC gates (from plans/m6-beam-search.md and SPEC.md):
SOLVE_RATE_GATE = 0.999  # per-depth threshold (≤1 fail per 1000)
V_STAR_EXCESS_GATE = 1.0  # mean over all depths


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


def _cell_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("event") == "cell"]


def _grid_axes(cells: list[dict]) -> tuple[list[int], list[int]]:
    depths = sorted({int(c["depth"]) for c in cells})
    widths = sorted({int(c["beam_width"]) for c in cells})
    return depths, widths


def _index(cells: list[dict]) -> dict[tuple[int, int], dict]:
    return {(int(c["depth"]), int(c["beam_width"])): c for c in cells}


def _format_pivot(
    cells: list[dict],
    *,
    field: str,
    fmt: str,
    none_label: str = "—",
) -> str:
    depths, widths = _grid_axes(cells)
    idx = _index(cells)
    header = "| depth \\ width | " + " | ".join(f"w={w}" for w in widths) + " |"
    sep = "|---:|" + "|".join(["---:"] * len(widths)) + "|"
    lines = [header, sep]
    for d in depths:
        row_cells = []
        for w in widths:
            c = idx.get((d, w))
            if c is None:
                row_cells.append(none_label)
                continue
            v = c.get(field)
            if v is None:
                row_cells.append(none_label)
            else:
                row_cells.append(format(v, fmt))
        lines.append(f"| d={d} | " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def _format_max_width_summary(cells: list[dict]) -> str:
    depths, widths = _grid_axes(cells)
    if not widths:
        return "_(no cells)_"
    max_w = max(widths)
    idx = _index(cells)
    lines = [
        f"At `beam_width={max_w}` (sweep's largest cell — surrogate for the "
        "C5 acceptance gate):",
        "",
        "| depth | n_solved | solve_rate | avg_solve_len | mean_excess | elapsed_s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for d in depths:
        c = idx.get((d, max_w))
        if c is None:
            lines.append(f"| d={d} | — | — | — | — | — |")
            continue
        avg_len = c.get("avg_solve_len")
        excess = c.get("mean_v_star_excess")
        avg_len_s = "—" if avg_len is None else f"{avg_len:.3f}"
        excess_s = "—" if excess is None else f"{excess:.3f}"
        lines.append(
            f"| d={d} | {c['n_solved']}/{c['n_per_depth']} | "
            f"{c['solve_rate']:.3f} | {avg_len_s} | {excess_s} | "
            f"{c['elapsed_seconds']:.1f} |"
        )
    return "\n".join(lines)


def _format_acceptance_section(acceptance_dir: Path) -> str:
    """Per-depth pass/fail vs SPEC gates from a dedicated acceptance run."""
    metrics_path = acceptance_dir / "metrics.jsonl"
    records = _load_records(metrics_path)
    cells = _cell_records(records)
    if not cells:
        return "_(no cells in acceptance run)_"
    starts = [r for r in records if r.get("event") == "run_start"]
    ends = [r for r in records if r.get("event") == "run_end"]
    s = starts[0] if starts else None
    e = ends[0] if ends else None

    n_per_depth = cells[0]["n_per_depth"]
    beam_widths = sorted({int(c["beam_width"]) for c in cells})
    if len(beam_widths) != 1:
        beam_widths_label = f"{beam_widths}"
    else:
        beam_widths_label = f"{beam_widths[0]}"

    lines: list[str] = [
        f"_Run dir: `{acceptance_dir.relative_to(REPO_ROOT)}`._",
        "",
        f"- **beam_width:** {beam_widths_label}",
        f"- **n_per_depth:** {n_per_depth}",
    ]
    if s is not None:
        lines.append(f"- **max_steps:** {s['max_steps']}")
    if e is not None:
        lines.append(f"- **wall:** {e['elapsed_seconds']:.1f}s")
    lines.append("")
    lines.append(
        f"**SPEC gates:** per-depth `solve_rate ≥ {SOLVE_RATE_GATE}` "
        f"(≤{int(round((1 - SOLVE_RATE_GATE) * n_per_depth))} fail per "
        f"{n_per_depth}-scramble cell) AND mean `v_star_excess ≤ "
        f"{V_STAR_EXCESS_GATE}` across all depths."
    )
    lines.append("")
    lines.append(
        "| depth | n_solved | solve_rate | gate | mean_excess | "
        "avg_solve_len | elapsed_s |"
    )
    lines.append("|---:|---:|---:|:---:|---:|---:|---:|")
    excesses = []
    n_pass = 0
    n_total_depths = 0
    for c in sorted(cells, key=lambda c: int(c["depth"])):
        n_total_depths += 1
        d = int(c["depth"])
        rate = c["solve_rate"]
        passed = rate >= SOLVE_RATE_GATE
        if passed:
            n_pass += 1
        gate_marker = "✅" if passed else "❌"
        excess = c.get("mean_v_star_excess")
        avg_len = c.get("avg_solve_len")
        if excess is not None:
            excesses.append(excess)
        excess_s = "—" if excess is None else f"{excess:.3f}"
        avg_len_s = "—" if avg_len is None else f"{avg_len:.3f}"
        lines.append(
            f"| d={d} | {c['n_solved']}/{c['n_per_depth']} | {rate:.3f} | "
            f"{gate_marker} | {excess_s} | {avg_len_s} | "
            f"{c['elapsed_seconds']:.1f} |"
        )
    mean_excess = sum(excesses) / len(excesses) if excesses else None
    excess_pass = mean_excess is not None and mean_excess <= V_STAR_EXCESS_GATE

    lines.append("")
    lines.append("**Verdict:**")
    lines.append("")
    lines.append(
        f"- Solve-rate gate: {n_pass}/{n_total_depths} depths pass "
        f"(per-depth `≥ {SOLVE_RATE_GATE}`)."
    )
    if mean_excess is not None:
        excess_marker = "✅ PASS" if excess_pass else "❌ FAIL"
        lines.append(
            f"- Mean V*-excess gate: {mean_excess:.3f} vs "
            f"`≤ {V_STAR_EXCESS_GATE}` — {excess_marker}."
        )
    overall = "✅ PASS" if (n_pass == n_total_depths and excess_pass) else "❌ FAIL"
    lines.append(f"- **Overall acceptance: {overall}**")
    return "\n".join(lines)


def _format_run_header(records: list[dict], run_dir: Path) -> str:
    starts = [r for r in records if r.get("event") == "run_start"]
    ends = [r for r in records if r.get("event") == "run_end"]
    cells = _cell_records(records)
    if not starts:
        return "_(no run_start record)_"
    s = starts[0]
    parts = [
        f"_Run dir: `{run_dir.relative_to(REPO_ROOT)}`._",
        "",
        f"- **checkpoint:** `{s['checkpoint_path']}`",
        f"- **device:** `{s['device']}`",
        f"- **params:** {s['n_params'] / 1e6:.2f}M",
        f"- **arch:** `body_widths={s['body_widths']}`, "
        f"`n_residual_blocks={s['n_residual_blocks']}`, "
        f"`normalization={s['normalization']}`",
        f"- **grid:** depths={s['depths']} × beam_widths={s['beam_widths']} "
        f"× n_per_depth={s['n_per_depth']} = {len(cells)} cells",
        f"- **max_steps:** {s['max_steps']}",
    ]
    if ends:
        parts.append(f"- **wall:** {ends[0]['elapsed_seconds']:.1f}s")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--acceptance-dir",
        type=Path,
        default=DEFAULT_ACCEPTANCE_DIR,
        help="Acceptance run dir; if its metrics.jsonl is missing, the "
        "section is skipped (sweep-only mode).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    metrics_path = run_dir / "metrics.jsonl"
    records = _load_records(metrics_path)
    cells = _cell_records(records)

    md: list[str] = []
    md.append("# beam-search-2x2 — results")
    md.append("")
    md.append("## Sweep run (C3)")
    md.append("")
    md.append(_format_run_header(records, run_dir))
    md.append("")

    md.append("## Solve rate (depth × beam_width)")
    md.append("")
    md.append(_format_pivot(cells, field="solve_rate", fmt=".3f"))
    md.append("")

    md.append("## Mean V*-excess (depth × beam_width)")
    md.append("")
    md.append(
        "_Mean over solved attempts only. Below the M6 acceptance gate of "
        "`mean_v_star_excess ≤ 1.0` per depth = good._"
    )
    md.append("")
    md.append(_format_pivot(cells, field="mean_v_star_excess", fmt=".3f"))
    md.append("")

    md.append("## Avg solve length (depth × beam_width)")
    md.append("")
    md.append(
        "_For solved attempts only. Compare to depth d (random scramble "
        "length) and to V*-optimal length (= avg_solve_len − mean_v_star_excess)._"
    )
    md.append("")
    md.append(_format_pivot(cells, field="avg_solve_len", fmt=".3f"))
    md.append("")

    md.append("## Cost (n_expansions, elapsed seconds)")
    md.append("")
    md.append("**n_expansions per cell:**")
    md.append("")
    md.append(_format_pivot(cells, field="n_expansions", fmt=","))
    md.append("")
    md.append("**Elapsed seconds per cell:**")
    md.append("")
    md.append(_format_pivot(cells, field="elapsed_seconds", fmt=".2f"))
    md.append("")

    md.append("## Acceptance-shape summary (max-width column)")
    md.append("")
    md.append(_format_max_width_summary(cells))
    md.append("")

    md.append("## Charts")
    md.append("")
    md.append(
        "See `beam_curves.html` — per-depth small multiples (5/5/4 banding) "
        "of solve-rate, V*-excess, and avg-solve-len vs beam_width, plus the "
        "(depth × beam_width) heatmap overview."
    )
    md.append("")

    acceptance_metrics = args.acceptance_dir / "metrics.jsonl"
    if acceptance_metrics.exists():
        md.append("## Acceptance gate (C5)")
        md.append("")
        md.append(_format_acceptance_section(args.acceptance_dir))
        md.append("")

    if INTUITION_PATH.exists():
        md.append("---")
        md.append("")
        md.append(INTUITION_PATH.read_text())

    RESULTS_PATH.write_text("\n".join(md))
    print(f"results.md written to {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
