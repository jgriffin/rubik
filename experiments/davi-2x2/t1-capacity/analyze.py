"""T1 analyzer: aggregate Phase A / B run logs and pick (h1*, h2*, n*).

Reads ``experiments/davi-2x2/runs/<ts>_phase{A,B}_*/metrics.jsonl`` and
produces:

- A printed table sorted by n_params (smallest first).
- ``experiments/davi-2x2/t1-capacity/results.md`` — Phase A + Phase B
  tables, Pareto frontier plot, chosen ``(h1*, h2*, n*)``.
- ``experiments/davi-2x2/t1-capacity/_picks.json`` — machine-readable:
  chosen ``(h1*, h2*, n*)`` plus a ``T1 comfortable`` (one size up) for
  later tiers that want headroom.
- ``experiments/davi-2x2/t1-capacity/plots/phaseA_pareto.html`` —
  hand-rolled SVG plot of params vs val_mae (no matplotlib).

Usage::

    uv run python experiments/davi-2x2/t1-capacity/analyze.py

The closing condition for Phase A: smallest cell with ``val_mae < 0.5``.
If nothing passes, we stop and debug instead of opening Phase B.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "experiments" / "davi-2x2" / "runs"
TIER_DIR = REPO_ROOT / "experiments" / "davi-2x2" / "t1-capacity"

PASS_THRESHOLD = 0.5  # methodology §T1


@dataclass
class CellResult:
    config_stem: str
    phase: str  # "A" or "B"
    body_widths: tuple[int, int]
    n_residual_blocks: int
    n_params: int
    final_val_mae: float
    run_dir: Path
    eval_curve: list[tuple[int, float]]  # (step, val_mae)


def _latest_run_per_config(phase: str) -> dict[str, Path]:
    """Map config stem → latest run dir for the given phase."""
    pattern = re.compile(rf"^\d{{8}}T\d{{6}}Z_(phase{phase}_\d+_\S+)$")
    by_config: dict[str, Path] = {}
    for run in RUNS_DIR.iterdir():
        if not run.is_dir():
            continue
        m = pattern.match(run.name)
        if not m:
            continue
        stem = m.group(1)
        # Pick the latest by directory name (timestamp is sortable)
        if stem not in by_config or run.name > by_config[stem].name:
            by_config[stem] = run
    return by_config


def _load_run(run_dir: Path, phase: str, config_stem: str) -> CellResult | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    body_widths: tuple[int, int] | None = None
    n_residual_blocks: int | None = None
    n_params = 0
    final_val_mae: float | None = None
    eval_curve: list[tuple[int, float]] = []

    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ev = rec.get("event")
        if ev == "run_start":
            n_params = rec.get("n_params", 0)
            bw = rec.get("body_widths")
            if bw is not None:
                body_widths = tuple(bw)  # type: ignore[assignment]
            n_residual_blocks = rec.get("n_residual_blocks")
        elif ev == "eval":
            eval_curve.append((rec["step"], rec["val_mae"]))
            if rec.get("final"):
                final_val_mae = rec["val_mae"]
        elif ev == "run_end":
            final_val_mae = rec.get("final_val_mae", final_val_mae)

    if body_widths is None or n_residual_blocks is None or final_val_mae is None:
        return None

    return CellResult(
        config_stem=config_stem,
        phase=phase,
        body_widths=body_widths,
        n_residual_blocks=n_residual_blocks,
        n_params=n_params,
        final_val_mae=final_val_mae,
        run_dir=run_dir,
        eval_curve=eval_curve,
    )


def _format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def _format_table(results: list[CellResult]) -> str:
    """Render a markdown table, sorted by n_params ascending."""
    rows = sorted(results, key=lambda r: r.n_params)
    lines = [
        "| cell | h1 × h2 | n_res | n_params | final val_mae | passes (<0.5) |",
        "|------|---------|-------|----------|--------------:|:-------------:|",
    ]
    for r in rows:
        h1, h2 = r.body_widths
        passes = "✓" if r.final_val_mae < PASS_THRESHOLD else "✗"
        lines.append(
            f"| {r.config_stem} | {h1} × {h2} | {r.n_residual_blocks} | "
            f"{_format_params(r.n_params)} ({r.n_params:,}) | "
            f"{r.final_val_mae:.4f} | {passes} |"
        )
    return "\n".join(lines)


def _build_pareto_svg(
    results: list[CellResult], title: str, threshold: float = PASS_THRESHOLD
) -> str:
    """Hand-rolled SVG: log10(n_params) vs final val_mae."""
    if not results:
        return "<p>(no results)</p>"
    import math

    xs = [math.log10(max(r.n_params, 1)) for r in results]
    ys = [r.final_val_mae for r in results]

    margin = 60
    width = 520
    height = 320
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    x_min, x_max = min(xs) - 0.1, max(xs) + 0.1
    y_min = 0.0
    y_max = max(max(ys), threshold + 0.2) * 1.05

    def sx(x: float) -> float:
        return margin + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'style="font-family: -apple-system, sans-serif; font-size: 11px;">',
        f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="13" '
        f'font-weight="bold">{title}</text>',
        # Axes
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
        f'stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="#333" stroke-width="1"/>',
        # Threshold line
        f'<line x1="{margin}" y1="{sy(threshold):.1f}" x2="{width - margin}" '
        f'y2="{sy(threshold):.1f}" stroke="#c44" stroke-width="1" stroke-dasharray="4 3"/>',  # noqa: E501
        f'<text x="{width - margin - 4}" y="{sy(threshold) - 4:.1f}" text-anchor="end" '
        f'fill="#c44">val_mae = {threshold} (T1 pass threshold)</text>',
        f'<text x="{margin - 8}" y="{(margin + height - margin) / 2}" text-anchor="end" '  # noqa: E501
        f'transform="rotate(-90 {margin - 8} {(margin + height - margin) / 2})">final val_mae</text>',  # noqa: E501
        f'<text x="{(margin + width - margin) / 2}" y="{height - margin + 36}" '
        f'text-anchor="middle">log10(n_params)</text>',
    ]

    # Y-axis ticks
    for ti in range(5):
        y = y_min + ti * (y_max - y_min) / 4
        parts.append(
            f'<line x1="{margin - 4}" y1="{sy(y):.1f}" x2="{margin}" y2="{sy(y):.1f}" stroke="#333"/>'  # noqa: E501
        )
        parts.append(
            f'<text x="{margin - 6}" y="{sy(y) + 3:.1f}" text-anchor="end">{y:.2f}</text>'  # noqa: E501
        )
    # X-axis ticks
    for x_tick in range(int(x_min), int(x_max) + 1):
        if x_tick < x_min or x_tick > x_max:
            continue
        parts.append(
            f'<line x1="{sx(x_tick):.1f}" y1="{height - margin}" x2="{sx(x_tick):.1f}" '
            f'y2="{height - margin + 4}" stroke="#333"/>'
        )
        parts.append(
            f'<text x="{sx(x_tick):.1f}" y="{height - margin + 16}" text-anchor="middle">'  # noqa: E501
            f"10^{x_tick}</text>"
        )

    # Connect cells in n_params order
    sorted_results = sorted(zip(xs, ys, results, strict=True), key=lambda t: t[0])
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in sorted_results)
    parts.append(
        f'<polyline points="{poly}" fill="none" stroke="#357" stroke-width="1.5" opacity="0.5"/>'  # noqa: E501
    )

    # Points
    for x, y, r in sorted_results:
        passes = r.final_val_mae < threshold
        color = "#1a8" if passes else "#c44"
        h1, h2 = r.body_widths
        label = f"{h1}×{h2} n={r.n_residual_blocks}"
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}" '
            f'stroke="#000" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{sx(x):.1f}" y="{sy(y) - 9:.1f}" text-anchor="middle" '
            f'font-size="10">{label}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _pick_phase_a(results: list[CellResult]) -> CellResult | None:
    """Smallest n_params cell with val_mae < threshold."""
    passing = [r for r in results if r.final_val_mae < PASS_THRESHOLD]
    if not passing:
        return None
    return min(passing, key=lambda r: r.n_params)


def _comfortable_one_up(
    results: list[CellResult], pick: CellResult
) -> CellResult | None:
    """Cell one size up from pick (next-largest n_params)."""
    larger = sorted(
        [r for r in results if r.n_params > pick.n_params], key=lambda r: r.n_params
    )
    return larger[0] if larger else None


def main() -> None:
    phase_a_runs = _latest_run_per_config("A")
    phase_b_runs = _latest_run_per_config("B")

    phase_a_results: list[CellResult] = []
    for stem, run_dir in sorted(phase_a_runs.items()):
        r = _load_run(run_dir, "A", stem)
        if r is not None:
            phase_a_results.append(r)

    phase_b_results: list[CellResult] = []
    for stem, run_dir in sorted(phase_b_runs.items()):
        r = _load_run(run_dir, "B", stem)
        if r is not None:
            phase_b_results.append(r)

    print("=== Phase A (widths, n_residual=0) ===")
    print(_format_table(phase_a_results))
    print()

    pick_a = _pick_phase_a(phase_a_results)
    if pick_a is None:
        print("⚠️  No Phase A cell met val_mae < 0.5 — STOP and debug.")
        print(
            "    Per methodology: failure isn't capacity, "
            "it's optimization/BN/eval/data."
        )
        # Still write results.md so the failure is documented.
    else:
        h1, h2 = pick_a.body_widths
        comfortable = _comfortable_one_up(phase_a_results, pick_a)
        print(
            f"Phase A pick (h1*, h2*) = ({h1}, {h2})  "
            f"[val_mae {pick_a.final_val_mae:.4f}, n_params {pick_a.n_params:,}]"
        )
        if comfortable is not None:
            ch1, ch2 = comfortable.body_widths
            print(
                f"T1 comfortable (one up) = ({ch1}, {ch2})  "
                f"[val_mae {comfortable.final_val_mae:.4f}, "
                f"n_params {comfortable.n_params:,}]"
            )
        print()

    if phase_b_results:
        print("=== Phase B (residual sweep at chosen widths) ===")
        # Sort Phase B by n_residual_blocks
        b_sorted = sorted(phase_b_results, key=lambda r: r.n_residual_blocks)
        print(_format_table(b_sorted))
        print()

    # Write results.md
    md_lines = ["# T1 — capacity floor: Phase A + Phase B results", ""]
    md_lines.append("## Phase A — width sweep (n_residual_blocks = 0)")
    md_lines.append("")
    md_lines.append(
        "Walk down from `(4096, 1024)` (anchor — re-anchored from `(8192, 2048)` "
        "per pre-flight extrapolation: T0 calibration projected `(8192, 2048)` "
        "n=0 batch 1024 to ~50–70 ms/step, exceeding the ~30 ms/step ceiling "
        "that keeps Phase A under a 30-min budget) to `(512, 256)` (smallest "
        "meaningfully wider than input dim 144). All cells share batch 1024, "
        f"7000 steps, lr=1e-3, seed 0, split_seed 42 (val eval 10K/{734832:,})."
    )
    md_lines.append("")
    md_lines.append(_format_table(phase_a_results))
    md_lines.append("")
    if pick_a is not None:
        h1, h2 = pick_a.body_widths
        md_lines.append(
            f"**Phase A pick: `(h1*, h2*) = ({h1}, {h2})`** — "
            f"smallest n_params cell with val_mae < {PASS_THRESHOLD} "
            f"(final val_mae = {pick_a.final_val_mae:.4f}, "
            f"n_params = {_format_params(pick_a.n_params)})."
        )
    else:
        md_lines.append(
            "**Phase A pick: NONE** — no cell met val_mae < "
            f"{PASS_THRESHOLD}. Per methodology: stop and debug "
            "instead of going to Phase B (failure isn't capacity, "
            "it's optimization/BN/eval/data)."
        )
    md_lines.append("")

    if phase_b_results:
        md_lines.append("## Phase B — residual sweep at `(h1*, h2*)`")
        md_lines.append("")
        md_lines.append(
            "Same widths, batch, seed, steps, lr as Phase A; varies "
            "`n_residual_blocks ∈ {0, 1, 2, 4}`. n=0 is reused from Phase A's pick."
        )
        md_lines.append("")
        b_table = sorted(phase_b_results, key=lambda r: r.n_residual_blocks)
        # Include the n=0 pick row
        if pick_a is not None and not any(r.n_residual_blocks == 0 for r in b_table):
            b_table = [pick_a] + b_table
        md_lines.append(_format_table(b_table))
        md_lines.append("")

    md_lines.append("## Pareto plot")
    md_lines.append("")
    md_lines.append(
        "![Phase A Pareto](plots/phaseA_pareto.svg) "
        "(open `plots/phaseA_pareto.html` for the live view)"
    )
    md_lines.append("")

    # Write Pareto SVG + HTML wrapper
    plots_dir = TIER_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    svg = _build_pareto_svg(phase_a_results, "T1 Phase A — capacity floor")
    (plots_dir / "phaseA_pareto.svg").write_text(svg)
    html = (
        "<!doctype html><html><head><title>T1 Phase A Pareto</title></head>"
        '<body style="background: #fafafa; padding: 30px;">' + svg + "</body></html>"
    )
    (plots_dir / "phaseA_pareto.html").write_text(html)

    md = "\n".join(md_lines)

    # Append intuition.md if it exists (per project convention)
    intuition_path = TIER_DIR / "intuition.md"
    if intuition_path.exists():
        md += "\n\n---\n\n"
        md += intuition_path.read_text()

    (TIER_DIR / "results.md").write_text(md)
    print(f"results.md written to {(TIER_DIR / 'results.md').relative_to(REPO_ROOT)}")
    print(
        f"plot written to {(plots_dir / 'phaseA_pareto.html').relative_to(REPO_ROOT)}"
    )

    # Write _picks.json
    picks: dict = {
        "phase_a_results": [
            {
                "config_stem": r.config_stem,
                "body_widths": list(r.body_widths),
                "n_residual_blocks": r.n_residual_blocks,
                "n_params": r.n_params,
                "final_val_mae": r.final_val_mae,
            }
            for r in sorted(phase_a_results, key=lambda r: r.n_params)
        ],
        "phase_b_results": [
            {
                "config_stem": r.config_stem,
                "body_widths": list(r.body_widths),
                "n_residual_blocks": r.n_residual_blocks,
                "n_params": r.n_params,
                "final_val_mae": r.final_val_mae,
            }
            for r in sorted(phase_b_results, key=lambda r: r.n_residual_blocks)
        ],
        "pass_threshold": PASS_THRESHOLD,
    }
    if pick_a is not None:
        h1, h2 = pick_a.body_widths
        picks["phase_a_pick"] = {
            "body_widths": [h1, h2],
            "n_residual_blocks": 0,
            "n_params": pick_a.n_params,
            "final_val_mae": pick_a.final_val_mae,
        }
        comfortable = _comfortable_one_up(phase_a_results, pick_a)
        if comfortable is not None:
            ch1, ch2 = comfortable.body_widths
            picks["t1_comfortable"] = {
                "body_widths": [ch1, ch2],
                "n_residual_blocks": 0,
                "n_params": comfortable.n_params,
                "final_val_mae": comfortable.final_val_mae,
            }
    else:
        picks["phase_a_pick"] = None

    if phase_b_results:
        # Pick smallest n that captures most of the gain
        all_b = sorted(
            ([pick_a] if pick_a is not None else []) + phase_b_results,
            key=lambda r: r.n_residual_blocks,
        )
        # If MAE flat across all (range < 0.05), ship n=0
        maes = [r.final_val_mae for r in all_b]
        if max(maes) - min(maes) < 0.05:
            best = all_b[0]  # smallest n (= 0 if pick_a, else smallest measured)
            rationale = (
                "Phase B MAE flat across all n (range < 0.05) → ship smallest n."
            )
        else:
            # Pick smallest n with MAE within 5% of the best
            best_mae = min(maes)
            for r in all_b:
                if r.final_val_mae <= best_mae + 0.05:
                    best = r
                    break
            rationale = (
                "Phase B: smallest n_res with MAE within 0.05 "
                f"of best ({best_mae:.4f})."
            )
        h1, h2 = best.body_widths
        picks["t1_architecture"] = {
            "body_widths": [h1, h2],
            "n_residual_blocks": best.n_residual_blocks,
            "n_params": best.n_params,
            "final_val_mae": best.final_val_mae,
            "rationale": rationale,
        }

    (TIER_DIR / "_picks.json").write_text(json.dumps(picks, indent=2))
    print(f"_picks.json written to {(TIER_DIR / '_picks.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
