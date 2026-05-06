"""Render DAVI runs' error trajectories as a static HTML/SVG page.

Reads each run's metrics.jsonl, extracts every ``event="eval"`` record,
and produces a stack of charts at ``error_trajectories.html``:

1. **Macro-MAE trajectory** — overlaid line charts, x=step, y=macro_mae.
2. **Per-depth MAE small multiples** — one mini-chart per depth (1..14),
   grouped into rows by depth range so cells in a row share a y-cap.
3. **Greedy solve-rate small multiples** — same shape as #2.
4. **Avg solve-length small multiples** — same shape as #2.
5. **Solve-length histograms** (post-hoc, N=200 per depth per run) —
   same depth-range grouping as #2–4.
6. **Per-run depth-step heatmap** — depth × step → MAE per run.

## Chart layout convention (project-wide)

When a per-depth metric needs to be charted across the full {1..14} depth
range, use **per-depth small multiples grouped by depth range**, not
multi-depth-overlay-on-one-axis. The grouping is:

- **Shallow** (depths 1–5): 5 cells per row, y-cap tight to the shallow
  scale.
- **Mid** (depths 6–10): 5 cells per row, y-cap tight to the mid scale.
- **Deep** (depths 11–14): 4 cells per row, y-cap tight to the deep
  scale.

Why grouped: depths 1–5 and depths 11–14 have very different natural
scales (e.g. avg solve length of d=2 is ~2, d=13 is ~7; squashing both
onto a single y-axis flattens the shallow detail or wastes the deep
range). Why per-depth: visual overlays of multiple depths on one axis
make per-depth trends hard to read; small multiples let each depth's
trajectory be inspected independently while keeping nearby depths
visually adjacent.

This convention applies to MAE, solve-rate, avg-solve-length, and
histograms below — and should apply to any future per-depth chart
added here.

Pure stdlib + inline SVG. Open with
``open experiments/davi-2x2/results/error_trajectories.html``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = EXPERIMENT_DIR / "runs"

# (label, run_dir, color, dasharray)
# Cycle-3 view: cycle-1 baseline + two K_max=20 fresh-start cells.
# Cycle-2 warm-start runs are documented in intuition.md; not charted here
# to keep the cycle-3 comparison readable. Re-add via this list to compare.
RUNS: list[tuple[str, str, str, str]] = [
    ("cycle-1 baseline-30k  (K=18, sync=500)", "baseline-30k", "#1f77b4", ""),
    ("cycle-3 sync500_kmax20-30k  (K=20)", "sync500_kmax20-30k", "#2ca02c", ""),
    ("cycle-3 sync1000_kmax20-30k  (K=20)", "sync1000_kmax20-30k", "#ff7f0e", ""),
    ("cycle-4 kmax28_warm-30k  (K=28, warm-start)", "kmax28_warm-30k", "#d62728", ""),
]

DEPTHS = list(range(1, 15))


def load_run(run_dir: str) -> list[dict]:
    """Return list of eval records (step, macro_mae, per_depth_mae)."""
    path = BASELINE_DIR / run_dir / "metrics.jsonl"
    evals: list[dict] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") != "eval":
                continue
            # Skip duplicate post-loop final eval (same step as last loop eval)
            if evals and evals[-1]["step"] == r["step"]:
                continue
            evals.append(r)
    return evals


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    cmds = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for x, y in points[1:]:
        cmds.append(f"L {x:.2f} {y:.2f}")
    return " ".join(cmds)


def chart_macro(runs_data: dict[str, list[dict]]) -> str:
    """Top chart: macro_mae over step, all 4 runs overlaid."""
    width, height = 900, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 40

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # x range: max step across runs
    max_step = max(
        max((r["step"] for r in evals), default=0) for evals in runs_data.values()
    )
    if max_step == 0:
        max_step = 1
    # y range: macro_mae spans ~3.0 to ~7.5
    all_macro = [r["macro_mae"] for evals in runs_data.values() for r in evals]
    y_min, y_max = 0, max(all_macro) * 1.05

    def x(s: float) -> float:
        return pad_l + (s / max_step) * plot_w

    def y(v: float) -> float:
        return pad_t + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="white" stroke="#ccc"/>',
    ]
    # y gridlines + labels
    for v in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        if v > y_max:
            continue
        gy = y(v)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            f'stroke="#eee"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#666">{v}</text>'
        )
    # gate line at macro_mae = 1.0
    parts.append(
        f'<line x1="{pad_l}" y1="{y(1.0):.1f}" x2="{pad_l + plot_w}" '
        f'y2="{y(1.0):.1f}" stroke="#000" stroke-dasharray="4 4" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{pad_l + plot_w - 8}" y="{y(1.0) - 4:.1f}" text-anchor="end" '
        f'font-size="11" fill="#000">acceptance gate (macro_mae &lt; 1.0)</text>'
    )
    # x labels
    for tick in [0, max_step // 4, max_step // 2, 3 * max_step // 4, max_step]:
        gx = x(tick)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{pad_t + plot_h}" x2="{gx:.1f}" '
            f'y2="{pad_t + plot_h + 4}" stroke="#666"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
            f'font-size="11" fill="#666">{tick}</text>'
        )

    # Plot each run
    legend_y = pad_t + 8
    for label, run_dir, color, _ in RUNS:
        evals = runs_data[run_dir]
        pts = [(x(r["step"]), y(r["macro_mae"])) for r in evals]
        parts.append(
            f'<path d="{line_path(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="2"/>'
        )
        # circles at each point
        for px, py in pts:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{color}"/>'
            )

    # Axis labels
    parts.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 8}" text-anchor="middle" '
        f'font-size="12" fill="#333">training step</text>'
    )
    parts.append(
        f'<text x="{14}" y="{pad_t + plot_h / 2}" text-anchor="middle" '
        f'font-size="12" fill="#333" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2})">macro_mae</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _band_chart(
    runs_data: dict[str, list[dict]],
    *,
    title: str,
    depths: list[int],
    y_cap: float,
    y_ticks: list[float],
    value_for: callable,  # type: ignore[type-arg]
    ref_lines: list[tuple[float, str]] | None = None,
) -> str:
    """Generic banded small-multiples chart.

    `value_for(record, depth)` returns a float (or None to skip) for that
    eval record at that depth. Used for both per-depth MAE and solve-rate
    bands; only the value extractor differs.
    """
    cell_w, cell_h = 220, 160
    cols = max(len(depths), 1)
    pad = 16
    total_w = cols * cell_w + pad * (cols + 1)
    total_h = cell_h + pad * 2 + 40
    pad_l, pad_t, pad_r, pad_b = 32, 22, 8, 22
    plot_w = cell_w - pad_l - pad_r
    plot_h = cell_h - pad_t - pad_b

    max_step = max(
        max((r["step"] for r in evals), default=0) for evals in runs_data.values()
    )

    parts: list[str] = [
        f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
        f'<text x="{total_w / 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="#222">{title}</text>',
    ]

    def y_for(v: float, oy: float) -> float:
        cv = max(0.0, min(y_cap, v))
        return oy + pad_t + plot_h - (cv / y_cap) * plot_h

    def x_for(s: float, ox: float) -> float:
        return ox + pad_l + (s / max_step) * plot_w

    for i, depth in enumerate(depths):
        ox = pad + i * (cell_w + pad)
        oy = 30 + pad

        # Cell frame
        parts.append(
            f'<rect x="{ox + pad_l}" y="{oy + pad_t}" width="{plot_w}" '
            f'height="{plot_h}" fill="white" stroke="#ccc"/>'
        )
        # Cell title
        parts.append(
            f'<text x="{ox + cell_w / 2}" y="{oy + 14}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#333">depth {depth}</text>'
        )

        # y-axis gridlines + labels
        for tick in y_ticks:
            if tick > y_cap:
                continue
            gy = y_for(tick, oy)
            parts.append(
                f'<line x1="{ox + pad_l}" y1="{gy:.1f}" '
                f'x2="{ox + pad_l + plot_w}" y2="{gy:.1f}" '
                f'stroke="#eee" stroke-width="1"/>'
            )
            label = f"{tick:g}" if tick == int(tick) else f"{tick:.1f}"
            parts.append(
                f'<text x="{ox + pad_l - 3}" y="{gy + 3:.1f}" text-anchor="end" '
                f'font-size="9" fill="#888">{label}</text>'
            )

        # Reference lines (e.g. MAE=1 acceptance gate)
        for ref_v, ref_color in ref_lines or []:
            if ref_v > y_cap:
                continue
            ref_y = y_for(ref_v, oy)
            parts.append(
                f'<line x1="{ox + pad_l}" y1="{ref_y:.1f}" '
                f'x2="{ox + pad_l + plot_w}" y2="{ref_y:.1f}" '
                f'stroke="{ref_color}" stroke-dasharray="3 3" stroke-width="1"/>'
            )

        # Detect clipping for this cell
        clipped = False
        for evals in runs_data.values():
            for r in evals:
                v = value_for(r, depth)
                if v is not None and v > y_cap:
                    clipped = True
                    break
            if clipped:
                break
        if clipped:
            parts.append(
                f'<text x="{ox + pad_l + plot_w - 4}" y="{oy + pad_t + 10}" '
                f'text-anchor="end" font-size="10" fill="#c00">▲ clipped</text>'
            )

        # Plot each run
        for label, run_dir, color, _ in RUNS:
            evals = runs_data[run_dir]
            pts = []
            for r in evals:
                v = value_for(r, depth)
                if v is None:
                    continue
                pts.append((x_for(r["step"], ox), y_for(v, oy)))
            if pts:
                parts.append(
                    f'<path d="{line_path(pts)}" fill="none" stroke="{color}" '
                    f'stroke-width="1.5" opacity="0.9"/>'
                )

    parts.append("</svg>")
    return "\n".join(parts)


def _mae_value(r: dict, depth: int) -> float | None:
    pdm = r.get("per_depth_mae", {})
    return pdm.get(str(depth))


def _solve_value(r: dict, depth: int) -> float | None:
    return r.get(f"solve_rate_d{depth}")


def _avg_solve_len_value(r: dict, depth: int) -> float | None:
    v = r.get(f"avg_solve_len_d{depth}")
    return v if v is not None else None


def chart_per_depth_banded(runs_data: dict[str, list[dict]]) -> str:
    """Three banded MAE charts: shallow / mid / deep with band-tight y-caps."""
    bands = [
        ("Shallow MAE — depths 1–5  (y-axis 0..2)", [1, 2, 3, 4, 5], 2.0, [0, 0.5, 1, 1.5, 2]),
        ("Mid MAE — depths 6–10  (y-axis 0..6)", [6, 7, 8, 9, 10], 6.0, [0, 1, 2, 3, 4, 5, 6]),
        ("Deep MAE — depths 11–14  (y-axis 0..12)", [11, 12, 13, 14], 12.0, [0, 2, 4, 6, 8, 10, 12]),
    ]
    return "\n".join(
        _band_chart(
            runs_data,
            title=title,
            depths=depths,
            y_cap=y_cap,
            y_ticks=ticks,
            value_for=_mae_value,
            ref_lines=[(1.0, "#aaa")],
        )
        for title, depths, y_cap, ticks in bands
    )


def chart_solve_rate_banded(runs_data: dict[str, list[dict]]) -> str:
    """Solve-rate small multiples grouped Shallow / Mid / Deep (5/5/4).

    Y-scale 0..1 across all bands. Reference lines: 0.99 (acceptance
    gate, solid black), 0.5 (better-than-coinflip, light grey). Cells
    with no data render as empty plots — when a run's eval cycle didn't
    sample that depth, the line is absent.
    """
    bands = [
        ("Shallow solve-rate — depths 1–5", [1, 2, 3, 4, 5]),
        ("Mid solve-rate — depths 6–10", [6, 7, 8, 9, 10]),
        ("Deep solve-rate — depths 11–14", [11, 12, 13, 14]),
    ]
    return "\n".join(
        _band_chart(
            runs_data,
            title=title,
            depths=depths,
            y_cap=1.0,
            y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
            value_for=_solve_value,
            ref_lines=[(0.99, "#000"), (0.5, "#bbb")],
        )
        for title, depths in bands
    )


def chart_avg_solve_len_banded(runs_data: dict[str, list[dict]]) -> str:
    """Avg-solve-length small multiples grouped Shallow / Mid / Deep (5/5/4).

    Per-cell horizontal reference at ``y = depth`` (the random-walk
    length used to scramble the test cube). When avg_solve_len < depth,
    the solver is finding a path shorter than the random walk took —
    confirming the audit finding that length-d random walks usually
    produce true-V*-depth states well below d. Y-cap rises by band so
    each band is tight to its natural scale.
    """
    bands = [
        ("Shallow avg solve-length — depths 1–5  (y 0..6)",
         [1, 2, 3, 4, 5], 6.0, [0, 1, 2, 3, 4, 5, 6]),
        ("Mid avg solve-length — depths 6–10  (y 0..12)",
         [6, 7, 8, 9, 10], 12.0, [0, 2, 4, 6, 8, 10, 12]),
        ("Deep avg solve-length — depths 11–14  (y 0..14)",
         [11, 12, 13, 14], 14.0, [0, 2, 4, 6, 8, 10, 12, 14]),
    ]
    out = []
    for title, depths, y_cap, ticks in bands:
        out.append(_band_chart_with_per_cell_ref(
            runs_data,
            title=title,
            depths=depths,
            y_cap=y_cap,
            y_ticks=ticks,
            value_for=_avg_solve_len_value,
            per_cell_ref=lambda d: float(d),
            per_cell_ref_color="#888",
        ))
    return "\n".join(out)


def _band_chart_with_per_cell_ref(
    runs_data: dict[str, list[dict]],
    *,
    title: str,
    depths: list[int],
    y_cap: float,
    y_ticks: list[float],
    value_for,
    per_cell_ref,
    per_cell_ref_color: str,
) -> str:
    """Like _band_chart but with a per-cell horizontal reference (y = f(depth))."""
    cell_w, cell_h = 220, 160
    cols = max(len(depths), 1)
    pad = 16
    total_w = cols * cell_w + pad * (cols + 1)
    total_h = cell_h + pad * 2 + 40
    pad_l, pad_t, pad_r, pad_b = 32, 22, 8, 22
    plot_w = cell_w - pad_l - pad_r
    plot_h = cell_h - pad_t - pad_b

    max_step = max(
        max((r["step"] for r in evals), default=0) for evals in runs_data.values()
    )

    parts = [
        f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
        f'<text x="{total_w / 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="#222">{title}</text>',
    ]

    def y_for(v: float, oy: float) -> float:
        cv = max(0.0, min(y_cap, v))
        return oy + pad_t + plot_h - (cv / y_cap) * plot_h

    def x_for(s: float, ox: float) -> float:
        return ox + pad_l + (s / max_step) * plot_w

    for i, depth in enumerate(depths):
        ox = pad + i * (cell_w + pad)
        oy = 30 + pad

        parts.append(
            f'<rect x="{ox + pad_l}" y="{oy + pad_t}" width="{plot_w}" '
            f'height="{plot_h}" fill="white" stroke="#ccc"/>'
        )
        parts.append(
            f'<text x="{ox + cell_w / 2}" y="{oy + 14}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#333">depth {depth}</text>'
        )

        for tick in y_ticks:
            if tick > y_cap:
                continue
            gy = y_for(tick, oy)
            parts.append(
                f'<line x1="{ox + pad_l}" y1="{gy:.1f}" '
                f'x2="{ox + pad_l + plot_w}" y2="{gy:.1f}" '
                f'stroke="#eee"/>'
            )
            label = f"{tick:g}" if tick == int(tick) else f"{tick:.1f}"
            parts.append(
                f'<text x="{ox + pad_l - 3}" y="{gy + 3:.1f}" text-anchor="end" '
                f'font-size="9" fill="#888">{label}</text>'
            )

        # Per-cell reference line at y = depth
        ref_v = per_cell_ref(depth)
        if ref_v <= y_cap:
            ref_y = y_for(ref_v, oy)
            parts.append(
                f'<line x1="{ox + pad_l}" y1="{ref_y:.1f}" '
                f'x2="{ox + pad_l + plot_w}" y2="{ref_y:.1f}" '
                f'stroke="{per_cell_ref_color}" stroke-dasharray="4 3" '
                f'stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{ox + pad_l + plot_w - 4}" y="{ref_y - 3:.1f}" '
                f'text-anchor="end" font-size="9" fill="#888">y = depth ({depth})</text>'
            )

        for label, run_dir, color, _ in RUNS:
            evals = runs_data[run_dir]
            pts = []
            for r in evals:
                v = value_for(r, depth)
                if v is None:
                    continue
                pts.append((x_for(r["step"], ox), y_for(v, oy)))
            if pts:
                parts.append(
                    f'<path d="{line_path(pts)}" fill="none" stroke="{color}" '
                    f'stroke-width="1.5" opacity="0.9"/>'
                )

    parts.append("</svg>")
    return "\n".join(parts)


def chart_solve_histograms(hist_data: dict, strategy: str, method: str) -> str:
    """Solve-length histograms grouped Shallow / Mid / Deep (5/5/4).

    For each (run, depth), shows the empirical distribution of solve
    lengths over N attempts (typically 200) at the run's terminal
    checkpoint, under the given sampling ``strategy`` and search
    ``method``. Bars dodged side-by-side per length bucket per run.
    Right strip shows the failed-attempt fraction. Cells without
    histogram data (depth not captured, or this strategy/method not
    run) render as empty plots — capture_solve_histograms.py's
    ``DEPTHS`` × ``STRATEGIES`` × ``METHODS`` control coverage.
    """
    strategy_label = {
        "random_walk_depth": "random-walk depth",
        "v_star_stratified": "V*-stratified",
    }.get(strategy, strategy)
    beam_width = (hist_data.get("config", {}) or {}).get("beam_width")
    if method == "greedy":
        method_label = "greedy"
    elif method == "beam":
        method_label = (
            f"beam(width={beam_width})" if beam_width is not None else "beam"
        )
    else:
        method_label = method
    bands = [
        (f"Solve-length histograms ({strategy_label}, {method_label}) — depths 1–5",
         [1, 2, 3, 4, 5]),
        (f"Solve-length histograms ({strategy_label}, {method_label}) — depths 6–10",
         [6, 7, 8, 9, 10]),
        (f"Solve-length histograms ({strategy_label}, {method_label}) — depths 11–14",
         [11, 12, 13, 14]),
    ]
    return "\n".join(
        _hist_band(hist_data, title, depths, strategy, method)
        for title, depths in bands
    )


def _hist_band(
    hist_data: dict,
    title: str,
    depths: list[int],
    strategy: str,
    method: str,
) -> str:
    # Cell width chosen so 5 cells per row fit in roughly the same visual
    # footprint as the MAE/solve-rate/avg-len bands above. Plot area is
    # tight at d=14 (28 buckets × 3 runs ≈ 2 px per bar) but legible —
    # bars use stroke + opacity so neighbours stay distinguishable.
    cell_w, cell_h = 280, 220
    cols = len(depths)
    pad = 16
    fail_zone_w = 40  # right strip for failed bar
    total_w = cols * cell_w + pad * (cols + 1)
    total_h = cell_h + pad * 2 + 40
    pad_l, pad_t, pad_r, pad_b = 36, 26, 8 + fail_zone_w, 30
    plot_w = cell_w - pad_l - pad_r
    plot_h = cell_h - pad_t - pad_b

    parts: list[str] = [
        f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
        f'<text x="{total_w / 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="#222">{title}</text>',
    ]

    n_per = hist_data["config"]["n_per_depth"]
    budget_factor = hist_data["config"]["depth_budget_factor"]
    beam_max_steps = hist_data["config"].get("beam_max_steps", 20)

    for i, depth in enumerate(depths):
        ox = pad + i * (cell_w + pad)
        oy = 30 + pad
        # Greedy budget grows linearly with depth (2*depth); beam budget is
        # the fixed M6 acceptance config (max_steps=20). Use the right cap
        # per method so the histogram x-axis matches what the search saw.
        if method == "beam":
            budget = beam_max_steps
        else:
            budget = budget_factor * depth

        # Build per-run histograms for this depth
        per_run_hist: dict[str, dict] = {}
        for label, run_subdir, color, _ in RUNS:
            run_data = hist_data["runs"].get(run_subdir)
            if run_data is None:
                continue
            strategy_data = run_data.get(strategy)
            if strategy_data is None:
                continue
            method_data = strategy_data.get(method)
            if method_data is None:
                continue
            lens = method_data["lens"].get(str(depth), [])
            counts: list[int] = [0] * (budget + 1)  # index 0 not used; 1..budget
            n_failed = 0
            for ln in lens:
                if ln < 0:
                    n_failed += 1
                elif 1 <= ln <= budget:
                    counts[ln] += 1
                # ln=0 (already-solved at start) is rare; ignored for hist
            per_run_hist[run_subdir] = {
                "color": color,
                "counts": counts,
                "n_failed": n_failed,
                "n_total": len(lens),
            }

        # Y-axis: max count across all runs (normalize the chart)
        max_count = 1
        for hd in per_run_hist.values():
            for c in hd["counts"][1:]:
                if c > max_count:
                    max_count = c
        # Round y_max up nicely
        y_max = max(max_count, 10)

        # frame
        parts.append(
            f'<rect x="{ox + pad_l}" y="{oy + pad_t}" width="{plot_w}" '
            f'height="{plot_h}" fill="white" stroke="#ccc"/>'
        )
        # Failed-zone strip
        fz_x = ox + pad_l + plot_w + 6
        parts.append(
            f'<rect x="{fz_x}" y="{oy + pad_t}" width="{fail_zone_w - 14}" '
            f'height="{plot_h}" fill="#fff" stroke="#ddd"/>'
        )
        parts.append(
            f'<text x="{fz_x + (fail_zone_w - 14) / 2}" y="{oy + pad_t + plot_h + 14}" '
            f'text-anchor="middle" font-size="10" fill="#666">failed</text>'
        )
        # Title
        parts.append(
            f'<text x="{ox + cell_w / 2}" y="{oy + 16}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#333">'
            f'depth {depth} (budget {budget}, n={n_per})</text>'
        )

        # x-axis ticks
        x_ticks = [1] + list(range(2, budget + 1, max(1, budget // 6))) + [budget]
        x_ticks = sorted(set(x_ticks))
        for t in x_ticks:
            tx = ox + pad_l + (t - 1) / (budget - 1 + 1e-9) * plot_w if budget > 1 else ox + pad_l + plot_w / 2
            parts.append(
                f'<line x1="{tx:.1f}" y1="{oy + pad_t + plot_h}" '
                f'x2="{tx:.1f}" y2="{oy + pad_t + plot_h + 4}" stroke="#666"/>'
            )
            parts.append(
                f'<text x="{tx:.1f}" y="{oy + pad_t + plot_h + 14}" '
                f'text-anchor="middle" font-size="9" fill="#666">{t}</text>'
            )
        # x-axis label
        parts.append(
            f'<text x="{ox + pad_l + plot_w / 2}" y="{oy + pad_t + plot_h + 26}" '
            f'text-anchor="middle" font-size="10" fill="#444">solve length (moves)</text>'
        )

        # y-axis labels
        for tick in [0, y_max // 4, y_max // 2, 3 * y_max // 4, y_max]:
            ty = oy + pad_t + plot_h - (tick / y_max) * plot_h
            parts.append(
                f'<line x1="{ox + pad_l}" y1="{ty:.1f}" '
                f'x2="{ox + pad_l + plot_w}" y2="{ty:.1f}" stroke="#eee"/>'
            )
            parts.append(
                f'<text x="{ox + pad_l - 3}" y="{ty + 3:.1f}" text-anchor="end" '
                f'font-size="9" fill="#888">{tick}</text>'
            )

        # Plot each run as a step-line histogram (overlaid)
        # Length 1..budget mapped to x in plot_w
        def x_at(length: int) -> float:
            return ox + pad_l + (length - 0.5) / budget * plot_w

        def y_at(count: int) -> float:
            return oy + pad_t + plot_h - (count / y_max) * plot_h

        for run_subdir, hd in per_run_hist.items():
            color = hd["color"]
            counts = hd["counts"]
            # Build a step path: rectangle bars per length bucket
            # Use thin filled bars with horizontal offset per run for dodging
            run_idx = next(
                k for k, (_, rd, _, _) in enumerate(RUNS) if rd == run_subdir
            )
            n_runs = len(RUNS)
            bar_group_w = max(2.0, plot_w / budget * 0.85)
            bar_w = bar_group_w / n_runs
            for ln in range(1, budget + 1):
                c = counts[ln]
                if c <= 0:
                    continue
                bx = x_at(ln) - bar_group_w / 2 + run_idx * bar_w
                by = y_at(c)
                bh = (oy + pad_t + plot_h) - by
                parts.append(
                    f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" '
                    f'height="{bh:.2f}" fill="{color}" opacity="0.85">'
                    f'<title>length={ln} count={c}</title></rect>'
                )

            # Failed bar in the right zone — full vertical with height
            # proportional to fail fraction
            if hd["n_failed"] > 0:
                fail_frac = hd["n_failed"] / hd["n_total"]
                fail_h = fail_frac * plot_h
                fail_w = (fail_zone_w - 14) / n_runs
                fail_x = fz_x + run_idx * fail_w
                fail_y = oy + pad_t + plot_h - fail_h
                parts.append(
                    f'<rect x="{fail_x:.2f}" y="{fail_y:.2f}" width="{fail_w:.2f}" '
                    f'height="{fail_h:.2f}" fill="{hd["color"]}" opacity="0.85">'
                    f'<title>failed: {hd["n_failed"]}/{hd["n_total"]} ({fail_frac:.1%})</title>'
                    f"</rect>"
                )

    parts.append("</svg>")
    return "\n".join(parts)


def chart_heatmaps(runs_data: dict[str, list[dict]]) -> str:
    """One heatmap per run: x=step, y=depth (1..14), color=MAE."""
    cell_w, cell_h = 800, 180
    pad = 30
    n = len(RUNS)
    total_w = cell_w + pad * 2
    total_h = n * cell_h + pad * (n + 1) + 60

    pad_l, pad_t, pad_r, pad_b = 65, 28, 60, 26
    plot_w = cell_w - pad_l - pad_r
    plot_h = cell_h - pad_t - pad_b

    # Global MAE range for color scale (cap at 14 — max possible)
    mae_min, mae_max = 0.0, 10.0

    def color_for(v: float) -> str:
        # Viridis-like: blue (low) → green → yellow → red (high)
        t = max(0.0, min(1.0, (v - mae_min) / (mae_max - mae_min)))
        # 4-stop gradient
        stops = [
            (0.0, (37, 52, 148)),  # dark blue
            (0.33, (44, 162, 95)),  # green
            (0.66, (254, 196, 79)),  # yellow
            (1.0, (215, 25, 28)),  # red
        ]
        for j in range(len(stops) - 1):
            t0, c0 = stops[j]
            t1, c1 = stops[j + 1]
            if t0 <= t <= t1:
                u = (t - t0) / (t1 - t0)
                r = int(c0[0] + (c1[0] - c0[0]) * u)
                g = int(c0[1] + (c1[1] - c0[1]) * u)
                b = int(c0[2] + (c1[2] - c0[2]) * u)
                return f"rgb({r},{g},{b})"
        return "rgb(0,0,0)"

    parts: list[str] = [
        f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
        f'<text x="{total_w / 2}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="600" fill="#222">per-depth MAE heatmap (wavefront view)</text>',
    ]

    for i, (label, run_dir, color, _) in enumerate(RUNS):
        evals = runs_data[run_dir]
        if not evals:
            continue
        steps = [r["step"] for r in evals]
        if len(steps) < 2:
            continue
        max_step = max(steps)
        ox = pad
        oy = 40 + i * (cell_h + pad)

        # frame
        parts.append(
            f'<rect x="{ox + pad_l}" y="{oy + pad_t}" width="{plot_w}" '
            f'height="{plot_h}" fill="white" stroke="#ccc"/>'
        )
        # row title
        parts.append(
            f'<text x="{ox + pad_l}" y="{oy + 18}" font-size="12" '
            f'font-weight="600" fill="{color}">{label}</text>'
        )

        # cells: each (step, depth) → rect colored by MAE
        n_depth = len(DEPTHS)
        depth_h = plot_h / n_depth
        for k, r in enumerate(evals):
            pdm = r.get("per_depth_mae", {})
            # cell width from current to next eval (or fixed for last)
            x0 = (steps[k] - 0) / max_step * plot_w
            x1 = (steps[k + 1] / max_step * plot_w) if k + 1 < len(evals) else plot_w
            for j, depth in enumerate(DEPTHS):
                key = str(depth)
                if key not in pdm:
                    continue
                v = pdm[key]
                cx = ox + pad_l + x0
                cy = oy + pad_t + j * depth_h
                cw = max(1, x1 - x0)
                ch = depth_h
                col = color_for(v)
                parts.append(
                    f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw:.1f}" '
                    f'height="{ch:.1f}" fill="{col}">'
                    f'<title>step={r["step"]} depth={depth} mae={v:.2f}</title>'
                    f"</rect>"
                )

        # depth labels (y axis)
        for j, depth in enumerate(DEPTHS):
            ly = oy + pad_t + j * depth_h + depth_h / 2 + 3
            parts.append(
                f'<text x="{ox + pad_l - 6}" y="{ly:.1f}" text-anchor="end" '
                f'font-size="10" fill="#666">d{depth}</text>'
            )

        # step ticks on x axis
        for tick in [0, max_step // 4, max_step // 2, 3 * max_step // 4, max_step]:
            tx = ox + pad_l + (tick / max_step) * plot_w
            parts.append(
                f'<line x1="{tx:.1f}" y1="{oy + pad_t + plot_h}" '
                f'x2="{tx:.1f}" y2="{oy + pad_t + plot_h + 4}" stroke="#666"/>'
            )
            parts.append(
                f'<text x="{tx:.1f}" y="{oy + pad_t + plot_h + 17}" '
                f'text-anchor="middle" font-size="10" fill="#666">{tick}</text>'
            )

        # Inline legend strip
        legend_w = 50
        legend_h = 8
        lx = ox + pad_l + plot_w + 6
        ly = oy + pad_t
        for k in range(20):
            t = k / 19
            v = mae_min + t * (mae_max - mae_min)
            parts.append(
                f'<rect x="{lx:.1f}" y="{ly + k * (plot_h - legend_h) / 20:.1f}" '
                f'width="{legend_w / 4:.1f}" height="{(plot_h - legend_h) / 20:.1f}" '
                f'fill="{color_for(v)}"/>'
            )
        parts.append(
            f'<text x="{lx + legend_w / 4 + 4}" y="{ly + 8}" '
            f'font-size="10" fill="#666">{mae_max:.0f}</text>'
        )
        parts.append(
            f'<text x="{lx + legend_w / 4 + 4}" y="{ly + plot_h - 4}" '
            f'font-size="10" fill="#666">{mae_min:.0f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_legend() -> str:
    """Color legend used by macro and per-depth charts."""
    items = []
    for label, _, color, _ in RUNS:
        items.append(
            f'<span style="display:inline-block;margin-right:18px">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color};border-radius:2px;vertical-align:middle"></span>'
            f'<span style="margin-left:6px;font-size:12px;color:#333">{label}</span>'
            f"</span>"
        )
    return f'<div style="margin:12px 0">{"".join(items)}</div>'


def main() -> None:
    runs_data: dict[str, list[dict]] = {}
    for label, run_dir, _, _ in RUNS:
        runs_data[run_dir] = load_run(run_dir)
        print(f"loaded {label}: {len(runs_data[run_dir])} eval records")

    # Load post-hoc histograms (captured by capture_solve_histograms.py).
    hist_path = EXPERIMENT_DIR / "results" / "solve_histograms.json"
    hist_data = None
    if hist_path.exists():
        with hist_path.open() as f:
            hist_data = json.load(f)
        print(f"loaded histograms: {len(hist_data['runs'])} runs")
    else:
        print(f"WARN: histogram file not found at {hist_path}")

    out_path = EXPERIMENT_DIR / "results" / "error_trajectories.html"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DAVI baseline-30k — error trajectories</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 24px 32px; max-width: 980px; color: #222; }}
    h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
    h2 {{ font-size: 14px; margin: 32px 0 8px 0; color: #555;
          border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    p.note {{ font-size: 12px; color: #666; margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>M5 DAVI — baseline-30k error trajectories</h1>
  <p class="note">
    Four runs compared. Eval is on the 1400-state depth-stratified V* eval set,
    macro_mae averaged uniformly over depths 1..14 (some depths under-sampled —
    eval at d=14 may be noisier than d=10).
  </p>
  {render_legend()}

  <h2>1. Macro-MAE over training step</h2>
  <p class="note">
    Acceptance gate dashed at macro_mae &lt; 1.0. None of the runs come
    close. Note that the "warm sync=*" runs all start near macro_mae 3.13
    because they warm-start from baseline-30k's final state.
  </p>
  {chart_macro(runs_data)}

  <h2>2. Per-depth MAE — banded by depth range</h2>
  <p class="note">
    Three sections grouped by depth, each with band-tight y-axis. Dashed
    grey reference at MAE=1.0 (acceptance gate level). Cells marked
    "▲ clipped" had at least one value above the band cap (mostly early
    baseline-30k iterations before the wavefront formed).
  </p>
  {chart_per_depth_banded(runs_data)}

  <h2>3. Greedy solve rate over training — banded by depth range</h2>
  <p class="note">
    Capability over time. Solve-rate is what we actually care about
    (acceptance gate is &gt; 0.99 across all tested depths, dashed black
    line). y-axis 0..1 throughout; light dashed reference at 0.5
    (better-than-coinflip). Greedy solve runs every eval cycle on a
    50-state-per-depth random sample, so individual points have variance
    of ~&plusmn;0.05.
  </p>
  {chart_solve_rate_banded(runs_data)}

  <h2>4. Average solve length (moves used, on solved attempts) — banded</h2>
  <p class="note">
    Per-depth average move count over training. Dashed grey reference at
    <code>y = depth</code> (the random-walk length that scrambled the test
    cube). When the line sits <em>below</em> the reference, the model is
    finding a path shorter than the random walk — confirming the audit
    finding that length-d random walks usually produce true-V*-depth
    states well below d. <strong>Caveat:</strong> averages are over
    solved attempts only (failed attempts contribute nothing); when
    solve_rate is low, this metric is over a small, biased subset.
  </p>
  {chart_avg_solve_len_banded(runs_data)}

  <h2>5. Solve-length histograms — random_walk_depth, greedy, n=200</h2>
  <p class="note">
    For each (depth, run), the empirical distribution of solve lengths
    over 200 attempts at the run's terminal checkpoint. Bars are
    color-coded per run, dodged side-by-side per length bucket. Right
    strip is the failed-attempt count (heights normalized so the
    plot-area max=plot height; failed strip max=plot height = 100% fail).
    Sampling here is <em>random-walk-depth</em>: scramble = uniform
    random walk of length <em>d</em>, so true V* is <em>≤ d</em>. Search
    is <em>greedy</em> (argmin V over the 12 children, budget = 2·d).
  </p>
  {chart_solve_histograms(hist_data, "random_walk_depth", "greedy") if hist_data else
   '<p style="color:#c00">solve_histograms.json not found — run capture_solve_histograms.py first.</p>'}

  <h2>6. Solve-length histograms — random_walk_depth, beam(width=256), n=200</h2>
  <p class="note">
    Same sampling and inputs as section 5 (states are drawn once per
    cell and shared across methods), but search is <em>beam</em> with
    width 256 and max_steps 20 (the M6 acceptance config). Where greedy
    gets stuck on local-minima paths, beam keeps the top-K survivors
    each layer — so the gap between sections 5 and 6 is roughly "what
    width buys on top of pure argmin V_θ".
  </p>
  {chart_solve_histograms(hist_data, "random_walk_depth", "beam") if hist_data else
   '<p style="color:#c00">solve_histograms.json not found — run capture_solve_histograms.py first.</p>'}

  <h2>7. Solve-length histograms — v_star_stratified, greedy, n=200</h2>
  <p class="note">
    Same banded layout as section 5, but states are drawn from the BFS
    oracle's <code>{{s : V*(s) == d}}</code> shell — so true V* is
    <em>exactly</em> <em>d</em>. This is the apples-to-apples view of
    "solve a cube that is genuinely d moves from solved", uncluttered by
    the random-walk-depth distribution's heavy bias toward easier states
    (length-d random walks usually produce states well below true V*=d).
    A failed bar growing as <em>d</em> increases is the model's true
    coverage frontier, not a sampling artifact. Search is <em>greedy</em>.
  </p>
  {chart_solve_histograms(hist_data, "v_star_stratified", "greedy") if hist_data else
   '<p style="color:#c00">solve_histograms.json not found — run capture_solve_histograms.py first.</p>'}

  <h2>8. Solve-length histograms — v_star_stratified, beam(width=256), n=200</h2>
  <p class="note">
    Same V*-stratified inputs as section 7, but search is <em>beam</em>
    with width 256 (M6 acceptance config). Greedy at deep V* often
    collapses to 0% solve, hiding cycle-to-cycle differences; beam is
    where the cycle-4 vs baseline comparison actually shows up at d=12+.
  </p>
  {chart_solve_histograms(hist_data, "v_star_stratified", "beam") if hist_data else
   '<p style="color:#c00">solve_histograms.json not found — run capture_solve_histograms.py first.</p>'}

  <h2>9. Per-run wavefront heatmap (depth × step → MAE)</h2>
  <p class="note">
    Each row is one run. y-axis is depth 1..14, x-axis is training step,
    color is MAE (blue=low, red≥10). The "wavefront" is the boundary
    between blue (learned) and red (unlearned) — its growth tells you
    whether the model is still extending depth coverage. Hover for exact values.
  </p>
  {chart_heatmaps(runs_data)}

</body>
</html>
"""

    out_path.write_text(html)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    print(f"open with:  open {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
