"""Render the beam-search-2x2 sweep results as a static HTML/SVG page.

Reads the run's ``metrics.jsonl`` (event="cell" records) and
``solve_histograms_beam.json``, and produces ``beam_curves.html`` with:

1. **(depth × beam_width) heatmap** — solve_rate as color, one
   quick-glance overview of where the M6 acceptance gate is met.
2. **Solve-rate per-depth small multiples** — x=beam_width (log4 spacing),
   y=solve_rate (0..1), gate line at 0.999. Banded 5/5/4 (shallow/mid/deep).
3. **Mean V*-excess per-depth small multiples** — x=beam_width, y=mean
   excess (capped). Acceptance gate line at 1.0. Banded 5/5/4.
4. **Avg solve length per-depth small multiples** — x=beam_width, y=avg
   solve length, with a horizontal reference line at the scramble depth d
   (random walks of length d are the inputs). Banded 5/5/4.
5. **Solve-length histograms at the max beam_width** — per depth, banded
   5/5/4. Same shape as the M5 cycle-3 ``solve_histograms.json`` chart;
   the cell that the C5 acceptance run scales up.

## Chart layout convention (project-wide)

When a per-depth metric needs to be charted across the full {1..14} depth
range, use **per-depth small multiples grouped by depth range**, not
multi-depth-overlay-on-one-axis. The grouping is:

- **Shallow** (depths 1–5): 5 cells per row.
- **Mid** (depths 6–10): 5 cells per row.
- **Deep** (depths 11–14): 4 cells per row.

Why grouped: shallow and deep cells have very different natural scales
(e.g. avg solve length of d=2 is ~2, d=13 is ~7). Why per-depth: visual
overlays make per-depth trends hard to read; small multiples let each
depth be inspected independently while keeping nearby depths visually
adjacent.

Pure stdlib + inline SVG. Open with
``open experiments/beam-search-2x2/beam_curves.html``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "sync500_kmax20-baseline"
OUT_PATH = EXPERIMENT_DIR / "beam_curves.html"

BANDS = [
    ("Shallow (d=1–5)", [1, 2, 3, 4, 5]),
    ("Mid (d=6–10)", [6, 7, 8, 9, 10]),
    ("Deep (d=11–14)", [11, 12, 13, 14]),
]


def _load_cells(metrics_path: Path) -> list[dict]:
    cells: list[dict] = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("event") == "cell":
            cells.append(rec)
    return cells


def _load_histograms(hist_path: Path) -> dict[tuple[int, int], list[int]]:
    if not hist_path.exists():
        return {}
    payload = json.loads(hist_path.read_text())
    return {
        (int(c["depth"]), int(c["beam_width"])): c["solve_lens"]
        for c in payload.get("cells", [])
    }


def _index_cells(cells: list[dict]) -> dict[tuple[int, int], dict]:
    return {(int(c["depth"]), int(c["beam_width"])): c for c in cells}


def _grid_axes(cells: list[dict]) -> tuple[list[int], list[int]]:
    depths = sorted({int(c["depth"]) for c in cells})
    widths = sorted({int(c["beam_width"]) for c in cells})
    return depths, widths


def _x_for_width(w: int, widths: list[int]) -> float:
    """Return x-position on a unit interval [0, 1] for log-spaced widths.

    Powers-of-4 spacing maps cleanly to log positions; for arbitrary
    widths we use the index-of-w / (n_widths - 1).
    """
    if len(widths) == 1:
        return 0.5
    return widths.index(w) / (len(widths) - 1)


def _heatmap(cells: list[dict]) -> str:
    depths, widths = _grid_axes(cells)
    idx = _index_cells(cells)
    cell_w, cell_h = 80, 32
    pad_l, pad_t = 60, 50
    pad_r, pad_b = 16, 16

    plot_w = len(widths) * cell_w
    plot_h = len(depths) * cell_h
    total_w = pad_l + plot_w + pad_r
    total_h = pad_t + plot_h + pad_b + 30

    parts: list[str] = [
        f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
        f'<text x="{total_w / 2}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="600" fill="#222">Solve rate (depth × beam_width)</text>',
    ]

    # Column headers
    for j, w in enumerate(widths):
        cx = pad_l + j * cell_w + cell_w / 2
        parts.append(
            f'<text x="{cx}" y="{pad_t - 6}" text-anchor="middle" '
            f'font-size="11" fill="#444">w={w}</text>'
        )

    # Row labels
    for i, d in enumerate(depths):
        cy = pad_t + i * cell_h + cell_h / 2 + 4
        parts.append(
            f'<text x="{pad_l - 8}" y="{cy}" text-anchor="end" '
            f'font-size="11" fill="#444">d={d}</text>'
        )

    # Cells
    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            c = idx.get((d, w))
            x = pad_l + j * cell_w
            y = pad_t + i * cell_h
            if c is None:
                fill = "#eee"
                label = "—"
                text_color = "#888"
            else:
                rate = c["solve_rate"]
                # Color: red (0) → yellow (0.5) → green (1.0).
                if rate >= 0.999:
                    fill = "#137333"  # deep green — gate met
                elif rate >= 0.95:
                    fill = "#5db378"
                elif rate >= 0.75:
                    fill = "#a8d08d"
                elif rate >= 0.5:
                    fill = "#f4c542"
                elif rate >= 0.25:
                    fill = "#e88c5a"
                else:
                    fill = "#c84545"
                label = f"{rate:.2f}"
                text_color = "#fff" if rate < 0.25 or rate >= 0.999 else "#222"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{fill}" stroke="white" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" '
                f'text-anchor="middle" font-size="11" fill="{text_color}">'
                f"{label}</text>"
            )

    # Legend
    legend_y = pad_t + plot_h + 22
    legend_x = pad_l
    legend_items = [
        ("≥0.999", "#137333"),
        ("≥0.95", "#5db378"),
        ("≥0.75", "#a8d08d"),
        ("≥0.5", "#f4c542"),
        ("≥0.25", "#e88c5a"),
        ("<0.25", "#c84545"),
    ]
    for label, color in legend_items:
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="10" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y - 1}" '
            f'font-size="10" fill="#333">{label}</text>'
        )
        legend_x += 70

    parts.append("</svg>")
    return "\n".join(parts)


def _band_chart_lines(
    cells: list[dict],
    *,
    title: str,
    field: str,
    y_caps: dict[str, float],
    y_ticks: dict[str, list[float]],
    ref_lines: list[tuple[float, str, str]] | None = None,
    ref_per_depth: bool = False,
) -> str:
    """Render banded small-multiples for `field` vs beam_width.

    Each cell: x=beam_width (log-spaced), y=cell value at that
    (depth, beam_width). One line per band, one cell per depth.
    """
    _, widths = _grid_axes(cells)
    idx = _index_cells(cells)

    cell_w, cell_h = 200, 150
    pad = 16
    pad_l, pad_t, pad_r, pad_b = 36, 24, 10, 28
    plot_w = cell_w - pad_l - pad_r
    plot_h = cell_h - pad_t - pad_b

    # Figure out vertical extent for all bands.
    n_rows = len(BANDS)
    max_cells_per_row = max(len(d) for _, d in BANDS)
    total_w = pad + max_cells_per_row * (cell_w + pad)
    total_h = 30 + n_rows * (cell_h + 30 + pad)

    parts: list[str] = [
        f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
        f'<text x="{total_w / 2}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="600" fill="#222">{title}</text>',
    ]

    row_y = 30
    for band_label, depths_in_band in BANDS:
        y_cap = y_caps[band_label]
        ticks = y_ticks[band_label]

        # Band label
        parts.append(
            f'<text x="{pad}" y="{row_y + 16}" '
            f'font-size="12" font-weight="600" fill="#333">{band_label}</text>'
        )
        row_y += 22

        for i, depth in enumerate(depths_in_band):
            ox = pad + i * (cell_w + pad)
            oy = row_y

            def x_for(w: int, _ox: float = ox) -> float:
                return _ox + pad_l + _x_for_width(w, widths) * plot_w

            def y_for(v: float, _oy: float = oy, _cap: float = y_cap) -> float:
                cv = max(0.0, min(_cap, v))
                return _oy + pad_t + plot_h - (cv / _cap) * plot_h

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

            # y gridlines + labels
            for tick in ticks:
                if tick > y_cap:
                    continue
                gy = y_for(tick)
                parts.append(
                    f'<line x1="{ox + pad_l}" y1="{gy:.1f}" '
                    f'x2="{ox + pad_l + plot_w}" y2="{gy:.1f}" '
                    f'stroke="#eee" stroke-width="1"/>'
                )
                label = f"{tick:g}" if tick == int(tick) else f"{tick:.2f}"
                parts.append(
                    f'<text x="{ox + pad_l - 3}" y="{gy + 3:.1f}" '
                    f'text-anchor="end" font-size="9" fill="#888">{label}</text>'
                )

            # Reference lines
            for ref_v, ref_color, ref_label in ref_lines or []:
                v = ref_v
                if ref_per_depth:
                    v = depth  # for "scramble length" reference
                if v > y_cap:
                    continue
                ref_y = y_for(v)
                parts.append(
                    f'<line x1="{ox + pad_l}" y1="{ref_y:.1f}" '
                    f'x2="{ox + pad_l + plot_w}" y2="{ref_y:.1f}" '
                    f'stroke="{ref_color}" stroke-dasharray="3 3" stroke-width="1"/>'
                )
                if ref_label:
                    parts.append(
                        f'<text x="{ox + pad_l + plot_w - 3}" y="{ref_y - 2:.1f}" '
                        f'text-anchor="end" font-size="8" fill="{ref_color}">'
                        f"{ref_label}</text>"
                    )

            # x ticks (one per width)
            for w in widths:
                gx = x_for(w)
                parts.append(
                    f'<line x1="{gx:.1f}" y1="{oy + pad_t + plot_h}" '
                    f'x2="{gx:.1f}" y2="{oy + pad_t + plot_h + 3}" '
                    f'stroke="#888"/>'
                )
                parts.append(
                    f'<text x="{gx:.1f}" y="{oy + pad_t + plot_h + 14}" '
                    f'text-anchor="middle" font-size="9" fill="#666">{w}</text>'
                )

            # Data line
            pts: list[tuple[float, float, float | None]] = []
            for w in widths:
                c = idx.get((depth, w))
                v = c.get(field) if c else None
                pts.append((x_for(w), y_for(v) if v is not None else None, v))

            # Connect non-None points
            xy_def = [(px, py) for (px, py, val) in pts if py is not None]
            if len(xy_def) >= 2:
                d_attr = f"M {xy_def[0][0]:.2f} {xy_def[0][1]:.2f} " + " ".join(
                    f"L {x:.2f} {y:.2f}" for (x, y) in xy_def[1:]
                )
                parts.append(
                    f'<path d="{d_attr}" fill="none" stroke="#1f77b4" '
                    f'stroke-width="2"/>'
                )
            for px, py, _val in pts:
                if py is None:
                    continue
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#1f77b4"/>'
                )

        row_y += cell_h + pad

    parts.append("</svg>")
    return "\n".join(parts)


def _histogram_chart(
    histograms: dict[tuple[int, int], list[int]],
    cells: list[dict],
    beam_width: int,
) -> str:
    _, widths = _grid_axes(cells)
    if beam_width not in widths:
        return ""

    cell_w, cell_h = 200, 150
    pad = 16
    pad_l, pad_t, pad_r, pad_b = 30, 22, 10, 26
    plot_w = cell_w - pad_l - pad_r
    plot_h = cell_h - pad_t - pad_b

    n_rows = len(BANDS)
    max_cells_per_row = max(len(d) for _, d in BANDS)
    total_w = pad + max_cells_per_row * (cell_w + pad)
    total_h = 30 + n_rows * (cell_h + 30 + pad)

    parts: list[str] = [
        f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
        f'<text x="{total_w / 2}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="600" fill="#222">'
        f"Solve-length histograms at beam_width={beam_width} "
        f"(N={cells[0]['n_per_depth']}/depth)</text>",
    ]

    # Use a global x range across all bands (max plausible solve length)
    max_len = 0
    for (_, w), lens in histograms.items():
        if w != beam_width:
            continue
        for v in lens:
            if v > max_len:
                max_len = v
    x_max = max(max_len + 1, 14)  # at least the QTM diameter

    row_y = 30
    for band_label, depths_in_band in BANDS:
        # y-cap: per-band, scaled to the largest count in any cell of the band
        y_cap = 1
        for d in depths_in_band:
            lens = histograms.get((d, beam_width), [])
            if not lens:
                continue
            counts: dict[int, int] = {}
            for v in lens:
                counts[v] = counts.get(v, 0) + 1
            local_max = max(counts.values()) if counts else 0
            y_cap = max(y_cap, local_max)
        # round y_cap up to nearest 10 for nicer grids
        y_cap = max(10, int(math.ceil(y_cap / 10.0)) * 10)

        parts.append(
            f'<text x="{pad}" y="{row_y + 16}" '
            f'font-size="12" font-weight="600" fill="#333">{band_label}</text>'
        )
        row_y += 22

        for i, depth in enumerate(depths_in_band):
            ox = pad + i * (cell_w + pad)
            oy = row_y

            parts.append(
                f'<rect x="{ox + pad_l}" y="{oy + pad_t}" width="{plot_w}" '
                f'height="{plot_h}" fill="white" stroke="#ccc"/>'
            )

            lens = histograms.get((depth, beam_width), [])
            n_solved = sum(1 for v in lens if v >= 0)
            n_total = len(lens) if lens else cells[0]["n_per_depth"]
            n_failed = n_total - n_solved

            parts.append(
                f'<text x="{ox + cell_w / 2}" y="{oy + 14}" text-anchor="middle" '
                f'font-size="11" font-weight="600" fill="#333">'
                f"depth {depth}  (solved {n_solved}/{n_total})</text>"
            )

            # y ticks
            for tick in [0, y_cap // 2, y_cap]:
                gy = oy + pad_t + plot_h - (tick / y_cap) * plot_h
                parts.append(
                    f'<line x1="{ox + pad_l}" y1="{gy:.1f}" '
                    f'x2="{ox + pad_l + plot_w}" y2="{gy:.1f}" '
                    f'stroke="#eee"/>'
                )
                parts.append(
                    f'<text x="{ox + pad_l - 3}" y="{gy + 3:.1f}" '
                    f'text-anchor="end" font-size="9" fill="#888">{tick}</text>'
                )

            # Reference: scramble length d
            ref_x = ox + pad_l + (depth / x_max) * plot_w
            parts.append(
                f'<line x1="{ref_x:.1f}" y1="{oy + pad_t}" '
                f'x2="{ref_x:.1f}" y2="{oy + pad_t + plot_h}" '
                f'stroke="#aaa" stroke-dasharray="3 3" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{ref_x:.1f}" y="{oy + pad_t - 2}" text-anchor="middle" '
                f'font-size="8" fill="#888">d={depth}</text>'
            )

            # Bars (only for solved attempts)
            counts2: dict[int, int] = {}
            for v in lens:
                if v < 0:
                    continue
                counts2[v] = counts2.get(v, 0) + 1
            bar_unit = plot_w / x_max
            for length, count in counts2.items():
                bx = ox + pad_l + length * bar_unit
                bw = max(1.0, bar_unit - 1)
                bh = (count / y_cap) * plot_h
                by = oy + pad_t + plot_h - bh
                parts.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                    f'height="{bh:.1f}" fill="#1f77b4" opacity="0.85"/>'
                )

            # Failed-bar at x = x_max (off-axis)
            if n_failed > 0:
                fail_x = ox + pad_l + plot_w - 12
                fail_h = (n_failed / y_cap) * plot_h
                fail_y = oy + pad_t + plot_h - fail_h
                parts.append(
                    f'<rect x="{fail_x:.1f}" y="{fail_y:.1f}" width="10" '
                    f'height="{fail_h:.1f}" fill="#c84545" opacity="0.85"/>'
                )
                parts.append(
                    f'<text x="{fail_x + 5:.1f}" y="{oy + pad_t + plot_h + 12}" '
                    f'text-anchor="middle" font-size="8" fill="#c84545">fail</text>'
                )

            # x ticks
            for tick in [0, x_max // 2, x_max]:
                gx = ox + pad_l + (tick / x_max) * plot_w
                parts.append(
                    f'<line x1="{gx:.1f}" y1="{oy + pad_t + plot_h}" '
                    f'x2="{gx:.1f}" y2="{oy + pad_t + plot_h + 3}" '
                    f'stroke="#888"/>'
                )
                parts.append(
                    f'<text x="{gx:.1f}" y="{oy + pad_t + plot_h + 14}" '
                    f'text-anchor="middle" font-size="9" fill="#666">{tick}</text>'
                )

        row_y += cell_h + pad

    parts.append("</svg>")
    return "\n".join(parts)


def _y_caps_solve_rate() -> tuple[dict[str, float], dict[str, list[float]]]:
    caps = {b: 1.0 for b, _ in BANDS}
    ticks = {b: [0.0, 0.25, 0.5, 0.75, 1.0] for b, _ in BANDS}
    return caps, ticks


def _y_caps_excess() -> tuple[dict[str, float], dict[str, list[float]]]:
    return (
        {
            "Shallow (d=1–5)": 1.5,
            "Mid (d=6–10)": 3.0,
            "Deep (d=11–14)": 5.0,
        },
        {
            "Shallow (d=1–5)": [0.0, 0.5, 1.0, 1.5],
            "Mid (d=6–10)": [0.0, 1.0, 2.0, 3.0],
            "Deep (d=11–14)": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )


def _y_caps_avg_len(
    cells: list[dict],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    return (
        {
            "Shallow (d=1–5)": 8.0,
            "Mid (d=6–10)": 14.0,
            "Deep (d=11–14)": 20.0,
        },
        {
            "Shallow (d=1–5)": [0.0, 2.0, 4.0, 6.0, 8.0],
            "Mid (d=6–10)": [0.0, 4.0, 8.0, 12.0],
            "Deep (d=11–14)": [0.0, 5.0, 10.0, 15.0, 20.0],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    run_dir = args.run_dir
    cells = _load_cells(run_dir / "metrics.jsonl")
    histograms = _load_histograms(run_dir / "solve_histograms_beam.json")
    _, widths = _grid_axes(cells)

    sr_caps, sr_ticks = _y_caps_solve_rate()
    ex_caps, ex_ticks = _y_caps_excess()
    al_caps, al_ticks = _y_caps_avg_len(cells)

    parts: list[str] = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8">',
        "<title>beam-search-2x2 — beam curves</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; margin: 24px; "
        "background: #f3f3f3; color: #222; }",
        "h1 { font-size: 18px; margin: 0 0 8px 0; }",
        "h2 { font-size: 14px; margin: 24px 0 8px 0; color: #444; }",
        ".meta { font-size: 12px; color: #666; margin-bottom: 16px; }",
        ".chart { background: white; border: 1px solid #ddd; "
        "padding: 8px; margin-bottom: 16px; }",
        "</style></head><body>",
        "<h1>beam-search-2x2 — sweep</h1>",
        f'<div class="meta">Run: <code>{run_dir.relative_to(REPO_ROOT)}</code>. '
        f"Cells: {len(cells)}. Beam widths: {widths}.</div>",
        "<h2>Heatmap (overview)</h2>",
        f'<div class="chart">{_heatmap(cells)}</div>',
        "<h2>Solve rate vs beam_width (per depth)</h2>",
        '<div class="chart">'
        + _band_chart_lines(
            cells,
            title="Solve rate vs beam_width",
            field="solve_rate",
            y_caps=sr_caps,
            y_ticks=sr_ticks,
            ref_lines=[(0.999, "#137333", "gate 0.999")],
        )
        + "</div>",
        "<h2>Mean V*-excess vs beam_width (per depth)</h2>",
        '<div class="chart">'
        + _band_chart_lines(
            cells,
            title="Mean V*-excess vs beam_width",
            field="mean_v_star_excess",
            y_caps=ex_caps,
            y_ticks=ex_ticks,
            ref_lines=[(1.0, "#c84545", "gate ≤1.0")],
        )
        + "</div>",
        "<h2>Avg solve length vs beam_width (per depth)</h2>",
        '<div class="chart">'
        + _band_chart_lines(
            cells,
            title="Avg solve length vs beam_width",
            field="avg_solve_len",
            y_caps=al_caps,
            y_ticks=al_ticks,
            ref_lines=[(0, "#888", "scramble length d")],
            ref_per_depth=True,
        )
        + "</div>",
    ]

    if histograms:
        max_w = max(widths)
        parts.append(f"<h2>Solve-length histograms at beam_width={max_w}</h2>")
        parts.append(
            f'<div class="chart">{_histogram_chart(histograms, cells, max_w)}</div>'
        )

    parts.append("</body></html>")

    args.out.write_text("\n".join(parts))
    print(f"beam_curves.html written to {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
