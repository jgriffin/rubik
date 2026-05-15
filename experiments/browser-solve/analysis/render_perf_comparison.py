"""Render browser-solve perf comparison as one canonical inline-SVG page.

Reads ``results/latencies.jsonl`` (same file analyze.py consumes),
groups by ``(solver, width)``, and writes a single chart at
``results/perf-comparison.html``:

- x-axis: beam width (log scale; 32 / 64 / 128 / 256).
- y-axis: median wall-time per solve (log scale; ms).
- three lines: FastAPI/MPS, ONNX/WebGPU, ONNX/WASM (legend bottom-right).
- p10–p90 band per line as a light fill.

Inline SVG, pure stdlib — project convention forbids matplotlib for
visual artifacts. Reproducible from the JSONL alone.

RUNS table at the top — append a tuple per cycle and regenerate.

Usage::

    uv run python experiments/browser-solve/analysis/render_perf_comparison.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUT_HTML = RESULTS_DIR / "perf-comparison.html"

# (label, jsonl-path-relative-to-experiment-dir). Append a tuple per cycle.
RUNS: list[tuple[str, str]] = [
    ("first cut (N=10 × widths × 3 solvers)", "results/latencies.jsonl"),
]

# (solver, label, color)
SERIES = [
    ("fastapi", "FastAPI / MPS", "#1f77b4"),
    ("onnx-webgpu", "ONNX / WebGPU", "#2ca02c"),
    ("onnx-wasm", "ONNX / WASM", "#d62728"),
]
WIDTHS = [32, 64, 128, 256]


def _q(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    idx = p * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for _label, rel in RUNS:
        path = EXPERIMENT_DIR / rel
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, int], dict]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        w = float(r["wall_ms"])
        if w < 0:
            continue
        grouped[(r["solver"], int(r["width"]))].append(w)
    out: dict[tuple[str, int], dict] = {}
    for key, walls in grouped.items():
        out[key] = {
            "n": len(walls),
            "median": statistics.median(walls),
            "p10": _q(walls, 0.10),
            "p90": _q(walls, 0.90),
        }
    return out


def render_svg(summary: dict[tuple[str, int], dict]) -> str:
    # Collect all median values to fix y-axis bounds.
    all_ys: list[float] = []
    for (_solver, _w), s in summary.items():
        all_ys.extend([s["median"], s["p10"], s["p90"]])
    all_ys = [y for y in all_ys if y > 0]
    if not all_ys:
        return "<svg><text>no data</text></svg>"

    # Log-log space. y in ms, x in beam width.
    y_min = max(1.0, min(all_ys) * 0.8)
    y_max = max(all_ys) * 1.2
    log_y_min = math.log10(y_min)
    log_y_max = math.log10(y_max)
    log_x_min = math.log2(WIDTHS[0]) - 0.2
    log_x_max = math.log2(WIDTHS[-1]) + 0.2

    W, H = 760, 460
    MARGIN_L, MARGIN_R = 70, 30
    MARGIN_T, MARGIN_B = 40, 60
    plot_w = W - MARGIN_L - MARGIN_R
    plot_h = H - MARGIN_T - MARGIN_B

    def x_to_px(width_val: float) -> float:
        return MARGIN_L + (math.log2(width_val) - log_x_min) / (log_x_max - log_x_min) * plot_w

    def y_to_px(y_val: float) -> float:
        return (
            MARGIN_T
            + (1.0 - (math.log10(y_val) - log_y_min) / (log_y_max - log_y_min)) * plot_h
        )

    elems: list[str] = []
    # Background.
    elems.append(
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="white" />'
    )
    # Plot frame.
    elems.append(
        f'<rect x="{MARGIN_L}" y="{MARGIN_T}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#333" stroke-width="1" />'
    )
    # Y-axis log gridlines + labels (powers of 10).
    y_tick_pow = int(math.ceil(log_y_min))
    while y_tick_pow <= int(math.floor(log_y_max)):
        y_val = 10 ** y_tick_pow
        if y_min <= y_val <= y_max:
            y_px = y_to_px(y_val)
            elems.append(
                f'<line x1="{MARGIN_L}" y1="{y_px:.1f}" x2="{MARGIN_L + plot_w}" '
                f'y2="{y_px:.1f}" stroke="#ddd" stroke-width="0.5" />'
            )
            label = f"{int(y_val):,}" if y_val < 1000 else f"{int(y_val / 1000):,}k"
            elems.append(
                f'<text x="{MARGIN_L - 8}" y="{y_px + 4:.1f}" '
                f'text-anchor="end" font-size="11" fill="#444" '
                f'font-family="ui-monospace, Menlo, monospace">{label}</text>'
            )
        y_tick_pow += 1
    # X-axis: one tick per width.
    for w in WIDTHS:
        x_px = x_to_px(w)
        elems.append(
            f'<line x1="{x_px:.1f}" y1="{MARGIN_T}" x2="{x_px:.1f}" '
            f'y2="{MARGIN_T + plot_h}" stroke="#ddd" stroke-width="0.5" />'
        )
        elems.append(
            f'<text x="{x_px:.1f}" y="{MARGIN_T + plot_h + 18}" '
            f'text-anchor="middle" font-size="11" fill="#444" '
            f'font-family="ui-monospace, Menlo, monospace">{w}</text>'
        )
    # Axis labels.
    elems.append(
        f'<text x="{MARGIN_L + plot_w / 2}" y="{H - 18}" '
        f'text-anchor="middle" font-size="12" fill="#222">beam width (log₂)</text>'
    )
    elems.append(
        f'<text x="{18}" y="{MARGIN_T + plot_h / 2}" '
        f'text-anchor="middle" font-size="12" fill="#222" '
        f'transform="rotate(-90 18 {MARGIN_T + plot_h / 2})">'
        f'median wall (ms, log₁₀)</text>'
    )
    # Title.
    elems.append(
        f'<text x="{MARGIN_L + plot_w / 2}" y="22" text-anchor="middle" '
        f'font-size="14" fill="#111" font-weight="600">'
        f'M11 Block D — 3x3 solve latency vs beam width</text>'
    )

    # Lines + bands per series.
    for solver, label, color in SERIES:
        pts: list[tuple[float, float]] = []
        band_top: list[tuple[float, float]] = []
        band_bot: list[tuple[float, float]] = []
        for w in WIDTHS:
            s = summary.get((solver, w))
            if s is None:
                continue
            pts.append((x_to_px(w), y_to_px(s["median"])))
            band_top.append((x_to_px(w), y_to_px(s["p90"])))
            band_bot.append((x_to_px(w), y_to_px(s["p10"])))
        if len(pts) < 2:
            continue
        # p10-p90 band (filled).
        band_d_parts = [f"M {band_top[0][0]:.1f} {band_top[0][1]:.1f}"]
        for x, y in band_top[1:]:
            band_d_parts.append(f"L {x:.1f} {y:.1f}")
        for x, y in reversed(band_bot):
            band_d_parts.append(f"L {x:.1f} {y:.1f}")
        band_d_parts.append("Z")
        elems.append(
            f'<path d="{" ".join(band_d_parts)}" fill="{color}" fill-opacity="0.12" '
            f'stroke="none" />'
        )
        # Median line.
        line_d = " ".join(
            [f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"]
            + [f"L {x:.1f} {y:.1f}" for x, y in pts[1:]]
        )
        elems.append(
            f'<path d="{line_d}" fill="none" stroke="{color}" stroke-width="2" />'
        )
        # Median dots.
        for x, y in pts:
            elems.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />'
            )

    # Legend (bottom-right).
    legend_x = MARGIN_L + plot_w - 180
    legend_y = MARGIN_T + 15
    elems.append(
        f'<rect x="{legend_x - 8}" y="{legend_y - 12}" width="186" height="62" '
        f'fill="white" fill-opacity="0.85" stroke="#ccc" stroke-width="0.5" />'
    )
    for i, (_solver, label, color) in enumerate(SERIES):
        y = legend_y + i * 18
        elems.append(
            f'<rect x="{legend_x}" y="{y - 6}" width="14" height="3" fill="{color}" />'
        )
        elems.append(
            f'<circle cx="{legend_x + 7}" cy="{y - 4.5}" r="3" fill="{color}" />'
        )
        elems.append(
            f'<text x="{legend_x + 22}" y="{y - 1}" font-size="11" fill="#222" '
            f'font-family="ui-monospace, Menlo, monospace">{label}</text>'
        )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;max-width:100%;height:auto;">'
        + "\n".join(elems)
        + "</svg>"
    )
    return svg


def render_html(svg: str, summary: dict[tuple[str, int], dict]) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    # Table fallback (numbers-first).
    rows_html: list[str] = []
    for solver, label, _color in SERIES:
        for w in WIDTHS:
            s = summary.get((solver, w))
            if s is None:
                rows_html.append(
                    f"<tr><td>{label}</td><td>{w}</td><td>—</td><td>—</td>"
                    f"<td>—</td><td>—</td></tr>"
                )
            else:
                rows_html.append(
                    f"<tr><td>{label}</td><td>{w}</td><td>{s['n']}</td>"
                    f"<td>{s['median']:.0f}</td><td>{s['p10']:.0f}</td>"
                    f"<td>{s['p90']:.0f}</td></tr>"
                )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>M11 D — browser-solve perf comparison</title>
<style>
  body {{ font: 14px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif;
         color: #111; max-width: 860px; margin: 30px auto; padding: 0 20px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  h2 {{ font-size: 14px; margin: 24px 0 8px; color: #444; }}
  .meta {{ color: #888; font-size: 11px; margin-bottom: 18px; }}
  table {{ border-collapse: collapse; font: 12px ui-monospace, Menlo, monospace; }}
  th, td {{ padding: 4px 10px; border-bottom: 1px solid #eee; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>M11 Block D — 3x3 solve latency: FastAPI/MPS vs ONNX/WebGPU vs ONNX/WASM</h1>
<div class="meta">generated {now} · N=10 corpus rows × widths {{32, 64, 128, 256}} × 3 solvers</div>
{svg}
<h2>Numbers (median wall-ms; n is solved-rows in group)</h2>
<table>
  <thead>
    <tr><th>solver</th><th>width</th><th>n</th><th>median</th><th>p10</th><th>p90</th></tr>
  </thead>
  <tbody>
    {"".join(rows_html)}
  </tbody>
</table>
<p class="meta">Read <code>results.md</code> for run conditions + speedup ratios; <code>../intuition.md</code> for hypotheses.</p>
</body>
</html>
"""


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = summarize(rows)
    svg = render_svg(summary)
    html = render_html(svg, summary)
    OUT_HTML.write_text(html)
    print(f"wrote {OUT_HTML.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
