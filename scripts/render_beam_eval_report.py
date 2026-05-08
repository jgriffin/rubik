"""Render a dedicated HTML report from one or more beam-eval JSONs.

Two input schemas are supported and auto-detected:

1. **Width-keyed schema (legacy)** — produced by the pre-refactor
   ``beam_eval_run.py``. Has ``results`` keyed by width string with
   per-width ``per_walk_depth`` lists. One input JSON renders a 3-chart
   set (lines / wall bars / heatmap) overlaying its widths.

2. **Flat schema (new)** — produced by ``beam_eval_model.py`` and
   ``beam_eval_sweep.py``. Each JSON has top-level ``per_walk_depth`` +
   ``beam_width`` (single scalar). The renderer wraps each flat-schema
   payload as a one-width entry in the same chart machinery, so multiple
   ``--input`` JSONs from a sweep overlay cleanly without code
   duplication.

When the input JSON(s) have ``v_star_results``, two extra sections
render the same charts for V*-stratified per-V* lists.

Multiple ``--input`` JSONs in flat-schema mode render as a single
overlay (one chart, one line per input). Legend labels prefer the
filename's ``_<param>=<value>`` suffix when present (sweep output) and
fall back to the filename stem otherwise.

Usage::

    uv run python scripts/render_beam_eval_report.py \\
        --input <path>.json [<path>...] \\
        [--output <path>]

Default output path is ``<first-input>.html`` (replacing ``.json`` with
``.html``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Distinct colors for multi-input overlays (colorblind-friendly).
INPUT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    cmds = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for x, y in points[1:]:
        cmds.append(f"L {x:.2f} {y:.2f}")
    return " ".join(cmds)


def _width_color(idx: int, n: int, base: str) -> str:
    """Map width-index to a color: shade the base from light → dark."""
    # Simple HSV-style ramp: fade from light gray to base color.
    base_rgb = _hex_to_rgb(base)
    if n <= 1:
        return base
    u = 0.15 + 0.85 * (idx / (n - 1))  # 0.15..1.0
    r = int(base_rgb[0] * u + 230 * (1 - u))
    g = int(base_rgb[1] * u + 230 * (1 - u))
    b = int(base_rgb[2] * u + 230 * (1 - u))
    return f"rgb({r},{g},{b})"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _sort_section_keys(keys) -> list[str]:
    """Sort keys numerically when they're all int-valued (legacy widths),
    else preserve insertion order (flat-schema overlay labels)."""
    keys_list = list(keys)
    try:
        return sorted(keys_list, key=lambda k: int(k))
    except (TypeError, ValueError):
        return keys_list


def _heat_color(rate: float) -> str:
    """Map solve_rate ∈ [0, 1] to a color (white=0 → dark blue=1)."""
    rate = max(0.0, min(1.0, float(rate)))
    # Blend white → #1f4e79 (dark blue).
    base = (31, 78, 121)
    r = int(255 * (1 - rate) + base[0] * rate)
    g = int(255 * (1 - rate) + base[1] * rate)
    b = int(255 * (1 - rate) + base[2] * rate)
    return f"rgb({r},{g},{b})"


def _format_legend_label(w_key: str) -> str:
    """Legend prefix: ``w=<n>`` for legacy width keys, raw label otherwise."""
    try:
        int(w_key)
        return f"w={w_key}"
    except (TypeError, ValueError):
        return str(w_key)


def chart_solve_rate_lines(
    payload: dict,
    *,
    base_color: str,
    title: str = "Per-walk-depth solve rate (one line per beam width)",
    x_key: str = "d",
    x_label: str = "walk depth",
    items_key: str = "per_walk_depth",
    section_key: str = "results",
) -> str:
    """Solve-rate × walk-depth (or V*) lines, one per width."""
    width, height = 900, 380
    pad_l, pad_r, pad_t, pad_b = 56, 180, 36, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    section = payload.get(section_key, {})
    width_keys = _sort_section_keys(section.keys())
    if not width_keys:
        return f'<p class="note">_(no entries in {section_key})_</p>'

    # Collect x-values from the first width entry (all widths share the same grid).
    first_pwd = section[width_keys[0]][items_key]
    x_values = [int(item[x_key]) for item in first_pwd]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_max = x_min + 1
    y_min, y_max = 0.0, 1.0

    def x_for(d: float) -> float:
        return pad_l + (d - x_min) / (x_max - x_min) * plot_w

    def y_for(v: float) -> float:
        return pad_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="#222">{title}</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="white" stroke="#ccc"/>',
    ]
    # Y gridlines (0, 0.25, 0.5, 0.75, 1).
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gy = y_for(tick)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            f'stroke="#eee"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="#666">{tick:.2f}</text>'
        )
    # X ticks at every integer.
    for d in x_values:
        gx = x_for(d)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{pad_t + plot_h}" x2="{gx:.1f}" '
            f'y2="{pad_t + plot_h + 4}" stroke="#666"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{pad_t + plot_h + 16}" text-anchor="middle" '
            f'font-size="10" fill="#666">{d}</text>'
        )
    # Axis labels.
    parts.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 8}" text-anchor="middle" '
        f'font-size="11" fill="#333">{x_label}</text>'
    )
    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" '
        f'font-size="11" fill="#333" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2})">solve rate</text>'
    )

    n_widths = len(width_keys)
    legend_x = pad_l + plot_w + 14
    legend_y = pad_t + 4
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-size="11" '
        f'font-weight="600" fill="#333">beam width</text>'
    )
    for i, w_key in enumerate(width_keys):
        color = _width_color(i, n_widths, base_color)
        entry = section[w_key]
        items = entry[items_key]
        pts = [
            (x_for(int(item[x_key])), y_for(float(item["solve_rate"])))
            for item in items
        ]
        parts.append(
            f'<path d="{_line_path(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" opacity="0.95"/>'
        )
        for px, py in pts:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="{color}"/>'
            )
        # Legend row.
        ly = legend_y + 16 + i * 16
        wall = entry.get("wall_time_seconds", 0.0)
        parts.append(
            f'<rect x="{legend_x}" y="{ly - 8}" width="14" height="10" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 18}" y="{ly + 1}" font-size="11" fill="#333">'
            f"{_format_legend_label(w_key)}  ({wall:.1f}s)</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def chart_wall_time_bars(
    payload: dict,
    *,
    base_color: str,
    section_key: str = "results",
    title: str = "Wall time per beam width",
) -> str:
    """Bar chart: x=width (log), y=seconds."""
    width, height = 720, 280
    pad_l, pad_r, pad_t, pad_b = 56, 16, 36, 50
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    section = payload.get(section_key, {})
    width_keys = _sort_section_keys(section.keys())
    if not width_keys:
        return f'<p class="note">_(no entries in {section_key})_</p>'

    walls = [float(section[k]["wall_time_seconds"]) for k in width_keys]
    y_max = max(walls) * 1.10 if walls else 1.0
    if y_max <= 0:
        y_max = 1.0

    n = len(width_keys)
    bar_slot = plot_w / max(n, 1)
    bar_w = bar_slot * 0.7

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="#222">{title}</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="white" stroke="#ccc"/>',
    ]
    # y ticks 5 levels.
    for tk in range(5):
        v = y_max * tk / 4
        gy = pad_t + plot_h - (v / y_max) * plot_h
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" '
            f'y2="{gy:.1f}" stroke="#eee"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="#666">{v:.1f}</text>'
        )
    for i, (w_key, wall) in enumerate(zip(width_keys, walls, strict=True)):
        color = _width_color(i, n, base_color)
        cx = pad_l + i * bar_slot + bar_slot / 2
        bx = cx - bar_w / 2
        bh = (wall / y_max) * plot_h
        by = pad_t + plot_h - bh
        legend_label = _format_legend_label(w_key)
        parts.append(
            f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" '
            f'height="{bh:.2f}" fill="{color}">'
            f"<title>{legend_label} wall={wall:.2f}s</title></rect>"
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t + plot_h + 16}" text-anchor="middle" '
            f'font-size="11" fill="#444">{legend_label}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{by - 4:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#222">{wall:.1f}s</text>'
        )

    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" '
        f'font-size="11" fill="#333" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2})">seconds</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def chart_heatmap(
    payload: dict,
    *,
    title: str = "Per-walk-depth × per-width solve-rate heatmap",
    x_key: str = "d",
    x_label: str = "walk depth",
    items_key: str = "per_walk_depth",
    section_key: str = "results",
) -> str:
    """Heatmap rows=x (walk-depth or V*), cols=width."""
    section = payload.get(section_key, {})
    width_keys = _sort_section_keys(section.keys())
    if not width_keys:
        return f'<p class="note">_(no entries in {section_key})_</p>'
    first_pwd = section[width_keys[0]][items_key]
    x_values = [int(item[x_key]) for item in first_pwd]

    cell_w, cell_h = 64, 22
    pad_l, pad_r, pad_t, pad_b = 60, 16, 56, 24
    n_cols, n_rows = len(width_keys), len(x_values)
    plot_w = n_cols * cell_w
    plot_h = n_rows * cell_h
    width = pad_l + plot_w + pad_r
    height = pad_t + plot_h + pad_b

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="#222">{title}</text>',
    ]

    # Column headers (widths or sweep labels).
    for ci, w_key in enumerate(width_keys):
        cx = pad_l + ci * cell_w + cell_w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t - 6}" text-anchor="middle" '
            f'font-size="11" fill="#444">{_format_legend_label(w_key)}</text>'
        )
    # Row headers (x values).
    for ri, x_val in enumerate(x_values):
        ry = pad_t + ri * cell_h + cell_h / 2 + 4
        parts.append(
            f'<text x="{pad_l - 8}" y="{ry:.1f}" text-anchor="end" '
            f'font-size="11" fill="#444">{x_label}={x_val}</text>'
        )

    # Cells.
    for ci, w_key in enumerate(width_keys):
        items_by_x = {int(it[x_key]): it for it in section[w_key][items_key]}
        for ri, x_val in enumerate(x_values):
            item = items_by_x.get(x_val)
            if item is None:
                continue
            rate = float(item["solve_rate"])
            cx = pad_l + ci * cell_w
            cy = pad_t + ri * cell_h
            color = _heat_color(rate)
            parts.append(
                f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w}" '
                f'height="{cell_h}" fill="{color}" stroke="#fff" stroke-width="1"/>'
            )
            text_color = "#fff" if rate >= 0.55 else "#222"
            parts.append(
                f'<text x="{cx + cell_w / 2:.1f}" y="{cy + cell_h / 2 + 4:.1f}" '
                f'text-anchor="middle" font-size="10" fill="{text_color}">'
                f"{rate:.2f}</text>"
            )

    parts.append("</svg>")
    return "\n".join(parts)


def is_flat_schema(payload: dict) -> bool:
    """Detect the new flat schema: top-level ``per_walk_depth`` + ``beam_width``."""
    return "per_walk_depth" in payload and "beam_width" in payload


def _legend_label_for_flat(path: Path) -> str:
    """Pull the ``_<param>=<value>`` suffix off a sweep-output filename, else stem.

    Examples:
        ``net_final_eval_fast_precision=bf16.json`` → ``precision=bf16``
        ``net_final_eval_fast.json``                  → ``net_final_eval_fast``
    """
    stem = path.stem
    if "=" in stem:
        # Take the trailing ``_<key>=<value>`` chunk if present.
        tail = stem.rsplit("_", 1)[-1]
        if "=" in tail:
            return tail
    return stem


def normalize_flat_payloads(
    payloads: list[tuple[Path, dict]],
) -> dict:
    """Wrap a list of flat-schema payloads as a single legacy-shaped payload.

    Synthesizes ``results`` keyed by a stable per-input label (sweep suffix
    or stem) so the existing chart functions can render the overlay
    without branching. ``v_star_results`` is similarly synthesized when
    any payload carries it.
    """
    first = payloads[0][1]
    # Use beam_width for x-axis ticks — we still need ``walk_depths`` for
    # the chart's x ticks, but the per_walk_depth lists themselves drive that.
    results: dict[str, dict] = {}
    v_star_results: dict[str, dict] = {}
    any_v_star = False
    for path, payload in payloads:
        label = _legend_label_for_flat(path)
        results[label] = {
            "per_walk_depth": payload["per_walk_depth"],
            "wall_time_seconds": float(payload.get("wall_time_seconds", 0.0)),
            "states_scored": int(payload.get("states_scored", 0)),
            "_legend_meta": {
                "beam_width": payload.get("beam_width", "?"),
                "precision": payload.get("precision", "?"),
            },
        }
        if "v_star_results" in payload:
            any_v_star = True
            v_star_results[label] = {
                "per_v_star": payload["v_star_results"]["per_v_star"],
                "wall_time_seconds": float(
                    payload["v_star_results"].get("wall_time_seconds", 0.0)
                ),
            }
    out: dict = {
        "model": first.get("model", "?"),
        "device": first.get("device", "?"),
        "max_walk_depth": first.get("max_depth", "?"),
        "n_per_depth": first.get("n_per_depth", "?"),
        "seed": first.get("seed", "?"),
        "config_name": first.get("config_name", "?"),
        "results": results,
    }
    if any_v_star:
        out["v_star_results"] = v_star_results
    return out


def render_input_section(
    payload: dict,
    *,
    label: str,
    base_color: str,
    include_v_star: bool,
    is_flat_overlay: bool = False,
) -> str:
    """Render the three (or five) charts for one input JSON."""
    if is_flat_overlay:
        # Flat-schema overlay: header reflects the sweep, not a single checkpoint.
        meta_line = (
            f"model: <code>{payload.get('model', '?')}</code><br>"
            f"config: <code>{payload.get('config_name', '?')}</code> · "
            f"device: <code>{payload.get('device', '?')}</code> · "
            f"max_depth: <code>{payload.get('max_walk_depth', '?')}</code> · "
            f"seed: <code>{payload.get('seed', '?')}</code>"
        )
        chart_x_legend_title = "input"
    else:
        meta_line = (
            f"checkpoint: <code>{payload.get('checkpoint', '?')}</code><br>"
            f"device: <code>{payload.get('device', '?')}</code> · "
            f"max_walk_depth: <code>{payload.get('max_walk_depth', '?')}</code> · "
            f"n_per_depth: <code>{payload.get('n_per_depth', '?')}</code> · "
            f"seed: <code>{payload.get('seed', '?')}</code>"
        )
        chart_x_legend_title = "beam width"
    parts: list[str] = [
        f'<h2 style="margin-top:32px">{label}</h2>',
        f'<p class="note">{meta_line}</p>',
        f"<h3>1. Per-walk-depth solve rate (lines per {chart_x_legend_title})</h3>",
        chart_solve_rate_lines(payload, base_color=base_color),
        f"<h3>2. Wall time per {chart_x_legend_title}</h3>",
        chart_wall_time_bars(payload, base_color=base_color),
        f"<h3>3. Per-walk-depth × per-{chart_x_legend_title} heatmap</h3>",
        chart_heatmap(payload),
    ]
    if include_v_star and "v_star_results" in payload:
        parts.append("<h3>4. Per-V* solve rate (lines per width)</h3>")
        parts.append(
            chart_solve_rate_lines(
                payload,
                base_color=base_color,
                title="Per-V* solve rate (one line per beam width)",
                x_key="v",
                x_label="V*",
                items_key="per_v_star",
                section_key="v_star_results",
            )
        )
        parts.append("<h3>5. Per-V* × per-width heatmap</h3>")
        parts.append(
            chart_heatmap(
                payload,
                title="Per-V* × per-width solve-rate heatmap",
                x_key="v",
                x_label="V*",
                items_key="per_v_star",
                section_key="v_star_results",
            )
        )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more beam-eval JSON paths (output of beam_eval_run.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <first-input>.html).",
    )
    args = parser.parse_args(argv)

    inputs: list[Path] = list(args.input)
    out_path: Path
    if args.output is not None:
        out_path = args.output
    else:
        first = inputs[0]
        out_path = first.with_suffix(".html")

    payloads: list[tuple[Path, dict]] = []
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"--input not found: {p}")
        payloads.append((p, json.loads(p.read_text())))

    flat_payloads = [(p, pl) for p, pl in payloads if is_flat_schema(pl)]
    legacy_payloads = [(p, pl) for p, pl in payloads if not is_flat_schema(pl)]

    sections: list[str] = []
    if flat_payloads:
        # Single overlay section across all flat-schema inputs.
        merged = normalize_flat_payloads(flat_payloads)
        any_v_star = "v_star_results" in merged
        label = (
            "Beam-eval overlay (sweep)"
            if len(flat_payloads) > 1
            else flat_payloads[0][0].name
        )
        sections.append(
            render_input_section(
                merged,
                label=label,
                base_color=INPUT_COLORS[0],
                include_v_star=any_v_star,
                is_flat_overlay=True,
            )
        )

    if legacy_payloads:
        # Legacy schema: one section per input, multi-width per section.
        any_v_star = any("v_star_results" in payload for _, payload in legacy_payloads)
        for i, (path, payload) in enumerate(legacy_payloads):
            color = INPUT_COLORS[i % len(INPUT_COLORS)]
            sections.append(
                render_input_section(
                    payload,
                    label=path.name,
                    base_color=color,
                    include_v_star=any_v_star,
                )
            )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Beam-eval report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 24px 32px; max-width: 1100px; color: #222; }}
    h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
    h2 {{ font-size: 15px; margin: 32px 0 6px 0; color: #333;
          border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    h3 {{ font-size: 13px; margin: 18px 0 6px 0; color: #555; }}
    p.note {{ font-size: 12px; color: #666; margin: 4px 0; }}
    code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 2px;
            font-size: 11px; }}
  </style>
</head>
<body>
  <h1>Beam-eval report</h1>
  <p class="note">
    Generated by <code>scripts/render_beam_eval_report.py</code> from
    <code>beam_eval_model.py</code> / <code>beam_eval_sweep.py</code>
    (flat schema) or legacy width-keyed outputs. Flat-schema inputs
    overlay into a single section; legacy inputs render one section
    each. The chart triplet (lines / wall-time bars / heatmap) shows
    capability vs. the swept axis (beam width or sweep label).
  </p>
  {"".join(sections)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
