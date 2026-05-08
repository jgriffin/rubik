"""Render a dedicated HTML report from one or more beam-eval payloads.

Inputs may be flat-schema per-checkpoint ``.json`` files or a consolidated
``.jsonl`` rollup (one payload per line). Mixed lists are accepted: each
``.jsonl`` is expanded to its constituent records, then concatenated with
any ``.json`` paths.

Three input *modes* are auto-detected:

1. **Single** — exactly one input. Renders the standard chart triplet
   (lines / wall-time bars / heatmap) for that one checkpoint.
2. **Trajectory** — multiple flat-schema inputs that all carry a
   ``step`` field with distinct values AND share a model parent
   directory (i.e. successive checkpoints from one training run, as
   produced by ``beam_eval_run.py``). Renders three trajectory-oriented
   charts:
     A. **Solve rate by depth, over training step** — banded small
        multiples (Shallow d=1..5 / Mid d=6..10 / Deep d=11..N / Very
        deep d=…), each band 5 cells wide except the last which may be
        narrower. Layout is derived from the actual max walk depth
        observed in the data, so K_max=14 keeps the legacy 5/5/4
        layout, K_max=18 produces 5/5/5/3, K_max=30 produces six
        bands of 5. x=step, y=solve_rate, one polyline per run.
        Matches the project's canonical banded chart idiom (see
        ``experiments/davi-3x3/analysis/render_error_trajectories.py``).
     B. **Solve rate by depth × training step (heatmap)** — x=step,
        y=walk depth with the deepest depth at the top, in-cell
        numeric labels.
     C. **Wall time across checkpoints** — x=step, y=seconds, line per run.
3. **Sweep** — multiple flat-schema inputs that don't qualify as a
   trajectory (same step or no step, but differ in ``beam_width`` /
   ``precision`` / etc.). Renders the standard chart triplet, with
   the legend keyed off the parameter that actually varies.

The legacy width-keyed schema (one JSON containing many widths) is
still accepted; each such input renders as its own section in the
sweep-style triplet layout.

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
import re
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


def trajectory_bands(max_depth: int) -> list[tuple[str, list[int]]]:
    """Return banded small-multiple layout for walk depths 1..max_depth.

    Project-wide convention: width-5 bands. Special-cased for
    backwards-compatibility with the legacy K_max=14 layout (5/5/4 with
    titles ``Shallow — walk-depths 1–5`` / ``Mid — walk-depths 6–10`` /
    ``Deep — walk-depths 11–14``).

    For other ``max_depth`` values, the first three bands carry the
    Shallow/Mid/Deep labels with their actual ranges; subsequent bands
    are labelled "Very deep d=A–B". When the deep band itself extends
    past d=14 (e.g. ``max_depth=15`` → Deep d=11–15), it stays a single
    band up to width 5; the next band starts at d=16.

    Examples::

        max_depth=14 → 5/5/4   ("Shallow … 1–5" / "Mid … 6–10" / "Deep … 11–14")
        max_depth=18 → 5/5/5/3 ("Shallow"/"Mid"/"Deep d=11–15"/"Very deep d=16–18")
        max_depth=30 → 5/5/5/5/5/5 (Shallow/Mid/Deep 11–15/Very deep 16–20/21–25/26–30)
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1; got {max_depth}")

    # Preserve the legacy K_max=14 titles exactly — existing test goldens
    # may match these strings.
    if max_depth == 14:
        return [
            ("Shallow — walk-depths 1–5", [1, 2, 3, 4, 5]),
            ("Mid — walk-depths 6–10", [6, 7, 8, 9, 10]),
            ("Deep — walk-depths 11–14", [11, 12, 13, 14]),
        ]

    bands: list[tuple[str, list[int]]] = []
    # Shallow: 1..min(5, max_depth)
    shallow_hi = min(5, max_depth)
    bands.append(
        (f"Shallow — walk-depths 1–{shallow_hi}", list(range(1, shallow_hi + 1)))
    )
    if max_depth <= 5:
        return bands

    # Mid: 6..min(10, max_depth)
    mid_hi = min(10, max_depth)
    bands.append((f"Mid — walk-depths 6–{mid_hi}", list(range(6, mid_hi + 1))))
    if max_depth <= 10:
        return bands

    # Deep: 11..min(15, max_depth) — width 5 to match Shallow/Mid.
    deep_hi = min(15, max_depth)
    bands.append((f"Deep — walk-depths 11–{deep_hi}", list(range(11, deep_hi + 1))))
    if max_depth <= 15:
        return bands

    # Very deep: width-5 bands starting at d=16.
    start = 16
    while start <= max_depth:
        end = min(start + 4, max_depth)
        bands.append(
            (f"Very deep — walk-depths {start}–{end}", list(range(start, end + 1)))
        )
        start = end + 1
    return bands


def _line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    cmds = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for x, y in points[1:]:
        cmds.append(f"L {x:.2f} {y:.2f}")
    return " ".join(cmds)


def _width_color(idx: int, n: int, base: str) -> str:
    """Map width-index to a color: shade the base from light → dark."""
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


def _format_step_label(step: int) -> str:
    """Compact training-step label: 20000 → '20k', 1500000 → '1.5M'."""
    if step >= 1_000_000:
        return f"{step / 1_000_000:g}M"
    if step >= 1_000:
        return f"{step / 1_000:g}k"
    return str(step)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def is_flat_schema(payload: dict) -> bool:
    """Detect the new flat schema: top-level ``per_walk_depth`` + ``beam_width``."""
    return "per_walk_depth" in payload and "beam_width" in payload


def _model_parent(payload: dict) -> str | None:
    """Return the parent dir of the JSON's ``model`` path, or None."""
    m = payload.get("model")
    if not m:
        return None
    return str(Path(m).parent)


def detect_input_mode(payloads: list[tuple[Path, dict]]) -> str:
    """Classify a set of flat-schema inputs as ``single``/``trajectory``/``sweep``.

    Trajectory criteria (ALL must hold):
      - >=2 inputs, all flat-schema
      - all have a non-None ``step`` field
      - the set of step values has size > 1 (not all equal)
      - all share the same ``model`` parent directory

    Single: exactly 1 input.
    Sweep: anything else (multiple inputs, but not a trajectory).
    """
    if len(payloads) <= 1:
        return "single"
    if not all(is_flat_schema(pl) for _, pl in payloads):
        return "sweep"
    steps = [pl.get("step") for _, pl in payloads]
    if any(s is None for s in steps):
        return "sweep"
    if len(set(steps)) <= 1:
        return "sweep"
    parents = {_model_parent(pl) for _, pl in payloads}
    if len(parents) != 1 or None in parents:
        return "sweep"
    return "trajectory"


def detect_varying_field(payloads: list[tuple[Path, dict]]) -> str | None:
    """For sweep mode: pick the parameter that varies across inputs.

    Preference order: ``beam_width``, ``precision``, ``config_name``,
    ``step``. Returns None if no preferred field varies (caller should
    fall back to the filename stem).
    """
    if len(payloads) <= 1:
        return None
    for field in ("beam_width", "precision", "config_name", "step"):
        vals = [pl.get(field) for _, pl in payloads]
        if all(v is not None for v in vals) and len(set(vals)) > 1:
            return field
    return None


# ---------------------------------------------------------------------------
# Sweep / single mode (legacy chart triplet)
# ---------------------------------------------------------------------------


def chart_solve_rate_lines(
    payload: dict,
    *,
    base_color: str,
    title: str = "Solve rate by walk depth (one line per beam width)",
    x_key: str = "d",
    x_label: str = "walk depth",
    items_key: str = "per_walk_depth",
    section_key: str = "results",
) -> str:
    """Solve-rate × walk-depth (or V*) lines, one per width / sweep entry."""
    width, height = 900, 380
    pad_l, pad_r, pad_t, pad_b = 56, 220, 36, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    section = payload.get(section_key, {})
    width_keys = _sort_section_keys(section.keys())
    if not width_keys:
        return f'<p class="note">_(no entries in {section_key})_</p>'

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
        f'font-weight="600" fill="#333">series</text>'
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
    title: str = "Wall time per series",
) -> str:
    """Bar chart: x=series-key, y=seconds."""
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
    title: str = "Solve rate by walk depth × series (heatmap)",
    x_key: str = "d",
    x_label: str = "walk depth",
    items_key: str = "per_walk_depth",
    section_key: str = "results",
) -> str:
    """Heatmap rows=x (walk-depth or V*), cols=series (width / sweep label)."""
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

    for ci, w_key in enumerate(width_keys):
        cx = pad_l + ci * cell_w + cell_w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t - 6}" text-anchor="middle" '
            f'font-size="11" fill="#444">{_format_legend_label(w_key)}</text>'
        )
    for ri, x_val in enumerate(x_values):
        ry = pad_t + ri * cell_h + cell_h / 2 + 4
        parts.append(
            f'<text x="{pad_l - 8}" y="{ry:.1f}" text-anchor="end" '
            f'font-size="11" fill="#444">{x_label}={x_val}</text>'
        )

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


# ---------------------------------------------------------------------------
# Trajectory mode — banded small multiples + reversed-Y heatmap + walltime
# ---------------------------------------------------------------------------


_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{8}T\d{6}Z_(.+)$")


def _friendly_run_label(run_dir_name: str) -> str:
    """Strip the leading ``YYYYMMDDTHHMMSSZ_`` timestamp prefix if present.

    Example: ``20260508T035718Z_smoke_ln_kmax30`` → ``smoke_ln_kmax30``.
    Falls back to the raw name when there's no timestamp prefix.
    """
    m = _TIMESTAMP_PREFIX_RE.match(run_dir_name)
    return m.group(1) if m else run_dir_name


def _trajectory_runs(
    payloads: list[tuple[Path, dict]],
) -> list[tuple[str, str, list[dict]]]:
    """Group trajectory payloads into one or more runs.

    Each "run" is a (label, color, sorted-by-step records) tuple. With
    the current beam_eval_run.py output, all checkpoints from one
    invocation share a model parent, so we collapse them into a single
    run colored ``INPUT_COLORS[0]``. (Multi-run overlay is left as a
    future extension once the use case appears.)
    """
    parents: dict[str, list[dict]] = {}
    for path, payload in payloads:
        parent = _model_parent(payload) or path.parent.as_posix()
        rec = dict(payload)
        rec["_path"] = path
        parents.setdefault(parent, []).append(rec)
    runs: list[tuple[str, str, list[dict]]] = []
    for i, (parent, recs) in enumerate(parents.items()):
        recs.sort(key=lambda r: int(r["step"]))
        raw = Path(parent).name or parent
        label = _friendly_run_label(raw)
        color = INPUT_COLORS[i % len(INPUT_COLORS)]
        runs.append((label, color, recs))
    return runs


def chart_trajectory_banded(
    runs: list[tuple[str, str, list[dict]]],
    *,
    title: str = "Solve rate by depth, over training step",
) -> str:
    """Banded width-5 small multiples: x=step, y=solve_rate, polyline per run.

    Layout is derived from the maximum walk depth observed in the input
    runs via ``trajectory_bands(max_depth)`` — K_max=14 keeps the legacy
    5/5/4 layout, K_max=18 produces 5/5/5/3, K_max=30 produces six bands.
    """
    cell_w, cell_h = 220, 160
    pad = 16
    pad_l, pad_t, pad_r, pad_b = 38, 24, 8, 26

    # Derive layout from the actual data, not a hardcoded constant. Falls
    # back to 14 (legacy) when there are no records — the empty-runs path
    # below short-circuits before any band SVG is built.
    observed_depths: set[int] = set()
    for _label, _color, recs in runs:
        for r in recs:
            for it in r.get("per_walk_depth", []):
                observed_depths.add(int(it["d"]))
    max_observed_depth = max(observed_depths) if observed_depths else 14
    bands = trajectory_bands(max_observed_depth)

    band_svgs: list[str] = []
    for band_title, depths in bands:
        cols = len(depths)
        total_w = cols * cell_w + pad * (cols + 1)
        total_h = cell_h + pad * 2 + 40
        plot_w = cell_w - pad_l - pad_r
        plot_h = cell_h - pad_t - pad_b

        all_steps = [int(r["step"]) for _, _, recs in runs for r in recs]
        max_step = max(all_steps, default=1) or 1
        min_step = min(all_steps, default=0)

        def y_for(v: float, oy: float, plot_h=plot_h, pad_t=pad_t) -> float:
            cv = max(0.0, min(1.0, v))
            return oy + pad_t + plot_h - cv * plot_h

        def x_for(
            s: float,
            ox: float,
            plot_w=plot_w,
            pad_l=pad_l,
            min_step=min_step,
            max_step=max_step,
        ) -> float:
            span = max_step - min_step or 1
            return ox + pad_l + (s - min_step) / span * plot_w

        parts: list[str] = [
            f'<svg viewBox="0 0 {total_w} {total_h}" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
            f'<text x="{total_w / 2}" y="22" text-anchor="middle" font-size="13" '
            f'font-weight="600" fill="#222">{band_title}</text>',
        ]

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

            # Y gridlines + labels (0, 0.25, 0.5, 0.75, 1.0).
            for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
                gy = y_for(tick, oy)
                parts.append(
                    f'<line x1="{ox + pad_l}" y1="{gy:.1f}" '
                    f'x2="{ox + pad_l + plot_w}" y2="{gy:.1f}" '
                    f'stroke="#eee" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{ox + pad_l - 3}" y="{gy + 3:.1f}" text-anchor="end" '
                    f'font-size="9" fill="#888">{tick:.2f}</text>'
                )

            # X ticks: a few representative step values.
            unique_steps = sorted({int(r["step"]) for _, _, recs in runs for r in recs})
            if unique_steps:
                if len(unique_steps) <= 4:
                    tick_steps = unique_steps
                else:
                    # First, last, and ~2 middle ticks.
                    idxs = [
                        0,
                        len(unique_steps) // 3,
                        2 * len(unique_steps) // 3,
                        len(unique_steps) - 1,
                    ]
                    tick_steps = sorted({unique_steps[i] for i in idxs})
                for s in tick_steps:
                    gx = x_for(s, ox)
                    parts.append(
                        f'<line x1="{gx:.1f}" y1="{oy + pad_t + plot_h}" '
                        f'x2="{gx:.1f}" y2="{oy + pad_t + plot_h + 3}" '
                        f'stroke="#888"/>'
                    )
                    parts.append(
                        f'<text x="{gx:.1f}" y="{oy + pad_t + plot_h + 14}" '
                        f'text-anchor="middle" font-size="9" fill="#666">'
                        f"{_format_step_label(s)}</text>"
                    )

            # One polyline per run.
            for _label, color, recs in runs:
                pts: list[tuple[float, float]] = []
                for r in recs:
                    s = int(r["step"])
                    items = {int(it["d"]): it for it in r.get("per_walk_depth", [])}
                    item = items.get(depth)
                    if item is None:
                        continue
                    rate = float(item["solve_rate"])
                    pts.append((x_for(s, ox), y_for(rate, oy)))
                if not pts:
                    continue
                parts.append(
                    f'<path d="{_line_path(pts)}" fill="none" stroke="{color}" '
                    f'stroke-width="1.8" opacity="0.95"/>'
                )
                for px, py in pts:
                    parts.append(
                        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.0" fill="{color}"/>'
                    )

        parts.append("</svg>")
        band_svgs.append("\n".join(parts))

    band_summaries = " / ".join(
        f"{btitle.split(' — ')[0]} d={depths[0]}–{depths[-1]}"
        for btitle, depths in bands
    )
    cell_widths = "/".join(str(len(depths)) for _, depths in bands)
    return (
        f'<p class="note">{title} — banded small multiples '
        f"({band_summaries}, {cell_widths} cells). "
        f"Each cell: x=training step, y=solve rate (0..1), one polyline per run."
        f"</p>\n" + "\n".join(band_svgs)
    )


def chart_trajectory_heatmap(
    runs: list[tuple[str, str, list[dict]]],
    *,
    title: str = "Solve rate by depth × training step",
) -> str:
    """Heatmap with x=step, y=walk_depth REVERSED (deepest at top, d=1 at bottom)."""
    if not runs:
        return '<p class="note">_(no runs)_</p>'
    # For now, render the first run's heatmap. (Multi-run heatmap overlay is
    # awkward in 2D; if needed later, render one per run vertically stacked.)
    _label, _color, recs = runs[0]
    if not recs:
        return '<p class="note">_(no records)_</p>'

    steps = sorted({int(r["step"]) for r in recs})
    walk_depths_present: set[int] = set()
    for r in recs:
        for it in r.get("per_walk_depth", []):
            walk_depths_present.add(int(it["d"]))
    walk_depths = sorted(walk_depths_present, reverse=True)  # d=14 at top.

    cell_w, cell_h = 76, 26
    # Extra top pad so rotated step ticks have room without clipping the title.
    pad_l, pad_r, pad_t, pad_b = 64, 16, 92, 56
    n_cols, n_rows = len(steps), len(walk_depths)
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

    # Column headers: step labels. Always rotate -45deg for clean,
    # non-overlapping ticks at any density.
    for ci, s in enumerate(steps):
        cx = pad_l + ci * cell_w + cell_w / 2
        ty = pad_t - 6
        parts.append(
            f'<text x="{cx:.1f}" y="{ty:.1f}" text-anchor="end" '
            f'font-size="11" fill="#444" '
            f'transform="rotate(-45 {cx:.1f} {ty:.1f})">'
            f"{_format_step_label(s)}</text>"
        )
    # X-axis label below the grid.
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 8}" '
        f'text-anchor="middle" font-size="11" fill="#333">training step</text>'
    )

    # Row headers: ``d=N`` (concise; matches per-cell label convention).
    for ri, d in enumerate(walk_depths):
        ry = pad_t + ri * cell_h + cell_h / 2 + 4
        parts.append(
            f'<text x="{pad_l - 8}" y="{ry:.1f}" text-anchor="end" '
            f'font-size="11" fill="#444">d={d}</text>'
        )
    # Y-axis label rotated on the left.
    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#333" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2:.1f})">walk depth</text>'
    )

    # Build (step, depth) -> rate index from the run's records.
    rate_map: dict[tuple[int, int], float] = {}
    for r in recs:
        s = int(r["step"])
        for it in r.get("per_walk_depth", []):
            rate_map[(s, int(it["d"]))] = float(it["solve_rate"])

    for ci, s in enumerate(steps):
        for ri, d in enumerate(walk_depths):
            rate = rate_map.get((s, d))
            if rate is None:
                continue
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
                f'text-anchor="middle" font-size="11" fill="{text_color}">'
                f"{rate:.2f}</text>"
            )

    parts.append("</svg>")
    return "\n".join(parts)


def chart_trajectory_walltime(
    runs: list[tuple[str, str, list[dict]]],
    *,
    title: str = "Wall time per checkpoint",
) -> str:
    """Line chart: x=step, y=wall_time_seconds, one line per run.

    When there is exactly one run, the legend is suppressed (it adds no
    information). When there are multiple, each run's friendly label
    appears in the legend.
    """
    width, height = 720, 280
    show_legend = len(runs) > 1
    if show_legend:
        pad_l, pad_r, pad_t, pad_b = 56, 180, 36, 50
    else:
        pad_l, pad_r, pad_t, pad_b = 56, 24, 36, 50
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    if not runs:
        return '<p class="note">_(no runs)_</p>'

    all_walls = [
        float(r.get("wall_time_seconds", 0.0)) for _, _, recs in runs for r in recs
    ]
    all_steps = [int(r["step"]) for _, _, recs in runs for r in recs]
    if not all_walls or not all_steps:
        return '<p class="note">_(no wall-time data)_</p>'
    y_max = max(all_walls) * 1.10
    if y_max <= 0:
        y_max = 1.0
    min_step, max_step = min(all_steps), max(all_steps)
    if min_step == max_step:
        max_step = min_step + 1

    def x_for(s: float) -> float:
        return pad_l + (s - min_step) / (max_step - min_step) * plot_w

    def y_for(v: float) -> float:
        return pad_t + plot_h - v / y_max * plot_h

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="600" fill="#222">{title}</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="white" stroke="#ccc"/>',
    ]
    for tk in range(5):
        v = y_max * tk / 4
        gy = y_for(v)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" '
            f'y2="{gy:.1f}" stroke="#eee"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="#666">{v:.1f}s</text>'
        )
    unique_steps = sorted(set(all_steps))
    for s in unique_steps:
        gx = x_for(s)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{pad_t + plot_h}" x2="{gx:.1f}" '
            f'y2="{pad_t + plot_h + 4}" stroke="#888"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
            f'font-size="10" fill="#666">step {_format_step_label(s)}</text>'
        )

    legend_x = pad_l + plot_w + 14
    legend_y = pad_t + 4
    for i, (label, color, recs) in enumerate(runs):
        pts = [
            (x_for(int(r["step"])), y_for(float(r.get("wall_time_seconds", 0.0))))
            for r in recs
        ]
        if pts:
            parts.append(
                f'<path d="{_line_path(pts)}" fill="none" stroke="{color}" '
                f'stroke-width="1.8"/>'
            )
            for px, py in pts:
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.4" fill="{color}"/>'
                )
        if show_legend:
            ly = legend_y + 16 + i * 16
            parts.append(
                f'<rect x="{legend_x}" y="{ly - 8}" width="14" height="10" '
                f'fill="{color}"/>'
            )
            parts.append(
                f'<text x="{legend_x + 18}" y="{ly + 1}" font-size="11" fill="#333">'
                f"{label}</text>"
            )

    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" '
        f'font-size="11" fill="#333" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2})">seconds</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sweep / single mode glue (legacy normalize → chart triplet)
# ---------------------------------------------------------------------------


def _legend_label_from_field(payload: dict, varying_field: str | None) -> str:
    """Build a legend label by leading with the varying field's value.

    Falls back to ``net_*`` filename or model stem when no preferred
    field varies.
    """
    if varying_field is not None:
        v = payload.get(varying_field)
        return f"{varying_field}={v}"
    # Fallback: use model filename stem.
    m = payload.get("model")
    if m:
        return Path(m).stem
    return "?"


def normalize_flat_payloads_for_sweep(
    payloads: list[tuple[Path, dict]],
    varying_field: str | None,
) -> dict:
    """Wrap flat-schema payloads as a single legacy-shaped overlay payload.

    The series key (``results`` keyed entries) is the varying field's
    value when one is detected, else the input filename stem. ``v_star_results``
    is similarly synthesized when any payload carries it.
    """
    first = payloads[0][1]
    results: dict[str, dict] = {}
    v_star_results: dict[str, dict] = {}
    any_v_star = False
    for _path, payload in payloads:
        label = _legend_label_from_field(payload, varying_field)
        results[label] = {
            "per_walk_depth": payload["per_walk_depth"],
            "wall_time_seconds": float(payload.get("wall_time_seconds", 0.0)),
            "states_scored": int(payload.get("states_scored", 0)),
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
        "beam_width": first.get("beam_width", "?"),
        "precision": first.get("precision", "?"),
        "results": results,
    }
    if any_v_star:
        out["v_star_results"] = v_star_results
    return out


def render_sweep_section(
    payload: dict,
    *,
    label: str,
    base_color: str,
    include_v_star: bool,
    varying_field: str | None,
    is_flat_overlay: bool = False,
) -> str:
    """Render the standard chart triplet for sweep / single / legacy inputs."""
    if is_flat_overlay:
        meta_line = (
            f"model: <code>{payload.get('model', '?')}</code><br>"
            f"config: <code>{payload.get('config_name', '?')}</code> · "
            f"device: <code>{payload.get('device', '?')}</code> · "
            f"max_depth: <code>{payload.get('max_walk_depth', '?')}</code> · "
            f"seed: <code>{payload.get('seed', '?')}</code>"
        )
        x_legend_title = varying_field or "input"
    else:
        meta_line = (
            f"checkpoint: <code>{payload.get('checkpoint', '?')}</code><br>"
            f"device: <code>{payload.get('device', '?')}</code> · "
            f"max_walk_depth: <code>{payload.get('max_walk_depth', '?')}</code> · "
            f"n_per_depth: <code>{payload.get('n_per_depth', '?')}</code> · "
            f"seed: <code>{payload.get('seed', '?')}</code>"
        )
        x_legend_title = "beam width"
    parts: list[str] = [
        f'<h2 style="margin-top:32px">{label}</h2>',
        f'<p class="note">{meta_line}</p>',
        f"<h3>1. Solve rate by walk depth (one line per {x_legend_title})</h3>",
        chart_solve_rate_lines(
            payload,
            base_color=base_color,
            title=(f"Solve rate by walk depth (one line per {x_legend_title})"),
        ),
        f"<h3>2. Wall time per {x_legend_title}</h3>",
        chart_wall_time_bars(
            payload,
            base_color=base_color,
            title=f"Wall time per {x_legend_title}",
        ),
        f"<h3>3. Solve rate by walk depth × {x_legend_title} (heatmap)</h3>",
        chart_heatmap(
            payload,
            title=f"Solve rate by walk depth × {x_legend_title} (heatmap)",
        ),
    ]
    if include_v_star and "v_star_results" in payload:
        parts.append(f"<h3>4. Solve rate by V* (one line per {x_legend_title})</h3>")
        parts.append(
            chart_solve_rate_lines(
                payload,
                base_color=base_color,
                title=f"Solve rate by V* (one line per {x_legend_title})",
                x_key="v",
                x_label="V*",
                items_key="per_v_star",
                section_key="v_star_results",
            )
        )
        parts.append(f"<h3>5. Solve rate by V* × {x_legend_title} (heatmap)</h3>")
        parts.append(
            chart_heatmap(
                payload,
                title=f"Solve rate by V* × {x_legend_title} (heatmap)",
                x_key="v",
                x_label="V*",
                items_key="per_v_star",
                section_key="v_star_results",
            )
        )
    return "\n".join(parts)


def render_trajectory_section(
    payloads: list[tuple[Path, dict]],
) -> str:
    """Render the trajectory-mode chart trio (banded / heatmap / walltime)."""
    runs = _trajectory_runs(payloads)
    first = payloads[0][1]
    run_dir = _model_parent(first) or "?"
    steps = sorted({int(pl["step"]) for _, pl in payloads})
    step_range = (
        f"{_format_step_label(steps[0])} → {_format_step_label(steps[-1])}"
        if steps
        else "?"
    )
    meta_line = (
        f"run dir: <code>{run_dir}</code><br>"
        f"checkpoints: <code>{len(payloads)}</code> "
        f"({step_range}) · "
        f"config: <code>{first.get('config_name', '?')}</code> · "
        f"device: <code>{first.get('device', '?')}</code> · "
        f"beam_width: <code>{first.get('beam_width', '?')}</code> · "
        f"precision: <code>{first.get('precision', '?')}</code>"
    )
    parts: list[str] = [
        '<h2 style="margin-top:32px">Beam-eval trajectory</h2>',
        f'<p class="note">{meta_line}</p>',
        "<h3>1. Solve rate by depth, over training step (banded)</h3>",
        chart_trajectory_banded(runs),
        "<h3>2. Solve rate by depth × training step (heatmap)</h3>",
        '<p class="note">'
        "x-axis: training step. y-axis: walk depth, deepest at the top "
        "(deepest walks first). Cells colored by solve_rate; numeric values "
        "shown in-cell."
        "</p>",
        chart_trajectory_heatmap(runs),
        "<h3>3. Wall time per checkpoint</h3>",
        chart_trajectory_walltime(runs),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more beam-eval JSON paths (output of beam_eval_*.py).",
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
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                payloads.append((p, json.loads(line)))
        else:
            payloads.append((p, json.loads(p.read_text())))

    flat_payloads = [(p, pl) for p, pl in payloads if is_flat_schema(pl)]
    legacy_payloads = [(p, pl) for p, pl in payloads if not is_flat_schema(pl)]

    sections: list[str] = []
    if flat_payloads:
        mode = detect_input_mode(flat_payloads)
        if mode == "trajectory":
            sections.append(render_trajectory_section(flat_payloads))
        else:
            varying_field = detect_varying_field(flat_payloads)
            merged = normalize_flat_payloads_for_sweep(flat_payloads, varying_field)
            any_v_star = "v_star_results" in merged
            if mode == "single":
                label = flat_payloads[0][0].name
            else:
                label = (
                    f"Beam-eval sweep ({varying_field})"
                    if varying_field
                    else "Beam-eval overlay"
                )
            sections.append(
                render_sweep_section(
                    merged,
                    label=label,
                    base_color=INPUT_COLORS[0],
                    include_v_star=any_v_star,
                    varying_field=varying_field,
                    is_flat_overlay=True,
                )
            )

    if legacy_payloads:
        any_v_star = any("v_star_results" in payload for _, payload in legacy_payloads)
        for i, (path, payload) in enumerate(legacy_payloads):
            color = INPUT_COLORS[i % len(INPUT_COLORS)]
            sections.append(
                render_sweep_section(
                    payload,
                    label=path.name,
                    base_color=color,
                    include_v_star=any_v_star,
                    varying_field=None,
                )
            )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Beam-eval report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 24px 32px; max-width: 1200px; color: #222; }}
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
    Generated by <code>scripts/render_beam_eval_report.py</code>. Modes
    auto-detected: <em>single</em> (one input) → standard chart triplet;
    <em>trajectory</em> (multiple checkpoints from one run) → banded
    per-depth solve-rate trajectories + step×depth heatmap (deepest on top)
    + wall time over step; <em>sweep</em> (multiple inputs varying in
    beam_width / precision / etc.) → standard chart triplet keyed off
    the varying parameter. Legacy width-keyed inputs render as their own
    sweep-style sections.
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
