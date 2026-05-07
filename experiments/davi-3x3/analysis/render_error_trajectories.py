"""Render DAVI 3x3 runs' error trajectories as a static HTML/SVG page.

Reads each run's metrics.jsonl, extracts every ``event="eval"`` record,
and produces a stack of charts at ``error_trajectories.html``:

1. **Corrected macro-MAE trajectory** — overlaid line charts, x=step,
   y=corrected_macro (mean of v_star_mae/dN for N>=1).
2. **Per-V*-layer MAE small multiples** — one mini-chart per V* depth
   (1..6, the bounded oracle range from `build_v_star_oracle_3x3.py`).
3. **Per walk-depth pred_mean small multiples** — banded
   Shallow / Mid / Deep (5/5/4 cells per row). Shows the network
   pushing predicted V* up over training.
4. **Per walk-depth pred_std small multiples** — same banded layout.
5. **Beam capability** — bars from `<run>/results/beam_eval_focused.json`:
   per V* solve-rate (d=1..6) and per walk-depth solve-rate (d=1..14)
   for each captured checkpoint.
6. **Solve-length histograms** — graceful placeholder (TBD: extend
   capture_solve_histograms.py for the 3x3 schema).
7. **Per-run depth-step heatmap** — graceful placeholder (TBD: ≥2 runs).

## Schema notes

This renderer is adapted to the 3x3 ``value_eval`` schema (different
from 2x2). Eval records carry ``per_walk_depth/d{d}/pred_{mean,std}``
for d=1..14 and ``v_star_mae/d{d}`` for d=0..K (K=6, the bounded
oracle's max true depth). The recorded ``macro_v_star_mae`` field
sometimes erroneously included d=0 in the average (a bug since fixed
upstream); this renderer ignores that field and recomputes
``corrected_macro`` from the per-layer values.

## Chart layout convention (project-wide)

When a per-walk-depth metric needs to be charted across {1..14}, use
**per-depth small multiples grouped by depth range** (Shallow 1-5 / Mid
6-10 / Deep 11-14, 5/5/4 cells). Shared with davi-2x2; see that file's
docstring for the rationale.

Pure stdlib + inline SVG. Open with
``open experiments/davi-3x3/results/error_trajectories.html``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = EXPERIMENT_DIR / "runs"

# (label, run_dir, color, dasharray) — append a tuple per run.
# Color slot: pick distinct colors from a colorblind-friendly palette
# when adding subsequent runs (e.g. "#2ca02c", "#ff7f0e", "#d62728").
RUNS: list[tuple[str, str, str, str]] = [
    ("smoke 10k (K=8, sync=500)", "20260506T203408Z_smoke", "#1f77b4", ""),
    (
        "full_train 16k early-stop (K=20, sync=500)",
        "20260507T043533Z_full_train",
        "#ff7f0e",
        "",
    ),
]

# Walk-depth bins for per-walk-depth charts.
WALK_DEPTHS = list(range(1, 15))
# V* layer range emitted by the bounded BFS oracle (build_v_star_oracle_3x3.py
# K=6). Layer 0 is dropped from corrected_macro and from the V* MAE chart.
V_STAR_DEPTHS = list(range(1, 7))


def _corrected_macro(rec: dict) -> float | None:
    """Mean of v_star_mae/dN for N >= 1.

    Recomputed here because the run's recorded ``macro_v_star_mae``
    erroneously included d=0 on some evals (an upstream bug fixed after
    the smoke run was recorded). For runs where d=0 wasn't present, the
    recorded value matches this recomputation.
    """
    vals = []
    for d in V_STAR_DEPTHS:
        v = rec.get(f"v_star_mae/d{d}")
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_run(run_dir: str) -> list[dict]:
    """Return list of eval records, with normalized fields attached.

    Each returned record carries the original raw eval fields plus:
    - ``corrected_macro``: float, mean of v_star_mae/dN for N>=1.
    - ``v_star_mae``: dict{int -> float} of per-layer MAE (only present
      layers).
    - ``per_walk_pred_mean``: dict{int -> float} per walk-depth d.
    - ``per_walk_pred_std``: dict{int -> float} per walk-depth d.

    Returns an empty list if the run dir or metrics.jsonl is missing.
    """
    path = BASELINE_DIR / run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    evals: list[dict] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") != "eval":
                continue
            # Skip duplicate post-loop final eval (same step as last loop eval)
            if evals and evals[-1]["step"] == r["step"]:
                continue
            # Normalize per-V*-layer MAE
            v_star_mae: dict[int, float] = {}
            for d in V_STAR_DEPTHS:
                v = r.get(f"v_star_mae/d{d}")
                if v is not None:
                    v_star_mae[d] = float(v)
            r["v_star_mae"] = v_star_mae
            # Normalize per-walk-depth pred fields
            pwd_mean: dict[int, float] = {}
            pwd_std: dict[int, float] = {}
            for d in WALK_DEPTHS:
                m = r.get(f"per_walk_depth/d{d}/pred_mean")
                s = r.get(f"per_walk_depth/d{d}/pred_std")
                if m is not None:
                    pwd_mean[d] = float(m)
                if s is not None:
                    pwd_std[d] = float(s)
            r["per_walk_pred_mean"] = pwd_mean
            r["per_walk_pred_std"] = pwd_std
            r["corrected_macro"] = _corrected_macro(r)
            evals.append(r)
    return evals


def load_beam_capability(run_dir: str) -> dict | None:
    """Return parsed post-hoc beam-eval JSON, or None if absent.

    Two filenames are accepted for backward compatibility:
    - ``post_run_beam_eval.json`` — written by ``scripts/post_run_beam_eval.py``
      (the canonical post-hoc tool used for cycle-2 onward).
    - ``beam_eval_focused.json`` — legacy filename used by the smoke
      run (committed in 2b6814d). Same schema; we accept it so the smoke
      run keeps showing up in the overlaid charts without re-running the
      eval.

    If both are present, ``post_run_beam_eval.json`` wins.
    """
    base = BASELINE_DIR / run_dir / "results"
    for fname in ("post_run_beam_eval.json", "beam_eval_focused.json"):
        path = base / fname
        if path.exists():
            with path.open() as f:
                return json.load(f)
    return None


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    cmds = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for x, y in points[1:]:
        cmds.append(f"L {x:.2f} {y:.2f}")
    return " ".join(cmds)


def chart_macro(runs_data: dict[str, list[dict]]) -> str:
    """Top chart: corrected_macro over step, all runs overlaid.

    The y=1.0 reference line is kept (carry-over from the 2x2 acceptance
    gate). For 3x3 it's a reasonable scale anchor — the network has
    converged when per-layer MAE averages well below 1 — but it's not a
    formal acceptance gate at this scale.
    """
    width, height = 900, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 40

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    max_step = max(
        (max((r["step"] for r in evals), default=0) for evals in runs_data.values()),
        default=0,
    )
    if max_step == 0:
        max_step = 1
    all_macro = [
        r["corrected_macro"]
        for evals in runs_data.values()
        for r in evals
        if r.get("corrected_macro") is not None
    ]
    y_min, y_max = 0.0, (max(all_macro) * 1.05) if all_macro else 1.0

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
    # Reference line at y=1.0 — scale anchor, not a 3x3 acceptance gate.
    if y_max >= 1.0:
        parts.append(
            f'<line x1="{pad_l}" y1="{y(1.0):.1f}" x2="{pad_l + plot_w}" '
            f'y2="{y(1.0):.1f}" stroke="#000" stroke-dasharray="4 4" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l + plot_w - 8}" y="{y(1.0) - 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#000">scale anchor (corrected_macro = 1.0)</text>'
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
    for _label, run_dir, color, _ in RUNS:
        evals = runs_data.get(run_dir, [])
        if not evals:
            continue
        pts = [
            (x(r["step"]), y(r["corrected_macro"]))
            for r in evals
            if r.get("corrected_macro") is not None
        ]
        if not pts:
            continue
        parts.append(
            f'<path d="{line_path(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="2"/>'
        )
        for px, py in pts:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{color}"/>'
            )

    parts.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 8}" text-anchor="middle" '
        f'font-size="12" fill="#333">training step</text>'
    )
    parts.append(
        f'<text x="{14}" y="{pad_t + plot_h / 2}" text-anchor="middle" '
        f'font-size="12" fill="#333" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2})">'
        f'corrected_macro (V*=1..6 mean)</text>'
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
    value_for: Callable[[dict, int], float | None],
    ref_lines: list[tuple[float, str]] | None = None,
) -> str:
    """Generic banded small-multiples chart (one mini-chart per depth)."""
    cell_w, cell_h = 220, 160
    cols = max(len(depths), 1)
    pad = 16
    total_w = cols * cell_w + pad * (cols + 1)
    total_h = cell_h + pad * 2 + 40
    pad_l, pad_t, pad_r, pad_b = 32, 22, 8, 22
    plot_w = cell_w - pad_l - pad_r
    plot_h = cell_h - pad_t - pad_b

    max_step = max(
        (max((r["step"] for r in evals), default=0) for evals in runs_data.values()),
        default=1,
    )
    if max_step == 0:
        max_step = 1

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
                f'stroke="#eee" stroke-width="1"/>'
            )
            label = f"{tick:g}" if tick == int(tick) else f"{tick:.1f}"
            parts.append(
                f'<text x="{ox + pad_l - 3}" y="{gy + 3:.1f}" text-anchor="end" '
                f'font-size="9" fill="#888">{label}</text>'
            )

        for ref_v, ref_color in ref_lines or []:
            if ref_v > y_cap:
                continue
            ref_y = y_for(ref_v, oy)
            parts.append(
                f'<line x1="{ox + pad_l}" y1="{ref_y:.1f}" '
                f'x2="{ox + pad_l + plot_w}" y2="{ref_y:.1f}" '
                f'stroke="{ref_color}" stroke-dasharray="3 3" stroke-width="1"/>'
            )

        # Detect clipping
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
                f'text-anchor="end" font-size="10" fill="#c00">clipped</text>'
            )

        for _label, run_dir, color, _ in RUNS:
            evals = runs_data.get(run_dir, [])
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


def _v_star_mae_value(r: dict, depth: int) -> float | None:
    return r.get("v_star_mae", {}).get(depth)


def _pred_mean_value(r: dict, depth: int) -> float | None:
    return r.get("per_walk_pred_mean", {}).get(depth)


def _pred_std_value(r: dict, depth: int) -> float | None:
    return r.get("per_walk_pred_std", {}).get(depth)


def chart_v_star_layer_mae(runs_data: dict[str, list[dict]]) -> str:
    """Per-V*-layer MAE small multiples for d=1..6 (bounded oracle)."""
    return _band_chart(
        runs_data,
        title="Per-V*-layer MAE (bounded oracle, d=1..6)",
        depths=V_STAR_DEPTHS,
        y_cap=6.0,
        y_ticks=[0, 1, 2, 3, 4, 5, 6],
        value_for=_v_star_mae_value,
        ref_lines=[(1.0, "#aaa")],
    )


def chart_pred_mean_banded(runs_data: dict[str, list[dict]]) -> str:
    """Banded per walk-depth pred_mean charts.

    Each cell carries a per-cell horizontal reference at y = depth (the
    walk length), so it's easy to see when the network's prediction
    catches up to the random-walk distance.
    """
    bands = [
        ("Shallow pred_mean — walk-depths 1–5  (y 0..6)",
         [1, 2, 3, 4, 5], 6.0, [0, 1, 2, 3, 4, 5, 6]),
        ("Mid pred_mean — walk-depths 6–10  (y 0..10)",
         [6, 7, 8, 9, 10], 10.0, [0, 2, 4, 6, 8, 10]),
        ("Deep pred_mean — walk-depths 11–14  (y 0..14)",
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
            value_for=_pred_mean_value,
            per_cell_ref=lambda d: float(d),
            per_cell_ref_color="#888",
        ))
    return "\n".join(out)


def chart_pred_std_banded(runs_data: dict[str, list[dict]]) -> str:
    """Banded per walk-depth pred_std charts. Y-cap shared at 2.0."""
    bands = [
        ("Shallow pred_std — walk-depths 1–5", [1, 2, 3, 4, 5]),
        ("Mid pred_std — walk-depths 6–10", [6, 7, 8, 9, 10]),
        ("Deep pred_std — walk-depths 11–14", [11, 12, 13, 14]),
    ]
    return "\n".join(
        _band_chart(
            runs_data,
            title=title,
            depths=depths,
            y_cap=2.0,
            y_ticks=[0, 0.5, 1.0, 1.5, 2.0],
            value_for=_pred_std_value,
            ref_lines=None,
        )
        for title, depths in bands
    )


def _band_chart_with_per_cell_ref(
    runs_data: dict[str, list[dict]],
    *,
    title: str,
    depths: list[int],
    y_cap: float,
    y_ticks: list[float],
    value_for: Callable[[dict, int], float | None],
    per_cell_ref: Callable[[int], float],
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
        (max((r["step"] for r in evals), default=0) for evals in runs_data.values()),
        default=1,
    )
    if max_step == 0:
        max_step = 1

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
                f'text-anchor="end" font-size="9" fill="#888">'
                f'y = depth ({depth})</text>'
            )

        for _label, run_dir, color, _ in RUNS:
            evals = runs_data.get(run_dir, [])
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


def chart_beam_capability(runs_beam: dict[str, dict]) -> str:
    """Two side-by-side bar charts per run from beam_eval_focused.json.

    Left: per V* solve-rate (d=1..6).
    Right: per walk-depth solve-rate (d=1..14).

    Bars within a depth are dodged per checkpoint. Heights are 0..1.
    """
    if not any(runs_beam.values()):
        return (
            '<p class="note">_(no beam_eval_focused.json found in any run\'s '
            'results/ directory)_</p>'
        )

    parts: list[str] = []
    for label, run_dir, color, _ in RUNS:
        beam = runs_beam.get(run_dir)
        if not beam:
            continue
        ckpt_keys = sorted(beam.keys(), key=lambda k: int(k))
        parts.append(
            f'<h3 style="font-size:13px;margin:14px 0 6px 0;color:#444">'
            f'{label} — beam capability</h3>'
        )
        parts.append(_beam_capability_svg(beam, ckpt_keys, color))
    return "\n".join(parts)


def _beam_capability_svg(beam: dict, ckpt_keys: list[str], base_color: str) -> str:
    """Render two side-by-side bar groups: V* (1..6) and walk-depth (1..14)."""
    # Two adjacent panels in one SVG
    panel_w = 420
    panel_h = 300
    gap = 30
    total_w = panel_w * 2 + gap + 40  # 40 for outer margin
    total_h = panel_h + 60

    pad_l, pad_t, pad_r, pad_b = 36, 30, 8, 60

    def shade(base: str, idx: int, n: int) -> str:
        """Return base color (idx=0) or a desaturated version for later checkpoints."""
        # Simple: alternate base color and a lighter variant.
        if idx == 0 or n == 1:
            return base
        # Map base hex to a faded color by mixing with white
        base_rgb = _hex_to_rgb(base)
        u = 0.4 + 0.4 * (idx / max(1, n - 1))  # 0.4..0.8 mix toward white
        r = int(base_rgb[0] * (1 - u) + 255 * u)
        g = int(base_rgb[1] * (1 - u) + 255 * u)
        b = int(base_rgb[2] * (1 - u) + 255 * u)
        return f"rgb({r},{g},{b})"

    n_ck = len(ckpt_keys)

    parts: list[str] = [
        f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#fafafa"/>',
    ]

    # Legend strip at top
    legend_x = 20
    for idx, ck in enumerate(ckpt_keys):
        col = shade(base_color, idx, n_ck)
        parts.append(
            f'<rect x="{legend_x}" y="10" width="14" height="10" fill="{col}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 18}" y="19" font-size="11" fill="#333">'
            f'step {ck}</text>'
        )
        legend_x += 110

    def render_panel(
        ox: float,
        depths: list[int],
        title: str,
        key_prefix: str,
        section: str,
    ) -> None:
        plot_w = panel_w - pad_l - pad_r
        plot_h = panel_h - pad_t - pad_b
        # Frame
        parts.append(
            f'<text x="{ox + panel_w / 2}" y="40" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="#222">{title}</text>'
        )
        parts.append(
            f'<rect x="{ox + pad_l}" y="{pad_t + 25}" width="{plot_w}" '
            f'height="{plot_h}" fill="white" stroke="#ccc"/>'
        )
        # y ticks 0..1
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ty = pad_t + 25 + plot_h - tick * plot_h
            parts.append(
                f'<line x1="{ox + pad_l}" y1="{ty:.1f}" '
                f'x2="{ox + pad_l + plot_w}" y2="{ty:.1f}" stroke="#eee"/>'
            )
            parts.append(
                f'<text x="{ox + pad_l - 4}" y="{ty + 3:.1f}" text-anchor="end" '
                f'font-size="10" fill="#666">{tick:.2f}</text>'
            )

        # Bars: one group per depth, dodge per checkpoint
        n_depths = len(depths)
        group_w = plot_w / n_depths
        bar_w = max(1.0, group_w * 0.85 / max(1, n_ck))
        for i, d in enumerate(depths):
            cx = ox + pad_l + i * group_w + group_w / 2
            # x label
            parts.append(
                f'<text x="{cx:.1f}" y="{pad_t + 25 + plot_h + 14:.1f}" '
                f'text-anchor="middle" font-size="10" fill="#666">d{d}</text>'
            )
            for idx, ck in enumerate(ckpt_keys):
                section_data = beam[ck].get(section, {})
                key = f"{key_prefix}{d}"
                rate = section_data.get(key)
                if rate is None:
                    continue
                col = shade(base_color, idx, n_ck)
                bx = cx - (n_ck * bar_w) / 2 + idx * bar_w
                bh = float(rate) * plot_h
                by = pad_t + 25 + plot_h - bh
                parts.append(
                    f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" '
                    f'height="{bh:.2f}" fill="{col}">'
                    f'<title>step={ck} d{d} solve_rate={rate:.3f}</title></rect>'
                )

        # Axis labels
        parts.append(
            f'<text x="{ox + panel_w / 2:.1f}" y="{panel_h + 50:.1f}" '
            f'text-anchor="middle" font-size="11" fill="#444">depth</text>'
        )

    render_panel(
        20,
        V_STAR_DEPTHS,
        "Per V* solve-rate (beam, d=1..6)",
        "solve_rate_v",
        "beam_v_star",
    )
    render_panel(
        20 + panel_w + gap,
        WALK_DEPTHS,
        "Per walk-depth solve-rate (beam, d=1..14)",
        "solve_rate_d",
        "beam_walk",
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def render_legend() -> str:
    """Color legend used by macro and per-depth charts."""
    if not RUNS:
        return (
            '<div style="margin:12px 0;color:#888;font-size:12px">'
            "_(no runs registered yet — append to RUNS in this script.)_</div>"
        )
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
    runs_beam: dict[str, dict] = {}
    for label, run_dir, _, _ in RUNS:
        runs_data[run_dir] = load_run(run_dir)
        beam = load_beam_capability(run_dir)
        if beam is not None:
            runs_beam[run_dir] = beam
        print(
            f"loaded {label}: {len(runs_data[run_dir])} eval records, "
            f"beam_eval_focused: {'yes' if beam else 'no'}"
        )

    hist_path = EXPERIMENT_DIR / "results" / "solve_histograms.json"
    if hist_path.exists():
        print(f"note: {hist_path.relative_to(REPO_ROOT)} present "
              "(but the 3x3 histogram renderer is TBD — placeholder section)")
    else:
        print(f"note: histogram file not found at {hist_path} (placeholder section)")

    out_path = EXPERIMENT_DIR / "results" / "error_trajectories.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not RUNS:
        body = (
            "<p class='note'>_(No runs registered yet. Append to <code>RUNS</code> "
            "in this script when a new run lands.)_</p>"
        )
    else:
        body = f"""
  {render_legend()}

  <h2>1. Corrected macro-MAE over training step</h2>
  <p class="note">
    Macro is recomputed per record from <code>v_star_mae/d{{1..6}}</code>
    (mean across present layers). The recorded
    <code>macro_v_star_mae</code> field is ignored because some evals in
    the smoke run erroneously included <code>v_star_mae/d0</code> in the
    average; the renderer derives from per-layer values to avoid that
    contamination. y=1.0 is a scale anchor only (not a 3x3 acceptance
    gate).
  </p>
  {chart_macro(runs_data)}

  <h2>2. Per-V*-layer MAE — bounded oracle (d=1..6)</h2>
  <p class="note">
    One mini-chart per V*-true depth from the bounded BFS oracle (K=6).
    Cells are populated only when at least one walk landed on a state
    with that V*-true depth in the eval batch — sparse at deeper layers
    early in training. Dashed grey reference at MAE=1.
  </p>
  {chart_v_star_layer_mae(runs_data)}

  <h2>3. Per walk-depth pred_mean — banded</h2>
  <p class="note">
    For each walk-depth d∈{{1..14}}, the network's mean predicted V*
    over states reached by random walks of length d. The dashed grey
    reference at <code>y = depth</code> shows where the prediction would
    sit if the network identified each walk's length as its V*. In
    practice walks fold back so true V* is ≤ d; the line should rise
    over training and sit somewhat below the reference.
  </p>
  {chart_pred_mean_banded(runs_data)}

  <h2>4. Per walk-depth pred_std — banded</h2>
  <p class="note">
    Spread of predictions per walk-depth. Useful as a coarse calibration
    signal; very low std at deep walks usually means the network
    collapsed onto a constant.
  </p>
  {chart_pred_std_banded(runs_data)}

  <h2>5. Beam capability (post-hoc, from <code>beam_eval_focused.json</code>)</h2>
  <p class="note">
    Beam(width=256) solve rates at the captured checkpoint(s). Two
    panels: per V*-true depth (d=1..6, BFS oracle) and per walk-depth
    (d=1..14). Live <code>solve_rate_dN</code> / <code>avg_solve_len_dN</code>
    are intentionally NOT captured during 3x3 training — beam cost would
    slow the training loop, per plan §P1c. The capability picture lives
    here, post-hoc.
  </p>
  {chart_beam_capability(runs_beam)}

  <h2>6. Solve-length histograms — placeholder</h2>
  <p class="note">
    _(post-hoc per-attempt histograms — TODO when
    <code>capture_solve_histograms.py</code> is extended for the 3x3
    schema. Section will populate once that capture pipeline lands.)_
  </p>

  <h2>7. Per-run wavefront heatmap — placeholder</h2>
  <p class="note">
    _(per-run depth-step heatmap — TODO when there are ≥2 3x3 runs.
    With a single run the cross-run wavefront comparison is degenerate;
    the section unlocks when a second cycle lands.)_
  </p>
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DAVI 3x3 — error trajectories</title>
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
  <h1>M8 DAVI 3x3 — error trajectories</h1>
  <p class="note">
    Eval is on the random-walk eval set + bounded V* oracle (K=6). Walks
    of length 1..14 generate scrambled states; walks that land within
    the bounded oracle contribute to <code>v_star_mae/d{{0..6}}</code>.
    All walks contribute to <code>per_walk_depth/d{{1..14}}</code>
    pred_mean/pred_std. The corrected macro is the mean of
    <code>v_star_mae/d{{1..6}}</code>.
  </p>
  {body}
</body>
</html>
"""
    out_path.write_text(html)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    print(f"open with:  open {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
