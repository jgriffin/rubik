"""Render DAVI 3x3 runs' error trajectories as a static HTML/SVG page.

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

Mirrors ``experiments/davi-2x2/analysis/render_error_trajectories.py``
cell-for-cell. RUNS list is empty in the P1a scaffold and populated as
cycles land. When RUNS is empty (or no run dirs are populated), this
script still writes a valid-but-mostly-empty HTML so the eval pipeline
runs end-to-end on an empty ``runs/`` per the phase 1 acceptance gate.

## Chart layout convention (project-wide)

When a per-depth metric needs to be charted across the full {1..14}
walk-depth range, use **per-depth small multiples grouped by depth
range**, not multi-depth-overlay-on-one-axis. The grouping is:

- **Shallow** (depths 1–5): 5 cells per row, y-cap tight to the shallow
  scale.
- **Mid** (depths 6–10): 5 cells per row, y-cap tight to the mid scale.
- **Deep** (depths 11–14): 4 cells per row, y-cap tight to the deep
  scale.

This convention is shared with the davi-2x2 renderer; see that file's
docstring for the full rationale.

Pure stdlib + inline SVG. Open with
``open experiments/davi-3x3/results/error_trajectories.html``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = EXPERIMENT_DIR / "runs"

# (label, run_dir, color, dasharray) — empty until cycles land in P2b.
# Append tuples here to overlay new runs.
RUNS: list[tuple[str, str, str, str]] = []

# Walk-depth bins. Mirrors the 2x2 grid (1..14) per plan §P1c — same
# size for symmetry with davi-2x2's analysis pipeline; the 3x3 QTM
# diameter is 26 but the bounded-oracle / smoke-eval phase doesn't need
# the full range.
DEPTHS = list(range(1, 15))


def load_run(run_dir: str) -> list[dict]:
    """Return list of eval records (step, macro_mae, per_depth_mae).

    Returns an empty list if the run dir or metrics.jsonl is missing —
    keeps the renderer graceful on a not-yet-populated runs/ tree.
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
    """Top chart: macro_mae over step, all runs overlaid."""
    width, height = 900, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 40

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # x range: max step across runs
    max_step = max(
        (max((r["step"] for r in evals), default=0) for evals in runs_data.values()),
        default=0,
    )
    if max_step == 0:
        max_step = 1
    all_macro = [r["macro_mae"] for evals in runs_data.values() for r in evals]
    y_min, y_max = 0, (max(all_macro) * 1.05) if all_macro else 1.0

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
    # gate line at macro_mae = 1.0
    if y_max >= 1.0:
        parts.append(
            f'<line x1="{pad_l}" y1="{y(1.0):.1f}" x2="{pad_l + plot_w}" '
            f'y2="{y(1.0):.1f}" stroke="#000" stroke-dasharray="4 4" stroke-width="1"/>'
        )

    # Plot each run
    for _label, run_dir, color, _ in RUNS:
        evals = runs_data.get(run_dir, [])
        if not evals:
            continue
        pts = [(x(r["step"]), y(r["macro_mae"])) for r in evals]
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
        f'transform="rotate(-90 14 {pad_t + plot_h / 2})">macro_mae</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


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
    for label, run_dir, _, _ in RUNS:
        runs_data[run_dir] = load_run(run_dir)
        print(f"loaded {label}: {len(runs_data[run_dir])} eval records")

    hist_path = EXPERIMENT_DIR / "results" / "solve_histograms.json"
    if hist_path.exists():
        with hist_path.open() as f:
            json.load(f)
        print(f"loaded histograms file at {hist_path.relative_to(REPO_ROOT)}")
    else:
        print(f"WARN: histogram file not found at {hist_path}")

    out_path = EXPERIMENT_DIR / "results" / "error_trajectories.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not RUNS:
        body = (
            "<p class='note'>_(No runs registered yet. The first cycle lands in "
            "P2b — see <code>plans/m8-3x3-davi.md</code>. "
            "Append to <code>RUNS</code> in this script when a new run lands.)_</p>"
        )
    else:
        body = (
            f"{render_legend()}\n"
            f"<h2>1. Macro-MAE over training step</h2>\n"
            f"{chart_macro(runs_data)}\n"
            f"<p class='note'>_(Per-depth, solve-rate, avg-len, histogram, and "
            f"heatmap sections come online once the davi-2x2 chart helpers are "
            f"ported into this 3x3 renderer — happens at the same time the "
            f"first solve_histograms.json is captured. P2b/P3+ work.)_</p>"
        )

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
    Phase-1 scaffold. Eval is on the depth-stratified eval set built by
    <code>build_eval_set_3x3.py</code>; macro_mae averages across walk-depth
    bins (1..14). Bounded oracle V* is a side-channel (depths 0..6 only).
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
