"""Compare two or more ``eval_payload.json`` files from ``eval_checkpoint.py``.

Renders a side-by-side comparison HTML page showing one bar per checkpoint
per (strategy, depth, method) cell. The same banded 5/5/4 small-multiples
layout the project uses for all per-depth charts.

Usage::

    uv run python scripts/compare_evals.py \\
        --payloads <pa.json> <pb.json> [<pc.json> ...] \\
        --labels A B [C ...] \\
        --out <out.html> \\
        [--title "Cycle 4 comparison"]

All payloads must share the same depths grid and strategy set; the script
asserts this and bails out otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Distinct, accessible palette. Up to 6 nets cleanly distinguishable.
NET_COLORS = ["#5d8cc9", "#c98c5d", "#5dc987", "#c95d8c", "#8c5dc9", "#c9c95d"]


def _index_cells(payload: dict) -> dict[tuple[str, str, int], dict]:
    return {
        (c["strategy"], c["method"], c["depth"]): c
        for c in payload["cells"]
    }


def _render_html(
    payloads: list[dict],
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    indexed = [_index_cells(p) for p in payloads]

    depths = list(payloads[0]["config"]["depths"])
    strategies = list(payloads[0]["config"]["strategies"])
    beam_width = payloads[0]["config"]["beam_width"]
    n_per_cell = payloads[0]["config"]["n_per_cell"]

    for p in payloads[1:]:
        if list(p["config"]["depths"]) != depths:
            raise ValueError("payloads disagree on depths")
        if list(p["config"]["strategies"]) != strategies:
            raise ValueError("payloads disagree on strategies")

    bands = [("Shallow", depths[0:5]), ("Mid", depths[5:10]), ("Deep", depths[10:])]
    if not bands[2][1]:
        bands = bands[:2]

    methods = ("greedy", "beam")

    def render_metric(metric: str, ylabel: str, ymax_fn) -> str:
        sections = []
        for strategy in strategies:
            method_blocks = []
            for method in methods:
                band_blocks = []
                for band_name, band_depths in bands:
                    cells_html = []
                    for d in band_depths:
                        rows = []
                        for net_idx, net_label in enumerate(labels):
                            cell = indexed[net_idx].get((strategy, method, d))
                            val = None if cell is None else cell.get(metric)
                            color = NET_COLORS[net_idx % len(NET_COLORS)]
                            ymax = ymax_fn(val)
                            pct = (
                                0.0
                                if val is None
                                else max(0.0, min(1.0, val / ymax)) * 100
                            )
                            v_str = "—" if val is None else f"{val:.3f}"
                            rows.append(
                                f'<div class="bar-row">'
                                f'  <div class="bar-label">{net_label}</div>'
                                f'  <div class="bar"><div class="bar-fill" '
                                f'style="background:{color}; width: {pct:.1f}%"></div>'
                                f'  <span class="bar-val">{v_str}</span></div>'
                                f"</div>"
                            )
                        cells_html.append(
                            f'<div class="cell">'
                            f'  <div class="d-label">d={d}</div>'
                            f'  {"".join(rows)}'
                            f"</div>"
                        )
                    band_blocks.append(
                        f'<div class="band">'
                        f"  <h5>{band_name}</h5>"
                        f'  <div class="band-grid">{"".join(cells_html)}</div>'
                        f"</div>"
                    )
                method_blocks.append(
                    f'<div class="method">'
                    f"  <h4>method: <code>{method}</code>"
                    f"  ({'width=1' if method == 'greedy' else f'width={beam_width}'})</h4>"
                    f'  <div class="bands">{"".join(band_blocks)}</div>'
                    f"</div>"
                )
            sections.append(
                f'<div class="strategy">'
                f"  <h3>strategy: <code>{strategy}</code></h3>"
                f'  {"".join(method_blocks)}'
                f"</div>"
            )
        return f"<h2>{ylabel}</h2>" + "".join(sections)

    def ymax_solve_rate(_v) -> float:
        return 1.0

    def ymax_excess(v) -> float:
        if v is None:
            return 1.0
        return max(1.0, min(10.0, v * 1.5))

    legend_swatches = "".join(
        f'<span><span class="swatch" '
        f'style="background:{NET_COLORS[i % len(NET_COLORS)]}"></span>'
        f'<b>{lbl}</b> &middot; <code>{Path(payloads[i]["checkpoint"]).parent.name}</code></span>'
        for i, lbl in enumerate(labels)
    )
    css = """
        body { font-family: -apple-system, system-ui, sans-serif; margin: 2em; color: #222; }
        h1 { font-size: 1.4em; }
        h2 { font-size: 1.15em; margin-top: 1.6em; border-bottom: 1px solid #ccc; }
        h3 { font-size: 1em; color: #555; margin-bottom: 0.4em; }
        h4 { font-size: 0.92em; color: #666; margin: 0.4em 0 0.2em 0.4em; }
        h5 { font-size: 0.85em; color: #888; margin: 0.4em 0 0.2em 0.4em; }
        .meta { color: #666; font-size: 0.9em; margin-bottom: 1em; }
        .strategy { margin-bottom: 1.4em; }
        .method { margin-left: 0.6em; margin-bottom: 0.8em; }
        .bands { display: flex; gap: 0.7em; }
        .band { flex: 1; }
        .band-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.4em; }
        .cell { background: #f8f8f8; padding: 0.4em; border-radius: 3px; font-size: 0.78em; }
        .d-label { font-weight: 600; margin-bottom: 0.2em; color: #444; }
        .bar-row { display: flex; align-items: center; gap: 0.3em; margin: 1px 0; }
        .bar-label { color: #666; width: 1.1em; font-size: 0.85em; font-weight: 600; }
        .bar { flex: 1; background: #e8e8e8; height: 13px; border-radius: 2px; position: relative; overflow: hidden; }
        .bar-fill { height: 100%; opacity: 0.75; }
        .bar-val { position: absolute; right: 4px; top: 0; line-height: 13px; font-size: 0.74em; color: #222; }
        legend { background: #fafafa; padding: 0.6em; border: 1px solid #eee; border-radius: 3px; font-size: 0.85em; margin: 1em 0; }
        .legend-row { display: flex; gap: 1.4em; align-items: center; flex-wrap: wrap; }
        .swatch { display: inline-block; width: 12px; height: 12px; vertical-align: middle; margin-right: 4px; opacity: 0.75; }
    """
    legend_html = (
        f'<legend><div class="legend-row">{legend_swatches}</div></legend>'
    )
    meta_html = (
        f'<div class="meta">'
        f"<b>n_per_cell:</b> {n_per_cell} &nbsp; "
        f'<b>depths:</b> {depths[0]}–{depths[-1]} &nbsp; '
        f'<b>beam_width:</b> {beam_width} &nbsp; '
        f"<b>methods:</b> greedy(w=1) + beam(w={beam_width})"
        f"</div>"
    )
    body = (
        f"<h1>{title}</h1>"
        f"{meta_html}"
        f"{legend_html}"
        f"{render_metric('solve_rate', 'Solve rate (per depth)', ymax_solve_rate)}"
        f"{render_metric('mean_v_star_excess', 'Mean V*-excess (per depth, solved attempts)', ymax_excess)}"
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{css}</style></head>"
        f"<body>{body}</body></html>"
    )
    out_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payloads",
        type=Path,
        nargs="+",
        required=True,
        help="paths to eval_payload.json files (one per checkpoint to compare)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="short labels for each payload (one per --payloads, same order)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Checkpoint comparison")
    args = parser.parse_args()

    if len(args.payloads) != len(args.labels):
        raise SystemExit("--payloads and --labels must have equal length")
    if len(args.payloads) < 2:
        raise SystemExit("--payloads needs at least 2 entries to compare")

    payloads = []
    for p in args.payloads:
        with p.open() as f:
            payloads.append(json.load(f))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _render_html(payloads, list(args.labels), args.title, args.out)
    try:
        out_display = args.out.relative_to(REPO_ROOT)
    except ValueError:
        out_display = args.out
    print(f"wrote {out_display}")


if __name__ == "__main__":
    main()
