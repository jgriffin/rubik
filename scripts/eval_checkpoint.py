"""Reusable V*-aware checkpoint evaluator (2x2).

The project's standard checkpoint-evaluation pattern. Takes any DAVI value-net
checkpoint and produces a side-by-side comparison across two state-sampling
strategies and two solve methods, at depths 1..14:

  Sampling strategies:
    - ``random_walk_depth`` — random-walk scramble of length d (V* ≤ d, biased
      toward smaller V* by walk redundancy). The "natural" eval shape used in
      M5/M6 — measures capability against the same input distribution training
      saw.
    - ``v_star_stratified`` — uniform-orbit sample at V* = d in the canonical
      (lex-min over the 24-rotation orbit) basis. Cleanly separates "the
      network knows V*=12 states" from "a random walk of length 12 happens
      to land on V*≤12 states." Canonical basis is required because the
      solver's ``is_solved`` checks strict equality against
      ``spec.solved_state``; a rotated V*=k state would only be recognized
      as solved if its random rotation happened to be the identity.

  Solve methods:
    - greedy (beam width=1) — argmin V_θ(child)
    - beam (default width=256) — M6 production setting

For each (strategy × depth × method) cell, captures solve rate, mean V*-excess,
average solve length, and full raw ``solve_lens`` / ``v_star_excess`` arrays.
Renders a banded small-multiples HTML page (5/5/4 per project convention)
showing solve-rate and V*-excess curves per strategy with greedy/beam overlay.

Usage::

    uv run python scripts/eval_checkpoint.py \\
        --checkpoint experiments/davi-2x2/runs/sync500_kmax20-30k/net_final.pt \\
        [--out-dir <dir>] \\
        [--config <run-dir/config.yaml>] \\
        [--n-per-cell 200] \\
        [--beam-width 256] \\
        [--max-steps 20]

If ``--config`` is omitted, the script looks for ``config.yaml`` next to the
checkpoint to recover the network architecture.

If ``--out-dir`` is omitted, the eval lands at
``<checkpoint-dir>/eval/<UTC-ts>/`` so successive evals don't clobber.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ValueNet
from rubik.oracle.v_star_2x2 import (
    load_v_star,
    load_v_star_arrays,
    sample_states_at_v_star,
)
from rubik.search import beam_solve_batch
from rubik.solve import compute_excess_vs_v_star
from rubik.training.config import DAVIConfig
from rubik.training.metric_logger import MetricLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
V_STAR_PATH = REPO_ROOT / "data" / "v_star_2x2.npz"

DEFAULT_DEPTHS = tuple(range(1, 15))
DEFAULT_BEAM_WIDTH = 256
DEFAULT_N_PER_CELL = 200
DEFAULT_MAX_STEPS = 20

STRATEGIES = ("random_walk_depth", "v_star_stratified")
METHOD_SPECS = ("greedy", "beam")  # rendered names; widths set per cell


def _resolve_config(checkpoint_path: Path, explicit: Path | None) -> DAVIConfig:
    if explicit is not None:
        return DAVIConfig.from_yaml(explicit)
    sibling = checkpoint_path.parent / "config.yaml"
    if sibling.exists():
        return DAVIConfig.from_yaml(sibling)
    raise FileNotFoundError(
        f"could not resolve net architecture: pass --config explicitly, or "
        f"place config.yaml next to {checkpoint_path}"
    )


def _resolve_out_dir(arg: Path | None, checkpoint_path: Path) -> Path:
    if arg is not None:
        return arg
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return checkpoint_path.parent / "eval" / ts


def _load_checkpoint_into(
    net: torch.nn.Module, path: Path, device: torch.device
) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "net_state" in ckpt:
        net.load_state_dict(ckpt["net_state"])
    else:
        net.load_state_dict(ckpt)


def _sample(
    strategy: str,
    depth: int,
    n: int,
    *,
    spec,
    states_arr: np.ndarray,
    depths_arr: np.ndarray,
    torch_gen: torch.Generator,
    np_rng: np.random.Generator,
) -> torch.Tensor:
    """Return ``(n, 24)`` int8 torch tensor (CPU) of starting states."""
    if strategy == "random_walk_depth":
        states_cpu, _ = random_scrambles(
            spec,
            batch_size=n,
            depth=depth,
            generator=torch_gen,
            prune_same_face=True,
        )
        return states_cpu
    if strategy == "v_star_stratified":
        # rotate=False — the solver's is_solved check requires strict equality
        # with canonical spec.solved_state; rotated V*=k states cannot reach a
        # solved target the solver recognizes.
        sample = sample_states_at_v_star(
            states_arr, depths_arr, depth, n=n, rng=np_rng, rotate=False
        )
        return torch.from_numpy(sample)
    raise ValueError(f"unknown strategy: {strategy!r}")


def _summarize(
    solve_lens: torch.Tensor,
    states_cpu: torch.Tensor,
    v_star: dict[bytes, int],
) -> dict:
    lens_list = solve_lens.cpu().tolist()
    excess = compute_excess_vs_v_star(solve_lens, states_cpu, v_star)
    excess_list = excess.tolist()
    n = len(lens_list)
    n_solved = sum(1 for x in lens_list if x >= 0)
    solve_rate = n_solved / n if n else 0.0
    solved_lens = [x for x in lens_list if x >= 0]
    avg_solve_len = sum(solved_lens) / len(solved_lens) if solved_lens else None
    solved_excess = [x for x in excess_list if x >= 0]
    mean_excess = sum(solved_excess) / len(solved_excess) if solved_excess else None
    return {
        "n": n,
        "n_solved": n_solved,
        "solve_rate": solve_rate,
        "avg_solve_len": avg_solve_len,
        "mean_v_star_excess": mean_excess,
        "solve_lens": lens_list,
        "v_star_excess": excess_list,
    }


def _render_html(payload: dict, out_path: Path) -> None:
    """Banded 5/5/4 small-multiples comparison page."""
    cells = payload["cells"]
    depths = payload["config"]["depths"]
    beam_width = payload["config"]["beam_width"]
    checkpoint = payload["checkpoint"]
    n_per_cell = payload["config"]["n_per_cell"]
    bands = [("Shallow", depths[0:5]), ("Mid", depths[5:10]), ("Deep", depths[10:14])]

    # Index by (strategy, method, depth) for chart construction.
    by_key: dict[tuple[str, str, int], dict] = {}
    for cell in cells:
        by_key[(cell["strategy"], cell["method"], cell["depth"])] = cell

    def metric_curves(metric: str, ylabel: str, ymax_fn) -> str:
        """Per-strategy per-depth metric, with greedy + beam overlay."""
        sections = []
        for strategy in STRATEGIES:
            band_blocks = []
            for band_name, band_depths in bands:
                pts = []
                for d in band_depths:
                    g = by_key.get((strategy, "greedy", d), {})
                    b = by_key.get((strategy, "beam", d), {})
                    pts.append(
                        {
                            "d": d,
                            "greedy": g.get(metric),
                            "beam": b.get(metric),
                        }
                    )
                cells_html = []
                for pt in pts:
                    g_val = pt["greedy"]
                    b_val = pt["beam"]
                    g_str = "—" if g_val is None else f"{g_val:.3f}"
                    b_str = "—" if b_val is None else f"{b_val:.3f}"
                    g_pct = (
                        0.0 if g_val is None else max(0.0, min(1.0, g_val / ymax_fn(pt)))
                    )
                    b_pct = (
                        0.0 if b_val is None else max(0.0, min(1.0, b_val / ymax_fn(pt)))
                    )
                    cells_html.append(
                        f'<div class="cell">'
                        f'  <div class="d-label">d={pt["d"]}</div>'
                        f'  <div class="bar-row">'
                        f'    <div class="bar-label">g</div>'
                        f'    <div class="bar"><div class="bar-fill greedy" '
                        f'style="width: {g_pct * 100:.1f}%"></div>'
                        f'    <span class="bar-val">{g_str}</span></div>'
                        f"  </div>"
                        f'  <div class="bar-row">'
                        f'    <div class="bar-label">b</div>'
                        f'    <div class="bar"><div class="bar-fill beam" '
                        f'style="width: {b_pct * 100:.1f}%"></div>'
                        f'    <span class="bar-val">{b_str}</span></div>'
                        f"  </div>"
                        f"</div>"
                    )
                band_blocks.append(
                    f'<div class="band">'
                    f"  <h4>{band_name}</h4>"
                    f'  <div class="band-grid">{"".join(cells_html)}</div>'
                    f"</div>"
                )
            sections.append(
                f'<div class="strategy">'
                f"  <h3>strategy: <code>{strategy}</code></h3>"
                f'  <div class="bands">{"".join(band_blocks)}</div>'
                f"</div>"
            )
        return f"<h2>{ylabel}</h2>" + "".join(sections)

    def ymax_solve_rate(_pt) -> float:
        return 1.0

    def ymax_excess(pt) -> float:
        # Cap at 5x the larger of the two values, bounded [1, 10] so the
        # chart stays legible even when one method is far better.
        vals = [v for v in (pt["greedy"], pt["beam"]) if v is not None]
        if not vals:
            return 1.0
        return max(1.0, min(10.0, max(vals) * 1.2))

    title = f"Eval — {checkpoint}"
    css = """
        body { font-family: -apple-system, system-ui, sans-serif; margin: 2em; color: #222; }
        h1 { font-size: 1.4em; }
        h2 { font-size: 1.15em; margin-top: 2em; border-bottom: 1px solid #ccc; }
        h3 { font-size: 1em; color: #555; margin-bottom: 0.4em; }
        h4 { font-size: 0.9em; color: #888; margin: 0.4em 0 0.2em 0.4em; }
        .meta { color: #666; font-size: 0.9em; margin-bottom: 1em; }
        .meta code { background: #f4f4f4; padding: 1px 4px; border-radius: 2px; }
        .strategy { margin-bottom: 1.4em; }
        .bands { display: flex; gap: 1em; }
        .band { flex: 1; }
        .band-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.4em; }
        .cell { background: #f8f8f8; padding: 0.4em; border-radius: 3px; font-size: 0.78em; }
        .d-label { font-weight: 600; margin-bottom: 0.2em; color: #444; }
        .bar-row { display: flex; align-items: center; gap: 0.3em; margin: 1px 0; }
        .bar-label { color: #888; width: 0.7em; font-size: 0.85em; }
        .bar { flex: 1; background: #e8e8e8; height: 14px; border-radius: 2px; position: relative; overflow: hidden; }
        .bar-fill { height: 100%; opacity: 0.7; }
        .bar-fill.greedy { background: #c98c5d; }
        .bar-fill.beam { background: #5d8cc9; }
        .bar-val { position: absolute; right: 4px; top: 0; line-height: 14px; font-size: 0.78em; color: #222; }
        legend { background: #fafafa; padding: 0.6em; border: 1px solid #eee; border-radius: 3px; font-size: 0.85em; margin: 1em 0; }
        .legend-row { display: flex; gap: 1.5em; align-items: center; }
        .swatch { display: inline-block; width: 12px; height: 12px; vertical-align: middle; margin-right: 4px; }
    """
    legend_html = (
        f'<legend><div class="legend-row">'
        f'<span><span class="swatch" style="background: #c98c5d; opacity: 0.7;"></span>'
        f'<b>g</b> = greedy (width=1)</span>'
        f'<span><span class="swatch" style="background: #5d8cc9; opacity: 0.7;"></span>'
        f'<b>b</b> = beam (width={beam_width})</span>'
        f"</div></legend>"
    )
    meta_html = (
        f'<div class="meta">'
        f"<b>checkpoint:</b> <code>{checkpoint}</code><br>"
        f"<b>n_per_cell:</b> {n_per_cell} &nbsp; "
        f'<b>depths:</b> {depths[0]}–{depths[-1]} &nbsp; '
        f"<b>beam_width:</b> {beam_width}<br>"
        f"<b>strategies:</b> "
        f"<code>random_walk_depth</code> (V*≤d, walk-distribution-biased) &nbsp;|&nbsp; "
        f"<code>v_star_stratified</code> (true V*=d, canonical basis)"
        f"</div>"
    )
    body = (
        f"<h1>{title}</h1>"
        f"{meta_html}"
        f"{legend_html}"
        f"{metric_curves('solve_rate', 'Solve rate (per depth)', ymax_solve_rate)}"
        f"{metric_curves('mean_v_star_excess', 'Mean V*-excess (per depth, solved attempts)', ymax_excess)}"
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{css}</style></head>"
        f"<body>{body}</body></html>"
    )
    out_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="DAVIConfig yaml (for net architecture). Defaults to config.yaml "
        "next to the checkpoint.",
    )
    parser.add_argument("--n-per-cell", type=int, default=DEFAULT_N_PER_CELL)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--depths",
        type=str,
        default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help="comma-separated list of depths to eval (default: 1..14)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    config = _resolve_config(args.checkpoint, args.config)
    out_dir = _resolve_out_dir(args.out_dir, args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    depths = tuple(int(s.strip()) for s in args.depths.split(",") if s.strip())

    device = torch.device(args.device)
    spec = CUBE_2X2

    net = ValueNet(
        spec,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
        normalization=config.normalization,
    ).to(device)
    _load_checkpoint_into(net, args.checkpoint, device)
    net.eval()

    if not V_STAR_PATH.exists():
        raise FileNotFoundError(
            f"V* cache not found at {V_STAR_PATH}. "
            "Run `uv run python scripts/build_v_star_2x2_cache.py` first."
        )
    v_star = load_v_star(V_STAR_PATH)
    states_arr, depths_arr = load_v_star_arrays(V_STAR_PATH)

    n_params = sum(p.numel() for p in net.parameters())
    try:
        ckpt_display = args.checkpoint.relative_to(REPO_ROOT)
    except ValueError:
        ckpt_display = args.checkpoint
    print(f"checkpoint:   {ckpt_display}")
    print(f"out_dir:      {out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir}")
    print(f"device:       {device}")
    print(f"params:       {n_params / 1e6:.2f}M")
    print(f"depths:       {list(depths)}")
    print(f"strategies:   {list(STRATEGIES)}")
    print(f"methods:      greedy(width=1) + beam(width={args.beam_width})")
    print(f"n_per_cell:   {args.n_per_cell}")
    print(f"max_steps:    {args.max_steps}")
    print()

    cells_payload: list[dict] = []
    method_widths = (("greedy", 1), ("beam", args.beam_width))

    with MetricLogger(out_dir / "eval.jsonl") as logger:
        logger.log(
            event="run_start",
            checkpoint=str(ckpt_display),
            n_params=n_params,
            device=str(device),
            depths=list(depths),
            strategies=list(STRATEGIES),
            beam_width=args.beam_width,
            n_per_cell=args.n_per_cell,
            max_steps=args.max_steps,
            seed=args.seed,
        )

        run_t0 = time.perf_counter()
        for strategy in STRATEGIES:
            for depth in depths:
                # Same seed per (strategy, depth) cell so cross-method/cross-net
                # comparisons see the same input distribution.
                torch_gen = torch.Generator(device="cpu").manual_seed(
                    args.seed + 100 * STRATEGIES.index(strategy) + depth
                )
                np_rng = np.random.default_rng(
                    args.seed + 1000 * STRATEGIES.index(strategy) + depth
                )
                states_cpu = _sample(
                    strategy,
                    depth,
                    args.n_per_cell,
                    spec=spec,
                    states_arr=states_arr,
                    depths_arr=depths_arr,
                    torch_gen=torch_gen,
                    np_rng=np_rng,
                )

                for method, width in method_widths:
                    t0 = time.perf_counter()
                    result = beam_solve_batch(
                        net,
                        spec,
                        states_cpu,
                        beam_width=width,
                        max_steps=args.max_steps,
                    )
                    elapsed = time.perf_counter() - t0
                    summary = _summarize(result.solve_lens, states_cpu, v_star)
                    cell_record = {
                        "strategy": strategy,
                        "method": method,
                        "depth": int(depth),
                        "beam_width": width,
                        **{
                            k: v
                            for k, v in summary.items()
                            if k not in ("solve_lens", "v_star_excess")
                        },
                        "n_expansions": result.n_expansions,
                        "elapsed_seconds": elapsed,
                    }
                    logger.log(event="cell", **cell_record)

                    avg_str = (
                        "—"
                        if summary["avg_solve_len"] is None
                        else f"{summary['avg_solve_len']:.2f}"
                    )
                    excess_str = (
                        "—"
                        if summary["mean_v_star_excess"] is None
                        else f"{summary['mean_v_star_excess']:.2f}"
                    )
                    print(
                        f"  {strategy:>20s}  d={depth:>2}  {method:>6s}(w={width:>3})  "
                        f"solved {summary['n_solved']:>3}/{summary['n']}  "
                        f"rate {summary['solve_rate']:.3f}  "
                        f"avg_len {avg_str}  excess {excess_str}  ({elapsed:.1f}s)",
                        flush=True,
                    )
                    cells_payload.append(
                        {
                            **cell_record,
                            "solve_lens": summary["solve_lens"],
                            "v_star_excess": summary["v_star_excess"],
                        }
                    )

        run_elapsed = time.perf_counter() - run_t0
        logger.log(
            event="run_end",
            elapsed_seconds=run_elapsed,
            n_cells=len(cells_payload),
        )

    payload = {
        "config": {
            "checkpoint": str(ckpt_display),
            "depths": list(depths),
            "strategies": list(STRATEGIES),
            "beam_width": args.beam_width,
            "n_per_cell": args.n_per_cell,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "checkpoint": str(ckpt_display),
        "cells": cells_payload,
    }
    payload_path = out_dir / "eval_payload.json"
    with payload_path.open("w") as f:
        json.dump(payload, f)

    html_path = out_dir / "eval.html"
    _render_html(payload, html_path)

    print()
    print(f"done. {len(cells_payload)} cells in {run_elapsed:.1f}s wall.")
    try:
        print(f"jsonl:    {(out_dir / 'eval.jsonl').relative_to(REPO_ROOT)}")
        print(f"payload:  {payload_path.relative_to(REPO_ROOT)}")
        print(f"html:     {html_path.relative_to(REPO_ROOT)}")
    except ValueError:
        print(f"out_dir:  {out_dir}")


if __name__ == "__main__":
    main()
