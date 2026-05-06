"""Beam-search evaluation entry point.

Reads a ``BeamEvalConfig`` YAML, loads a trained value-net checkpoint,
and runs ``beam_solve_batch`` across a ``(depth, beam_width)`` grid.

For each ``depth``, generates ``n_per_depth`` random scrambles ONCE, then
sweeps ``beam_widths`` against the *same* states so cross-width comparison
is crisp (not muddied by different scramble samples). Per-cell metrics
land in ``metrics.jsonl`` (``event="cell"`` records); raw per-attempt
``solve_lens`` and ``v_star_excess`` arrays land in
``solve_histograms_beam.json`` for post-hoc histogram rendering.

Usage::

    uv run python experiments/beam-search-2x2/run.py \\
        --config experiments/beam-search-2x2/configs/<name>.yaml \\
        [--out-dir experiments/beam-search-2x2/runs/<custom>]

If ``--out-dir`` is omitted, the run lands at
``experiments/beam-search-2x2/runs/<UTC-ts>_<config-stem>/``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ValueNet
from rubik.oracle.v_star_2x2 import load_v_star
from rubik.search import BeamEvalConfig, beam_solve_batch
from rubik.solve import compute_excess_vs_v_star
from rubik.training.metric_logger import MetricLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "experiments" / "beam-search-2x2" / "runs"
V_STAR_PATH = REPO_ROOT / "data" / "v_star_2x2.npz"


def _resolve_out_dir(arg: Path | None, config_stem: str) -> Path:
    if arg is not None:
        return arg
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{ts}_{config_stem}"


def _load_checkpoint_into(
    net: torch.nn.Module, path: Path, device: torch.device
) -> None:
    """Load a checkpoint (dict or legacy bare state_dict) into ``net``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "net_state" in ckpt:
        net.load_state_dict(ckpt["net_state"])
    else:
        net.load_state_dict(ckpt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = BeamEvalConfig.from_yaml(args.config)
    out_dir = _resolve_out_dir(args.out_dir, args.config.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_dir / "config.yaml")

    device = torch.device(config.device)
    spec = CUBE_2X2

    ckpt_path = REPO_ROOT / config.checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found at {ckpt_path}")

    net = ValueNet(
        spec,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
        normalization=config.normalization,
    ).to(device)
    _load_checkpoint_into(net, ckpt_path, device)
    net.eval()

    if not V_STAR_PATH.exists():
        raise FileNotFoundError(
            f"V* cache not found at {V_STAR_PATH}. "
            "Run `uv run python scripts/build_v_star_2x2_cache.py` first."
        )
    v_star = load_v_star(V_STAR_PATH)

    n_params = sum(p.numel() for p in net.parameters())
    try:
        run_dir_display = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        run_dir_display = out_dir
    print(f"run dir:        {run_dir_display}")
    print(f"checkpoint:     {config.checkpoint_path}")
    print(f"device:         {device}")
    print(f"params:         {n_params / 1e6:.2f}M")
    print(f"depths:         {list(config.depths)}")
    print(f"beam_widths:    {list(config.beam_widths)}")
    print(f"n_per_depth:    {config.n_per_depth}")
    print(f"max_steps:      {config.max_steps}")
    print(f"V* states:      {len(v_star):,}")
    print()

    # Per-cell raw arrays — captured for the histogram renderer.
    cells_payload: list[dict] = []

    with MetricLogger(out_dir / "metrics.jsonl") as logger:
        logger.log(
            event="run_start",
            checkpoint_path=config.checkpoint_path,
            n_params=n_params,
            device=str(device),
            body_widths=list(config.body_widths),
            n_residual_blocks=config.n_residual_blocks,
            normalization=config.normalization,
            depths=list(config.depths),
            beam_widths=list(config.beam_widths),
            n_per_depth=config.n_per_depth,
            max_steps=config.max_steps,
        )

        run_t0 = time.perf_counter()
        for depth in config.depths:
            # Same scrambles across all beam widths for this depth.
            gen = torch.Generator(device="cpu").manual_seed(config.seed + int(depth))
            states_cpu, _ = random_scrambles(
                spec,
                batch_size=config.n_per_depth,
                depth=int(depth),
                generator=gen,
                prune_same_face=True,
            )

            for beam_width in config.beam_widths:
                t0 = time.perf_counter()
                result = beam_solve_batch(
                    net,
                    spec,
                    states_cpu,
                    beam_width=int(beam_width),
                    max_steps=config.max_steps,
                )
                elapsed = time.perf_counter() - t0

                lens_list = result.solve_lens.cpu().tolist()
                excess = compute_excess_vs_v_star(result.solve_lens, states_cpu, v_star)
                excess_list = excess.tolist()

                n_solved = sum(1 for x in lens_list if x >= 0)
                solve_rate = n_solved / config.n_per_depth
                solved_lens = [x for x in lens_list if x >= 0]
                avg_solve_len = (
                    sum(solved_lens) / len(solved_lens) if solved_lens else None
                )
                solved_excess = [x for x in excess_list if x >= 0]
                mean_excess = (
                    sum(solved_excess) / len(solved_excess) if solved_excess else None
                )

                logger.log(
                    event="cell",
                    depth=int(depth),
                    beam_width=int(beam_width),
                    n_per_depth=config.n_per_depth,
                    n_solved=n_solved,
                    solve_rate=solve_rate,
                    avg_solve_len=avg_solve_len,
                    mean_v_star_excess=mean_excess,
                    n_expansions=result.n_expansions,
                    elapsed_seconds=elapsed,
                )
                avg_str = "N/A" if avg_solve_len is None else f"{avg_solve_len:.2f}"
                excess_str = "N/A" if mean_excess is None else f"{mean_excess:.2f}"
                print(
                    f"  d={int(depth):>2} w={int(beam_width):>4}  "
                    f"solved {n_solved:>4}/{config.n_per_depth}  "
                    f"rate {solve_rate:.3f}  "
                    f"avg_len {avg_str}  "
                    f"excess {excess_str}  "
                    f"({elapsed:.1f}s, {result.n_expansions:,} exp)",
                    flush=True,
                )

                cells_payload.append(
                    {
                        "depth": int(depth),
                        "beam_width": int(beam_width),
                        "solve_lens": lens_list,
                        "v_star_excess": excess_list,
                    }
                )

        run_elapsed = time.perf_counter() - run_t0
        logger.log(
            event="run_end",
            elapsed_seconds=run_elapsed,
            n_cells=len(cells_payload),
        )

    histograms = {
        "config": config.to_dict(),
        "checkpoint": config.checkpoint_path,
        "cells": cells_payload,
    }
    hist_path = out_dir / "solve_histograms_beam.json"
    with hist_path.open("w") as f:
        json.dump(histograms, f, indent=2)

    print()
    print(f"done. {len(cells_payload)} cells in {run_elapsed:.1f}s wall.")
    try:
        hist_display = hist_path.relative_to(REPO_ROOT)
    except ValueError:
        hist_display = hist_path
    print(f"histograms: {hist_display}")


if __name__ == "__main__":
    main()
