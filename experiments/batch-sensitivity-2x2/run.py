"""M4 — Batch-sensitivity sweep for apply_moves and random_scrambles on MPS.

Reads experiments/batch-sensitivity-2x2/config.yaml, sweeps each
(op, batch_size) cell, captures per-trial seconds via src/rubik/perf
time_op, and writes a structured JSON file with the full grid plus
metadata (config snapshot, torch version, machine info, git SHA).

This produces sync-bracket per-call numbers — the time_op brackets each
call with mps.synchronize(). For the regime difference vs steady-state
pipelined throughput, see experiments/mps-methodology/results.md §5.

Output: experiments/batch-sensitivity-2x2/runs/<ts>/data.json

This script does NOT run the full sweep when invoked with --smoke; pass
--smoke to run a tiny grid (1 batch size x 1 op x 3 trials) for
end-to-end verification. The full sweep happens in commit 7.

Streaming-output policy (M4 lock-in): one line per (op, batch_size) cell
plus a final `data: <path>` line. No per-trial output.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from rubik.cube.env import apply_moves, random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.perf import bootstrap_ci, time_op

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "batch-sensitivity-2x2" / "config.yaml"
RUNS_ROOT = REPO_ROOT / "experiments" / "batch-sensitivity-2x2" / "runs"


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_apply_moves_closure(batch_size: int, device: torch.device) -> Any:
    spec = CUBE_2X2
    states = spec.solved_state.to(device=device, dtype=torch.int8)
    states = states.unsqueeze(0).expand(batch_size, -1).contiguous()
    g = torch.Generator(device="cpu").manual_seed(0)
    move_idxs = torch.randint(0, spec.n_moves, (batch_size,), generator=g).to(device)

    def fn() -> None:
        apply_moves(states, move_idxs, spec)

    return fn


def build_random_scrambles_closure(batch_size: int, depth: int, seed: int) -> Any:
    spec = CUBE_2X2
    # random_scrambles internally drives torch.multinomial / torch.randint on
    # CPU tensors; the produced states/move_seqs live on CPU. The whole
    # operation is what's being measured.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def fn() -> None:
        random_scrambles(spec, batch_size, depth, generator=generator)

    return fn


def run_cell(
    op: str,
    batch_size: int,
    *,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if op == "apply_moves":
        closure = build_apply_moves_closure(batch_size, device)
    elif op == "random_scrambles":
        closure = build_random_scrambles_closure(
            batch_size, config["random_scrambles_depth"], config["seed"]
        )
    else:
        raise ValueError(f"unknown op {op!r}")

    trials = time_op(
        closure,
        warmup=config["warmup"],
        trials=config["trials"],
        device=config["device"],
    )

    # Bootstrap CI on per-trial seconds.
    median_s, lo_s, hi_s = bootstrap_ci(trials, seed=config["seed"])

    # Throughput in items/sec. For apply_moves: states/sec (one move applied
    # to B states per call). For random_scrambles: scrambles/sec (the call
    # produces B scrambles of length `depth`).
    # CI on throughput is computed by inverting the seconds CI bounds:
    # high seconds -> low throughput, and vice versa.
    throughput_median = batch_size / median_s if median_s > 0 else 0.0
    throughput_lo = batch_size / hi_s if hi_s > 0 else 0.0
    throughput_hi = batch_size / lo_s if lo_s > 0 else 0.0

    return {
        "op": op,
        "batch_size": batch_size,
        "trials": trials,
        "median_seconds": median_s,
        "ci_lo_seconds": lo_s,
        "ci_hi_seconds": hi_s,
        "throughput_per_sec_median": throughput_median,
        "throughput_per_sec_lo": throughput_lo,
        "throughput_per_sec_hi": throughput_hi,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to sweep config (default: {DEFAULT_CONFIG.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny 1-cell grid (B=64, 3 trials) for end-to-end verification.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/<ts>).",
    )
    args = ap.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found at {args.config}", file=sys.stderr)
        return 1

    with args.config.open("r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if args.smoke:
        config["batch_sizes"] = [64]
        config["ops"] = ["apply_moves"]
        config["trials"] = 3
        config["warmup"] = 2

    if config["device"] == "mps" and not torch.backends.mps.is_available():
        print(
            "ERROR: device='mps' but torch.backends.mps.is_available() is False",
            file=sys.stderr,
        )
        return 1

    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = args.out_dir if args.out_dir is not None else RUNS_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    per_op_batch_sizes = config.get("per_op_batch_sizes", {}) or {}
    for op in config["ops"]:
        op_batch_sizes = per_op_batch_sizes.get(op, config["batch_sizes"])
        for batch_size in op_batch_sizes:
            cell = run_cell(op, batch_size, config=config, device=device)
            cells.append(cell)
            print(
                f"cell op={cell['op']} B={cell['batch_size']} "
                f"median={cell['median_seconds']:.3e}s "
                f"ci=[{cell['ci_lo_seconds']:.3e}, {cell['ci_hi_seconds']:.3e}] "
                f"throughput={cell['throughput_per_sec_median']:.3e}/s",
                flush=True,
            )

    data = {
        "schema_version": 1,
        "machine": {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        },
        "git_sha": git_sha(),
        "config": config,
        "ran_at": datetime.now(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "cells": cells,
    }

    data_path = out_dir / "data.json"
    with data_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"data: {data_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
