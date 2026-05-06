"""Tier 0 — DAVI step time calibration on 3x3.

Mirrors ``experiments/davi-2x2/calibrate_step_time.py`` cell-for-cell,
swapping ``CUBE_2X2`` → ``CUBE_3X3``. **The calibration grid is
intentionally empty in this scaffold — it is earned in phase 2a, not
copied from the 2x2 grid** (CLAUDE.md "Earn every hyperparameter — do
not borrow"). Invoking this script before P2a fills in the grid will
raise ``NotImplementedError``.

For each ``(batch_size, body_widths, n_residual_blocks)`` cell in the
grid, this script:

1. Constructs a fresh ``ValueNet`` + target net + Adam optimizer.
2. Runs ``warmup`` untimed DAVI steps (cache + JIT + driver warmup).
3. Runs ``trials`` timed DAVI steps, each bracketed by
   ``torch.mps.synchronize()`` so wall-time captures GPU work, not just
   dispatch (the M4 ``time_op`` primitive).
4. Logs one JSONL record per cell: cell coords, parameter count, median
   ms/step, 95% bootstrap CI, full per-trial samples (raw seconds).

Critically, the timed inner loop **includes batch generation + device
transfer**, not just ``davi_step`` — that's what real training pays per
step. Splitting them is for follow-up profiling, not the calibration.

Output goes to
``experiments/davi-3x3/runs/<UTC-ts>_calibration/calibration.jsonl`` plus
a one-line stdout digest per cell.
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.optim import Adam

from rubik.cube.spec import CUBE_3X3, CubeSpec
from rubik.model.network import ValueNet
from rubik.perf.bench import bootstrap_ci, time_op
from rubik.training.davi import davi_step, sync_target
from rubik.training.metric_logger import MetricLogger
from rubik.training.scrambles import generate_adi_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "experiments" / "davi-3x3" / "runs"

# Calibration grid — TBD, earned in phase 2a (see plan Q-P2a). Do not
# pre-fill from the 2x2 grid; that's the cheat the methodology exists to
# avoid. Author the grid here when P2a opens, then re-run.
GRID_BATCH_SIZES: tuple[int, ...] = ()
GRID_BODY_WIDTHS: tuple[tuple[int, int], ...] = ()
GRID_N_RESIDUAL_BLOCKS: tuple[int, ...] = ()

# DAVI is *not* sensitive to scramble depth at the per-step-time level —
# `apply_all_moves` is fixed-cost; depth only changes which scrambled
# state you start from. Hold it at a TBD cycle-level value (the 2x2
# calibration used 14 = QTM diameter; 3x3 QTM diameter is 26 but step
# time doesn't depend on depth). P2a fills this.
SCRAMBLE_DEPTH: int | None = None
LEARNING_RATE: float = 1e-4  # any nonzero value; we measure step time, not convergence
WARMUP_STEPS: int = 5
TIMED_STEPS: int = 20


@dataclass(frozen=True)
class Cell:
    batch_size: int
    body_widths: tuple[int, int]
    n_residual_blocks: int


def _make_step_fn(
    spec: CubeSpec,
    cell: Cell,
    device: torch.device,
    generator: torch.Generator,
):
    """Build a closed-over zero-arg step function for `time_op` to bracket.

    Each invocation: generate one ADI batch on CPU → move to device →
    one DAVI step (forward + Bellman target + backward + optim). This
    matches what `run.py`'s training loop pays per step.
    """
    if SCRAMBLE_DEPTH is None:
        raise NotImplementedError(
            "SCRAMBLE_DEPTH not set — fill in P2a before invoking this script"
        )
    net = ValueNet(
        spec,
        body_widths=cell.body_widths,
        n_residual_blocks=cell.n_residual_blocks,
    ).to(device)
    target_net = ValueNet(
        spec,
        body_widths=cell.body_widths,
        n_residual_blocks=cell.n_residual_blocks,
    ).to(device)
    sync_target(net, target_net)
    optimizer = Adam(net.parameters(), lr=LEARNING_RATE)

    def step() -> None:
        states, _, _ = generate_adi_batch(
            spec,
            batch_size=cell.batch_size,
            max_depth=SCRAMBLE_DEPTH,
            generator=generator,
        )
        states = states.to(device)
        davi_step(net, target_net, optimizer, states, spec)

    n_params = sum(p.numel() for p in net.parameters())
    return step, n_params


def _resolve_out_dir(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{ts}_calibration"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (GRID_BATCH_SIZES and GRID_BODY_WIDTHS and GRID_N_RESIDUAL_BLOCKS):
        raise NotImplementedError(
            "Calibration grid not authored — earn it in P2a (plan Q-P2a). "
            "Do not copy from the 2x2 grid."
        )

    out_dir = _resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "calibration.jsonl"

    device = torch.device(args.device)
    spec = CUBE_3X3
    torch.manual_seed(args.seed)

    cells = [
        Cell(bs, bw, nrb)
        for bs, bw, nrb in itertools.product(
            GRID_BATCH_SIZES, GRID_BODY_WIDTHS, GRID_N_RESIDUAL_BLOCKS
        )
    ]
    print(f"calibrating {len(cells)} cells on device={device}")
    print(f"output: {jsonl_path.relative_to(REPO_ROOT)}")
    print()
    print(
        f"{'#':>3} {'batch':>6} {'body':>14} {'blocks':>7} "
        f"{'params':>9} {'median (ms)':>13} {'95% CI (ms)':>20} {'wall (s)':>9}"
    )
    print("-" * 100)

    grid_t0 = time.perf_counter()
    with MetricLogger(jsonl_path) as logger:
        logger.log(
            event="grid_start",
            n_cells=len(cells),
            warmup=WARMUP_STEPS,
            trials=TIMED_STEPS,
            scramble_depth=SCRAMBLE_DEPTH,
            device=str(device),
            seed=args.seed,
        )

        for i, cell in enumerate(cells):
            # Per-cell generator so identical re-runs reproduce per-cell.
            generator = torch.Generator(device="cpu").manual_seed(args.seed + i)

            cell_t0 = time.perf_counter()
            step_fn, n_params = _make_step_fn(spec, cell, device, generator)
            timings = time_op(
                step_fn,
                warmup=WARMUP_STEPS,
                trials=TIMED_STEPS,
                device=args.device,
            )
            wall_seconds = time.perf_counter() - cell_t0
            median_s, lo_s, hi_s = bootstrap_ci(timings, seed=args.seed + i)
            median_ms = median_s * 1000.0
            lo_ms = lo_s * 1000.0
            hi_ms = hi_s * 1000.0

            logger.log(
                event="cell",
                cell_idx=i,
                batch_size=cell.batch_size,
                body_widths=list(cell.body_widths),
                n_residual_blocks=cell.n_residual_blocks,
                n_params=n_params,
                step_seconds_samples=timings,
                step_ms_median=median_ms,
                step_ms_ci_low=lo_ms,
                step_ms_ci_high=hi_ms,
                wall_seconds=wall_seconds,
            )
            body_str = f"{cell.body_widths[0]}x{cell.body_widths[1]}"
            print(
                f"{i:>3} {cell.batch_size:>6} {body_str:>14} "
                f"{cell.n_residual_blocks:>7} {n_params / 1e6:>8.1f}M "
                f"{median_ms:>13.2f} {lo_ms:>9.2f}–{hi_ms:<9.2f} {wall_seconds:>9.1f}",
                flush=True,
            )

        logger.log(event="grid_end", wall_seconds=time.perf_counter() - grid_t0)

    print()
    print(f"done. {len(cells)} cells in {time.perf_counter() - grid_t0:.1f} s")
    print(f"jsonl: {jsonl_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
