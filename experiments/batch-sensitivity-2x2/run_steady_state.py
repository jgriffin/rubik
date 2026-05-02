"""M4 — Steady-state-vs-bench correction-factor spot-check.

Mirrors experiments/mps-methodology/_macmon_workload.py's tight-loop
pattern (sync every 64 calls, wall-time bound) but is a top-level entry
point: produces the bench-bracket vs steady-state correction factor at
ONE batch size.

The bench-bracket per-call time (sync before / sync after) is the
"primary truth" of M4 measurement, but it bakes in ~0.22 ms of pure
sync stall per call at B=8192 on this M4 Max. That's a regime-dependent
underestimate of throughput for any pipelined workload (training,
beam search). This script measures the gap so analyze.py can embed the
correction factor next to the swept numbers.

Output: experiments/batch-sensitivity-2x2/runs/<ts>/steady_state.json
plus one stdout line:
    correction: bench_per_call=<X.XXXe-3>s steady_per_call=<X.XXXe-3>s factor=<X.XXX>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from rubik.cube.env import apply_moves
from rubik.cube.spec import CUBE_2X2
from rubik.perf import time_op

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_ROOT = REPO_ROOT / "experiments" / "batch-sensitivity-2x2" / "runs"

# Sync cadence matches _macmon_workload.py — keeps the GPU engaged without
# stalling Python on every iteration.
SYNC_EVERY = 64

# Beat the kernel-compile cache before the recorded section opens.
WARMUP_CALLS = 50

# time_op parameters for the bench-bracket comparison. Match config.yaml's
# warmup/trials so the correction factor is honest about what the sweep
# numbers represent.
BENCH_WARMUP = 5
BENCH_TRIALS = 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size for the spot-check (default: 8192).",
    )
    ap.add_argument(
        "--duration-seconds",
        type=float,
        default=10.0,
        help="Wall-time budget for the steady-state loop (default: 10.0).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: runs/<ts>/steady_state.json).",
    )
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        print(
            "ERROR: torch.backends.mps.is_available() is False",
            file=sys.stderr,
        )
        return 1

    device = torch.device("mps")
    spec = CUBE_2X2

    # (B, 24) int8 states + (B,) int64 move idxs — same setup as the macmon
    # workload so the steady-state regime is apples-to-apples with that
    # methodology piece.
    states = spec.solved_state.to(device=device, dtype=torch.int8)
    states = states.unsqueeze(0).expand(args.batch_size, -1).contiguous()
    g = torch.Generator(device="cpu").manual_seed(0)
    move_idxs = torch.randint(0, spec.n_moves, (args.batch_size,), generator=g).to(
        device
    )

    def closure() -> None:
        apply_moves(states, move_idxs, spec)

    # 1. Warmup before the steady-state loop.
    for _ in range(WARMUP_CALLS):
        closure()
    torch.mps.synchronize()

    # 2. Steady-state pipelined loop — sync every SYNC_EVERY calls, bound by
    # wall time.
    t0 = time.perf_counter()
    n_calls = 0
    deadline = t0 + args.duration_seconds
    while time.perf_counter() < deadline:
        closure()
        n_calls += 1
        if n_calls % SYNC_EVERY == 0:
            torch.mps.synchronize()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    steady_state_per_call = elapsed / n_calls if n_calls > 0 else float("nan")

    # 3. Bench-bracket per-call at the same batch size — uses time_op so the
    # number is exactly what the sweep would record for this cell.
    bench_trials = time_op(
        closure,
        warmup=BENCH_WARMUP,
        trials=BENCH_TRIALS,
        device="mps",
    )
    bench_per_call = sum(bench_trials) / len(bench_trials)

    correction_factor = (
        steady_state_per_call / bench_per_call if bench_per_call > 0 else float("nan")
    )

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = (
        args.out if args.out is not None else RUNS_ROOT / ts / "steady_state.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "batch_size": args.batch_size,
        "duration_s": args.duration_seconds,
        "n_calls": n_calls,
        "steady_state_per_call_s": steady_state_per_call,
        "bench_bracket_per_call_s": bench_per_call,
        "correction_factor": correction_factor,
        "ran_at": datetime.now(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
    }
    with out_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(
        f"correction: bench_per_call={bench_per_call:.3e}s "
        f"steady_per_call={steady_state_per_call:.3e}s "
        f"factor={correction_factor:.3f}"
    )
    print(f"data: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
