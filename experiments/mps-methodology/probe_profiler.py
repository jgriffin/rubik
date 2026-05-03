"""M4 — Profiler probe for apply_moves on MPS.

Captures a torch.profiler trace of apply_moves at a representative batch
size. On torch 2.11, ProfilerActivity.MPS is not available, so this
captures CPU dispatch only — the trace shows op order and any CPU sync
points (scalar extractions, copies, etc.), NOT GPU kernel times.

For wall-time measurements, use src/rubik/perf/bench.py instead. This
script is purely for finding hidden CPU round-trips.

Output: Chrome trace JSON at
  experiments/mps-methodology/runs/<timestamp>/profiler/trace.json
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from rubik.cube.env import apply_moves
from rubik.cube.spec import CUBE_2X2

# Hard-coded batch size — representative mid-batch where MPS dispatch overhead
# should be amortized but we're not yet pushing allocator pressure.
BATCH_SIZE = 8192

# Profiler schedule: 1 wait + 2 warmup + 5 active = 8 prof.step() iterations.
# Only the 5 active iterations land in the trace.
WAIT_STEPS = 1
WARMUP_STEPS = 2
ACTIVE_STEPS = 5
TOTAL_STEPS = WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS

# Bypass the MPS kernel-compile cache before the profiler context opens.
# The schedule's `warmup` discards trace data from those iterations, but
# kernel compilation still hits and inflates apparent CPU dispatch time
# inside the profiler. Doing a separate uninstrumented warmup keeps the
# active window clean.
PRE_PROFILER_WARMUP = 20

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    if not torch.backends.mps.is_available():
        print("ERROR: torch.backends.mps.is_available() is False", file=sys.stderr)
        return 1

    device = torch.device("mps")
    spec = CUBE_2X2

    # Build inputs once, reuse across all iterations. Steady-state cost
    # is what we're after, not per-call allocation overhead.
    states = spec.solved_state.to(device=device, dtype=torch.int8)
    states = states.unsqueeze(0).expand(BATCH_SIZE, -1).contiguous()
    g = torch.Generator(device="cpu").manual_seed(0)
    # CPU move_idxs — matches production reality. random_scrambles' inner
    # loop and the future beam-search expansion both build move tags on
    # CPU and let apply_moves migrate. The 2026-05-02 cleanup-loop block
    # aligned this with bench_apply_moves.py so the trace honestly shows
    # "no syncs in the realistic call pattern" after the bounds-check
    # relocation. If a future caller pre-migrates to MPS, it pays the
    # same sync cost as the pre-fix code (verifier won't catch that path
    # — by design, since it's not the production pattern).
    move_idxs = torch.randint(0, spec.n_moves, (BATCH_SIZE,), generator=g)

    # Pre-profiler warmup — beat the kernel-compile cache.
    for _ in range(PRE_PROFILER_WARMUP):
        apply_moves(states, move_idxs, spec)
    torch.mps.synchronize()

    # Set up output dir.
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = REPO_ROOT / "experiments" / "mps-methodology" / "runs" / ts / "profiler"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.json"

    def on_trace_ready(prof: profile) -> None:
        prof.export_chrome_trace(str(trace_path))

    try:
        with profile(
            activities=[ProfilerActivity.CPU],
            schedule=schedule(
                wait=WAIT_STEPS, warmup=WARMUP_STEPS, active=ACTIVE_STEPS
            ),
            on_trace_ready=on_trace_ready,
        ) as prof:
            for _ in range(TOTAL_STEPS):
                apply_moves(states, move_idxs, spec)
                torch.mps.synchronize()
                prof.step()
    except Exception as exc:  # noqa: BLE001 — surface any profiler error to stderr
        print(f"ERROR: profiler failed: {exc}", file=sys.stderr)
        return 2

    if not trace_path.exists():
        print(f"ERROR: trace not exported to {trace_path}", file=sys.stderr)
        return 3

    # Count events in the trace for the verifier's downstream check.
    with trace_path.open("r") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", [])

    # Print exactly two lines: trace path (relative to repo root) + event count.
    rel = trace_path.relative_to(REPO_ROOT)
    print(f"trace: {rel}")
    print(f"events: {len(events)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
