"""M4 — apply_moves workload driver for the macmon probe.

Internal helper invoked by `probe_macmon.sh`. Runs apply_moves repeatedly
on MPS for ~25 seconds, recording the workload window start/end + call
count to the window.json file passed in as the only positional argument.

Underscore prefix marks this as "internal helper, not a top-level entry
point" — there is no useful CLI invocation outside of probe_macmon.sh.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from rubik.cube.env import apply_moves
from rubik.cube.spec import CUBE_2X2

# Same B as probe_profiler.py — keeps the two probes comparable.
BATCH_SIZE = 8192

# Beat the kernel-compile cache before the recorded section opens.
WARMUP_CALLS = 50

# Approximate target wall time for the recorded section. The actual number
# of apply_moves calls depends on per-call latency on this machine.
TARGET_SECONDS = 25.0

# Sync every N calls so the GPU stays engaged (no unbounded queue) without
# stalling Python on every iteration.
SYNC_EVERY = 64


def main() -> int:
    if len(sys.argv) != 2:
        print(
            f"usage: {sys.argv[0]} <window.json path>",
            file=sys.stderr,
        )
        return 2

    window_path = Path(sys.argv[1])
    if not window_path.exists():
        print(f"window.json not found at {window_path}", file=sys.stderr)
        return 2

    if not torch.backends.mps.is_available():
        print("torch.backends.mps.is_available() is False", file=sys.stderr)
        return 2

    device = torch.device("mps")
    spec = CUBE_2X2

    # (B, 24) int8 states + (B,) int64 move idxs — same setup as the
    # profiler probe so layer-1 → layer-2 comparisons are apples-to-apples.
    states = spec.solved_state.to(device=device, dtype=torch.int8)
    states = states.unsqueeze(0).expand(BATCH_SIZE, -1).contiguous()
    g = torch.Generator(device="cpu").manual_seed(0)
    move_idxs = torch.randint(0, spec.n_moves, (BATCH_SIZE,), generator=g).to(device)

    # Warmup.
    for _ in range(WARMUP_CALLS):
        apply_moves(states, move_idxs, spec)
    torch.mps.synchronize()

    # Recorded section — bound by wall time, not iteration count.
    start_ts = (
        datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    t0 = time.perf_counter()
    n_calls = 0
    deadline = t0 + TARGET_SECONDS
    while time.perf_counter() < deadline:
        apply_moves(states, move_idxs, spec)
        n_calls += 1
        if n_calls % SYNC_EVERY == 0:
            torch.mps.synchronize()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    end_ts = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    # Merge into window.json (preserve any existing fields like `start`).
    with window_path.open("r") as f:
        window = json.load(f)
    window["start_ts"] = start_ts
    window["end_ts"] = end_ts
    window["start"] = start_ts
    window["end"] = end_ts
    window["workload_calls"] = n_calls
    window["batch_size"] = BATCH_SIZE
    window["elapsed_seconds"] = elapsed
    with window_path.open("w") as f:
        json.dump(window, f, indent=2, sort_keys=True)

    print(f"workload: {n_calls} apply_moves calls in {elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
