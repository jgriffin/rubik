"""M4 — Verify no UNEXPECTED GPU sync stalls in apply_moves on MPS.

Reads the most recent Chrome trace under
experiments/mps-methodology/runs/<ts>/profiler/trace.json and counts
scalar-extraction events that are **slow enough to be GPU sync stalls**
rather than CPU-side dispatch.

**Why duration-based, not name-based.** The 2026-05-02 cleanup-loop block
revealed a verifier limitation: `aten::item` and friends appear in the
trace whenever Python `__bool__` is invoked on a 0-d tensor — even if
that tensor is CPU-resident, in which case the call costs sub-μs and
does NOT stall the GPU. A pure name-counting verifier flagged these as
syncs. Duration tells the story instead:

  - CPU `aten::is_nonzero`: ~0.25-1.2 μs (memory read)
  - MPS `aten::is_nonzero` waiting on a queued .any(): 100-400 μs
    (forces dispatch queue to drain — the actual sync stall)

Three orders of magnitude separation. Threshold at 50 μs catches every
real GPU sync and ignores every CPU dispatch event.

History:
  - M2 (initial code): bounds check at env.py ran on MPS post-migration.
    8 scalar extractions per call (4 per side x 2 sides), each taking
    150-325 μs. Baseline = 8 events/call.
  - 2026-05-02 cleanup-loop block: bounds check relocated to before the
    `.to(states.device)` migration. With probe_profiler.py's CPU
    move_idxs (matching production), the check now runs on CPU. Same
    event names, but durations drop to sub-μs. Verifier reframed to
    count `dur > SYNC_DURATION_THRESHOLD_US` events; baseline = 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# A GPU sync stall is at least an order of magnitude longer than CPU
# dispatch. CPU `aten::item` runs sub-μs; MPS sync stalls take 100s of μs
# to drain the queue. 50 μs is the safe middle threshold.
SYNC_DURATION_THRESHOLD_US = 50.0

# After the 2026-05-02 cleanup-loop block, the bounds check at env.py
# runs on CPU when callers pass a CPU `move_idxs` (production pattern:
# random_scrambles inner loop, beam-search children expansion). CPU
# evaluation emits trace events but no GPU sync. Expect zero events
# exceeding the duration threshold.
EXPECTED_GPU_SYNCS_PER_CALL = 0

# Matches probe_profiler.py's `schedule(wait=1, warmup=2, active=5)`.
ACTIVE_ITERATIONS = 5

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "experiments" / "mps-methodology" / "runs"

# Events that, when slow, indicate a host-blocking GPU sync.
SYNC_EVENT_NAMES = (
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::is_nonzero",
)


def find_latest_trace() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(RUNS_DIR.glob("*/profiler/trace.json"))
    return candidates[-1] if candidates else None


def count_slow_syncs(events: list[dict]) -> tuple[int, list[float]]:
    """Return (count, durations) for sync-event names whose dur exceeds threshold."""
    slow = [
        e
        for e in events
        if e.get("name") in SYNC_EVENT_NAMES
        and float(e.get("dur", 0)) > SYNC_DURATION_THRESHOLD_US
    ]
    return len(slow), [float(e.get("dur", 0)) for e in slow]


def count_event(events: list[dict], name: str) -> int:
    return sum(1 for e in events if e.get("name") == name)


def main() -> int:
    trace_path = find_latest_trace()
    if trace_path is None:
        print(
            f"FAIL: no trace found under {RUNS_DIR.relative_to(REPO_ROOT)}/"
            "*/profiler/trace.json — run probe_profiler.py first",
            file=sys.stderr,
        )
        return 2

    with trace_path.open("r") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", [])

    sync_count, sync_durs = count_slow_syncs(events)
    copy_count = count_event(events, "aten::_to_copy")
    gather_count = count_event(events, "aten::gather")

    # Total scalar-extraction event count regardless of duration — kept as a
    # context line so reviewers can see the CPU-side dispatch volume too.
    raw_event_count = sum(count_event(events, n) for n in SYNC_EVENT_NAMES)

    expected_ceiling = EXPECTED_GPU_SYNCS_PER_CALL * ACTIVE_ITERATIONS

    if sync_count <= expected_ceiling:
        print(
            f"PASS: {sync_count} GPU sync stalls "
            f"(events with dur > {SYNC_DURATION_THRESHOLD_US:.0f} μs; "
            f"expected <= {expected_ceiling})"
        )
        print(
            f"context: {raw_event_count} total sync-name events (most CPU-side, sub-μs)"
        )
        print(f"copies: {copy_count} aten::_to_copy events")
        print(f"gathers: {gather_count} aten::gather events")
        return 0

    delta = sync_count - expected_ceiling
    print(
        f"FAIL: {sync_count} GPU sync stalls exceed "
        f"expected <= {expected_ceiling}. Excess: {delta}.",
        file=sys.stderr,
    )
    print(f"slow-event durations (μs): {sync_durs}", file=sys.stderr)
    print(f"copies: {copy_count} aten::_to_copy events", file=sys.stderr)
    print(f"gathers: {gather_count} aten::gather events", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
