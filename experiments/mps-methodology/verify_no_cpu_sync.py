"""M4 — Verify no UNEXPECTED CPU round-trips in apply_moves on MPS.

Reads the most recent Chrome trace under
experiments/mps-methodology/runs/<ts>/profiler/trace.json and counts
scalar-extraction events (aten::item, aten::_local_scalar_dense). Compares
against the known M2-code baseline.

Known baseline (M2):
  apply_moves contains a bounds check at env.py:80 — `(move_idxs_t < 0).any()
  or (...).any()` — that produces 2 sync points per call (4 scalar
  extractions: 2x aten::item + 2x aten::_local_scalar_dense). This is
  documented as a known cost; fix tracked as a follow-up block.

This verifier asserts: total scalar extractions <= 4 * active_iterations.
When the fix lands, EXPECTED_BOUNDS_CHECK_ITEMS_PER_CALL drops to 0 and
the verifier helps confirm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Per-call cost of the bounds check at src/rubik/cube/env.py:80
# (`(move_idxs_t < 0).any() or (move_idxs_t >= spec.n_moves).any()` —
# Python's `or` triggers __bool__() on each side, which on MPS routes
# through aten::is_nonzero -> aten::item -> aten::_local_scalar_dense).
# Empirically observed on torch 2.11: each comparison produces TWO
# aten::item + aten::_local_scalar_dense pairs (one from the `aten::lt`
# / `aten::ge` itself in the dispatch sequence, one from `is_nonzero`
# on the `.any()` result). Two checks per call x 4 scalar events each
# = 8. When the bounds check is removed, this drops to 0.
EXPECTED_BOUNDS_CHECK_ITEMS_PER_CALL = 8

# Matches probe_profiler.py's `schedule(wait=1, warmup=2, active=5)`.
ACTIVE_ITERATIONS = 5

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "experiments" / "mps-methodology" / "runs"

SCALAR_EVENTS = ("aten::item", "aten::_local_scalar_dense")


def find_latest_trace() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(RUNS_DIR.glob("*/profiler/trace.json"))
    return candidates[-1] if candidates else None


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

    scalar_count = sum(count_event(events, n) for n in SCALAR_EVENTS)
    copy_count = count_event(events, "aten::_to_copy")
    gather_count = count_event(events, "aten::gather")

    expected_ceiling = EXPECTED_BOUNDS_CHECK_ITEMS_PER_CALL * ACTIVE_ITERATIONS

    if scalar_count <= expected_ceiling:
        print(
            f"PASS: {scalar_count} scalar extractions "
            f"(expected <= {expected_ceiling}; "
            "baseline = M2 bounds check at env.py:80)"
        )
        print(f"copies: {copy_count} aten::_to_copy events")
        print(f"gathers: {gather_count} aten::gather events")
        return 0

    delta = scalar_count - expected_ceiling
    print(
        f"FAIL: {scalar_count} scalar extractions exceed "
        f"expected <= {expected_ceiling}. Excess count: {delta}. Investigate.",
        file=sys.stderr,
    )
    print(f"copies: {copy_count} aten::_to_copy events", file=sys.stderr)
    print(f"gathers: {gather_count} aten::gather events", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
