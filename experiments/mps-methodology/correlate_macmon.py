"""M4 — Correlate macmon NDJSON samples against the workload window.

Reads `runs/<ts>/macmon/samples.ndjson` + `runs/<ts>/macmon/window.json`,
filters the samples to the workload window, and prints ONE summary line.

Two modes:
    correlate_macmon.py
        Without --expected-busy: prints raw measured stats.
    correlate_macmon.py --expected-busy 0.78
        Compares the bench-layer-predicted GPU busy fraction against the
        macmon-measured one. Verdict AGREE if |delta| < 0.10, else DIVERGE.

Streaming policy: NDJSON is read line-by-line (no full-file load); output
is exactly one line. Long traces never enter the LLM context.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "experiments" / "mps-methodology" / "runs"

AGREE_DELTA = 0.10


def find_latest_run() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(RUNS_DIR.glob("*/macmon/samples.ndjson"))
    if not candidates:
        return None
    # Return the run dir (parent of `macmon/`).
    return candidates[-1].parent.parent


def parse_iso(ts: str) -> datetime:
    # Accept both ...Z and ...+00:00 forms.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        type=Path,
        default=None,
        help="runs/<ts> directory (auto-picks newest if omitted).",
    )
    ap.add_argument(
        "--expected-busy",
        type=float,
        default=None,
        help="Bench-predicted GPU busy fraction. If given, prints AGREE/DIVERGE.",
    )
    args = ap.parse_args()

    run_dir = args.run if args.run is not None else find_latest_run()
    if run_dir is None or not run_dir.exists():
        print("ERROR: no macmon run found (looked for runs/*/macmon/)", file=sys.stderr)
        return 1

    samples_path = run_dir / "macmon" / "samples.ndjson"
    window_path = run_dir / "macmon" / "window.json"
    if not samples_path.exists() or not window_path.exists():
        print(
            f"ERROR: missing {samples_path.name} or {window_path.name} in {run_dir}",
            file=sys.stderr,
        )
        return 1

    with window_path.open("r") as f:
        window = json.load(f)
    start = parse_iso(window["start"])
    end = parse_iso(window["end"])

    busy_sum = 0.0
    power_sum = 0.0
    in_window = 0
    with samples_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_iso(rec["timestamp"])
            if ts < start or ts > end:
                continue
            gpu_usage = rec.get("gpu_usage")
            if isinstance(gpu_usage, list) and len(gpu_usage) >= 2:
                busy_sum += float(gpu_usage[1])
            gpu_power = rec.get("gpu_power")
            if gpu_power is not None:
                power_sum += float(gpu_power)
            in_window += 1

    if in_window == 0:
        print(
            f"ERROR: no macmon samples fell inside window [{start}, {end}]",
            file=sys.stderr,
        )
        return 1

    busy_mean = busy_sum / in_window
    power_mean = power_sum / in_window
    window_seconds = (end - start).total_seconds()

    base = (
        f"gpu_busy={busy_mean:.3f} gpu_power={power_mean:.2f}W "
        f"samples={in_window} window={window_seconds:.1f}s"
    )

    if args.expected_busy is None:
        print(base)
        return 0

    delta = abs(busy_mean - args.expected_busy)
    verdict = "AGREE" if delta < AGREE_DELTA else "DIVERGE"
    print(f"{verdict}: {base} expected={args.expected_busy:.3f}")
    return 0 if verdict == "AGREE" else 1


if __name__ == "__main__":
    sys.exit(main())
