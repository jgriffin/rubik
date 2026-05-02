#!/usr/bin/env bash
# M4 — macmon probe for apply_moves on MPS.
#
# Launches `macmon pipe --samples 150 --interval 200` (NDJSON, 30 seconds at
# 5 Hz) into runs/<ts>/macmon/samples.ndjson, captures the timestamps of a
# ~25-second apply_moves workload window, then exits cleanly.
#
# Streaming-output policy (M4 lock-in): every long-running tool writes to a
# file. This script prints exactly two lines to stdout — paths to the
# samples and the workload window — for the correlator to pick up.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUN_DIR="${REPO_ROOT}/experiments/mps-methodology/runs/${TS}/macmon"
mkdir -p "${RUN_DIR}"

SAMPLES="${RUN_DIR}/samples.ndjson"
WINDOW="${RUN_DIR}/window.json"

# 1. Launch macmon in the background. 150 samples × 200ms = 30s wall.
macmon pipe --samples 150 --interval 200 > "${SAMPLES}" 2>/dev/null &
MACMON_PID=$!

# 2. Let macmon get its first pre-workload baseline sample.
sleep 1

# 3. Initialize window.json as empty; _macmon_workload.py owns it from here
#    and writes start/end timestamps with sub-second precision via Python's
#    datetime (BSD `date` on macOS doesn't support %N for fractional seconds).
echo '{}' > "${WINDOW}"

uv run python "${REPO_ROOT}/experiments/mps-methodology/_macmon_workload.py" "${WINDOW}"

# 4. Let macmon capture a post-workload baseline sample.
sleep 1

# 5. Wait for macmon to finish naturally (it exits after 150 samples).
wait "${MACMON_PID}"

# 6. Print the two canonical paths (relative to repo root).
echo "samples: experiments/mps-methodology/runs/${TS}/macmon/samples.ndjson"
echo "window: experiments/mps-methodology/runs/${TS}/macmon/window.json"
