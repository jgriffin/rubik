#!/usr/bin/env bash
# Cycle 4: two sequential 30k-step DAVI runs, both warm-started from
# sync500_kmax20-30k/net_final.pt.
#
#  Cell 1 — kmax28_warm: K_max=28 fixed for all 30k steps
#  Cell 2 — kmax28_curriculum: K_max ramps 14→28 over first 15k, holds at 28
#
# Cell 2 only launches if Cell 1 succeeds. Designed to be nohup'd:
#   nohup bash experiments/davi-2x2/run_cycle4.sh > /tmp/cycle4-watcher.log 2>&1 &
#   disown
#
# Usage (foreground for testing): bash experiments/davi-2x2/run_cycle4.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESUME="experiments/davi-2x2/runs/sync500_kmax20-30k/net_final.pt"

if [[ ! -f "$RESUME" ]]; then
  echo "ERROR: resume checkpoint not found: $RESUME" >&2
  exit 1
fi

run_cell () {
  local CONFIG="$1"
  local OUT_DIR="$2"
  mkdir -p "$OUT_DIR"
  echo "=== launching cell: $CONFIG -> $OUT_DIR ==="
  echo "  resume: $RESUME"
  echo "  start:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uv run python experiments/davi-2x2/run.py \
    --config "$CONFIG" \
    --resume "$RESUME" \
    --out-dir "$OUT_DIR" \
    > "$OUT_DIR/stdout.log" 2>&1
  echo "=== cell done: $OUT_DIR ==="
  echo "  end:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_cell \
  "experiments/davi-2x2/configs/kmax28_warm.yaml" \
  "experiments/davi-2x2/runs/kmax28_warm-30k"

run_cell \
  "experiments/davi-2x2/configs/kmax28_curriculum.yaml" \
  "experiments/davi-2x2/runs/kmax28_curriculum-30k"

echo "=== cycle 4 sweep complete ==="
