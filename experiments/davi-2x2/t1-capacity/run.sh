#!/usr/bin/env bash
# T1 Phase A driver: loop through configs/phaseA_*.yaml sequentially.
# Each cell writes runs/<UTC-ts>_<config-stem>/{config.yaml,metrics.jsonl,net_final.pt}.
#
# Usage:
#   bash experiments/davi-2x2/t1-capacity/run.sh [phase]
#
# phase: "A" (default) or "B" — selects configs/phase{A,B}_*.yaml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TIER_DIR="$REPO_ROOT/experiments/davi-2x2/t1-capacity"
PHASE="${1:-A}"

shopt -s nullglob
configs=("$TIER_DIR"/configs/phase${PHASE}_*.yaml)
if (( ${#configs[@]} == 0 )); then
    echo "No configs found at $TIER_DIR/configs/phase${PHASE}_*.yaml" >&2
    exit 1
fi

echo "Phase $PHASE: ${#configs[@]} cells"
for cfg in "${configs[@]}"; do
    name="$(basename "$cfg" .yaml)"
    echo "===== $name ====="
    uv run python "$TIER_DIR/supervised.py" --config "$cfg"
    echo
done

echo "Phase $PHASE complete."
