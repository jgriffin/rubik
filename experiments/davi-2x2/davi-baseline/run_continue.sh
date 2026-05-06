#!/usr/bin/env bash
# Warm-start continuation of baseline-30k with target_sync_interval=100.
# Captures a 60s macmon GPU-utilization probe at ~step 5k (steady-state) so
# we can answer the wider-batch question after the run.
#
# Usage: bash experiments/davi-2x2/davi-baseline/run_continue.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="experiments/davi-2x2/davi-baseline/configs/baseline_continue_sync100.yaml"
RESUME="experiments/davi-2x2/davi-baseline/runs/baseline-30k/net_final.pt"
OUT_DIR="experiments/davi-2x2/davi-baseline/runs/baseline-continue-sync100-30k"
MACMON_DIR="${OUT_DIR}/macmon"

if [[ ! -f "$RESUME" ]]; then
  echo "ERROR: resume checkpoint not found: $RESUME" >&2
  echo "       The legacy baseline-30k/net_final.pt is gitignored — locally" >&2
  echo "       preserved. If missing, the prior run must be re-completed." >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$MACMON_DIR"

echo "=== launching DAVI continuation ==="
echo "config:  $CONFIG"
echo "resume:  $RESUME"
echo "out:     $OUT_DIR"
echo "macmon:  $MACMON_DIR/samples.ndjson  (60s probe @ ~step 5k)"
echo

uv run python experiments/davi-2x2/run.py \
  --config "$CONFIG" \
  --resume "$RESUME" \
  --out-dir "$OUT_DIR" \
  > "$OUT_DIR/stdout.log" 2>&1 &
TRAIN_PID=$!

echo "training PID: $TRAIN_PID"
echo "stdout:       $OUT_DIR/stdout.log"
echo

# Wait for training to reach step 5000 before firing macmon. Polls the
# metrics.jsonl every 30 s; bails out if the training PID dies first.
TARGET_STEP=5000
MACMON_FIRED=0
echo "waiting for step >= $TARGET_STEP before firing macmon probe..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
  METRICS="$OUT_DIR/metrics.jsonl"
  if [[ -f "$METRICS" ]]; then
    LAST_STEP=$(grep -E '"event":\s*"step"' "$METRICS" 2>/dev/null \
      | tail -1 \
      | python -c "import json,sys; line=sys.stdin.read().strip(); print(json.loads(line)['step'] if line else 0)" \
      2>/dev/null || echo 0)
    LAST_STEP="${LAST_STEP:-0}"
    echo "  poll: last_step=$LAST_STEP"
    if [[ "$LAST_STEP" -ge "$TARGET_STEP" ]]; then
      echo "  reached step $LAST_STEP — firing macmon probe"
      MACMON_FIRED=1
      break
    fi
  fi
done

if [[ "$MACMON_FIRED" == "1" ]]; then
  # 60s @ 5 Hz = 300 samples, plenty of statistical power.
  echo
  echo "=== macmon probe (60s, 5 Hz) ==="
  macmon pipe --samples 300 --interval 200 \
    > "$MACMON_DIR/samples.ndjson" 2>/dev/null

  python - <<'PY' "$MACMON_DIR/samples.ndjson" "$MACMON_DIR/digest.txt"
import json, sys, statistics
ndjson_path, digest_path = sys.argv[1], sys.argv[2]
busy, power, gpu_temp = [], [], []
with open(ndjson_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        busy.append(rec["gpu_usage"][1])
        power.append(rec["gpu_power"])
        gpu_temp.append(rec["temp"]["gpu_temp_avg"])
n = len(busy)
def stats(xs):
    return statistics.mean(xs), statistics.median(xs), min(xs), max(xs)
b_mean, b_med, b_min, b_max = stats(busy)
p_mean, p_med, p_min, p_max = stats(power)
t_mean, _, t_min, t_max = stats(gpu_temp)
digest = (
    f"samples={n}  window={n*0.2:.1f}s\n"
    f"gpu_busy   mean={b_mean:.3f}  median={b_med:.3f}  min={b_min:.3f}  max={b_max:.3f}\n"
    f"gpu_power  mean={p_mean:.2f}W median={p_med:.2f}W min={p_min:.2f}W max={p_max:.2f}W\n"
    f"gpu_temp   mean={t_mean:.1f}C min={t_min:.1f}C max={t_max:.1f}C\n"
)
with open(digest_path, "w") as f:
    f.write(digest)
print(digest, end="")
PY
  echo
else
  echo "training process exited before reaching step $TARGET_STEP — skipping macmon"
fi

echo
echo "=== waiting for training to finish (PID $TRAIN_PID) ==="
wait "$TRAIN_PID"
TRAIN_EXIT=$?
echo "training exit: $TRAIN_EXIT"
echo "metrics:       $OUT_DIR/metrics.jsonl"
echo "stdout:        $OUT_DIR/stdout.log"
exit "$TRAIN_EXIT"
