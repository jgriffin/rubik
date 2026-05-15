#!/usr/bin/env bash
# Stage net_final.{onnx,onnx.data} from the experiments run dir into
# web/public/models/ so vite (and the production build) can serve them
# at /models/net_final.onnx.
#
# Wired up via pnpm predev / prebuild so the dev server always has a
# fresh artifact. mtime-based — skips copy if dest is up-to-date so the
# 59 MB external-data blob isn't shoveled on every dev restart.
#
# Source is gitignored (.onnx + .onnx.data); so is the destination
# (web/public/models/) — neither ships in the repo.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)/experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/public/models"
mkdir -p "$DEST_DIR"

for f in net_final.onnx net_final.onnx.data; do
  if [[ -f "$SRC_DIR/$f" ]]; then
    if [[ ! -f "$DEST_DIR/$f" || "$SRC_DIR/$f" -nt "$DEST_DIR/$f" ]]; then
      cp "$SRC_DIR/$f" "$DEST_DIR/$f"
      echo "synced $f ($(du -h "$DEST_DIR/$f" | cut -f1))"
    fi
  else
    echo "WARN: $SRC_DIR/$f not found — run scripts/export_onnx_3x3.py first" >&2
  fi
done
