#!/usr/bin/env bash
# Stage onnxruntime-web's runtime assets (.wasm + .mjs workers) from
# node_modules to web/public/ort/ so the OnnxSolver can configure
# `ort.env.wasm.wasmPaths = "/ort/"` and load them from a stable URL
# instead of guessing relative paths inside Vite's module graph.
#
# Wired into models:sync (which is the predev/prebuild hook), so the
# dev server always has fresh runtime assets. The destination is
# gitignored — these are downstream of `pnpm install`, not source.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)/node_modules/onnxruntime-web/dist"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/public/ort"
mkdir -p "$DEST_DIR"

# Glob is small (~30 files); brute-force copy is simpler than mtime
# tracking and the .wasm payload is only ~50 MB total.
cp "$SRC_DIR"/*.wasm "$DEST_DIR/" 2>/dev/null || true
cp "$SRC_DIR"/*.mjs "$DEST_DIR/" 2>/dev/null || true
echo "synced ort-web runtime assets to $DEST_DIR"
ls "$DEST_DIR" | head -20
