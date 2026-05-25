#!/usr/bin/env bash
# Make web/public/models/net_final.{onnx,onnx.data} available for vite to
# serve at /models/. The weights come from one of three places, in order:
#
#   1. Local experiments run dir (dev) — copied if newer (mtime-based, so
#      the 59 MB blob isn't shoveled on every dev restart).
#   2. A real committed copy already on disk (dev after `git lfs pull`, or
#      a prior sync) — used as-is.
#   3. A Git LFS *pointer* on disk with no local source (Vercel's cloud
#      build): Vercel checks out LFS pointers but can't fetch the objects
#      themselves, so we pull the real bytes from GitHub's public media CDN
#      for the commit being built.
#
# Wired via pnpm predev / prebuild. The experiments source is gitignored;
# the two model files are committed via Git LFS (see .gitattributes).
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)/experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/public/models"
REPO_SLUG="jgriffin/rubik"
REF="${VERCEL_GIT_COMMIT_SHA:-main}"
MEDIA_BASE="https://media.githubusercontent.com/media/${REPO_SLUG}/${REF}/web/public/models"
mkdir -p "$DEST_DIR"

# True when the file exists and holds real content, not a Git LFS pointer stub.
is_real() { [[ -f "$1" ]] && ! head -c 64 "$1" | grep -q "git-lfs.github.com"; }

for f in net_final.onnx net_final.onnx.data; do
  dest="$DEST_DIR/$f"
  if [[ -f "$SRC_DIR/$f" && ( ! -f "$dest" || "$SRC_DIR/$f" -nt "$dest" ) ]]; then
    cp "$SRC_DIR/$f" "$dest"
    echo "synced $f from experiments ($(du -h "$dest" | cut -f1))"
  elif is_real "$dest"; then
    echo "using committed $f ($(du -h "$dest" | cut -f1))"
  else
    echo "fetching $f from GitHub media CDN ($REF) — LFS pointer on disk..."
    curl -fsSL "$MEDIA_BASE/$f" -o "$dest"
    echo "fetched $f ($(du -h "$dest" | cut -f1))"
  fi
done
