"""Build the bounded V* oracle cache for 3x3 at K=6.

Reverse-BFS from `CUBE_3X3.solved_state` up to depth 6 (~983,926 states),
then save to ``data/v_star_bounded_3x3_k6.npz``. One-shot script — re-run
on demand if the cache is missing or corrupt. Cache is gitignored
(matches the ``data/v_star_2x2.npz`` pattern).

Usage:
    uv run python scripts/build_v_star_bounded_3x3.py

Expected output: per-layer counts matching the well-known QTM table
(layer counts: 1, 12, 114, 1068, 10011, 93840, 878880; cumulative 983926
at K=6). Wall time ~30–120s on M4 Max.
"""

from __future__ import annotations

import time
from pathlib import Path

from rubik.cube.spec import CUBE_3X3
from rubik.oracle.v_star_bounded_3x3 import compute_v_star_bounded_3x3

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"
K = 6


def main() -> None:
    if CACHE_PATH.exists():
        print(f"cache already exists at {CACHE_PATH}; remove to rebuild")
        return
    print(f"building bounded V* oracle for 3x3, K={K} → {CACHE_PATH}")
    t0 = time.perf_counter()
    v_star = compute_v_star_bounded_3x3(
        CUBE_3X3,
        max_depth=K,
        cache_path=CACHE_PATH,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    size_mb = CACHE_PATH.stat().st_size / (1024 * 1024)
    print()
    print(f"done in {elapsed:.1f}s — {len(v_star):,} states, cache {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
