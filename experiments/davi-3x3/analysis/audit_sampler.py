"""Audit the actual V* depth distribution of `generate_adi_batch` output (3x3).

Mirrors ``experiments/davi-2x2/analysis/audit_sampler.py``. The 2x2
audit's central finding — that random walks of length d typically land
on states with true V* < d — is expected to apply on 3x3 as well, but
**only the d ≤ K=6 portion can be measured directly** (the bounded V*
oracle from P1b covers depths 0..6 exhaustively; beyond K the audit can
only report walk-depth, not V*).

The script generates several full batches at the production config and
reports:
- Claimed-depth histogram (should match per-depth balance).
- True-V* depth histogram for states with V* ≤ K (the rest fall in a
  ``> K`` overflow bucket).
- 2D heatmap of (claimed_depth, true_depth_or_overflow) — how much each
  claimed bucket spreads across measurable true depths vs lands in the
  ``> K`` tail.

If the bounded V* table doesn't yet exist (pre-P1b), this script
short-circuits with a stub message rather than crashing.

Usage:
    uv run python experiments/davi-3x3/analysis/audit_sampler.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rubik.cube.spec import CUBE_3X3
from rubik.training.scrambles import generate_adi_batch

REPO_ROOT = Path(__file__).resolve().parents[3]
V_STAR_CACHE = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"
V_STAR_K = 6  # bounded oracle's K (plan §P1b)

BATCH_SIZE = 4096
MAX_DEPTH = 14  # walk depth — matches the eval set's depth bins (P1c)
N_BATCHES = 8  # 8 * 4096 = 32k states
SEED = 0


def main() -> None:
    if not V_STAR_CACHE.exists():
        print(
            f"bounded V* cache not found at {V_STAR_CACHE.relative_to(REPO_ROOT)} — "
            "build it in P1b before running this audit. Skipping."
        )
        return

    # Lazy import — module doesn't exist until P1b lands.
    from rubik.oracle.v_star_bounded_3x3 import (
        load_v_star,
        lookup_v_star_batch,
    )

    print(f"loading bounded V* cache from {V_STAR_CACHE.relative_to(REPO_ROOT)}...")
    v_star = load_v_star(V_STAR_CACHE)
    print(f"  {len(v_star):,} states; max depth = {V_STAR_K}")

    spec = CUBE_3X3
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    claimed_total: Counter[int] = Counter()
    true_total: Counter[int] = Counter()  # depths 0..K plus K+1 = "> K" bucket
    overflow_bucket = V_STAR_K + 1
    joint = np.zeros((MAX_DEPTH + 1, V_STAR_K + 2), dtype=np.int64)

    print(f"\ngenerating {N_BATCHES} batches of B={BATCH_SIZE} at K_max={MAX_DEPTH}...")
    for i in range(N_BATCHES):
        states, depths, _ = generate_adi_batch(
            spec,
            batch_size=BATCH_SIZE,
            max_depth=MAX_DEPTH,
            generator=generator,
        )
        # bounded lookup returns -1 (or sentinel) for states with V* > K;
        # remap the sentinel into the overflow bucket index.
        true_raw = lookup_v_star_batch(states, v_star, k=V_STAR_K)
        true = [t if 0 <= t <= V_STAR_K else overflow_bucket for t in true_raw]
        claimed_np = depths.numpy()
        claimed_total.update(claimed_np.tolist())
        true_total.update(true)
        for c, t in zip(claimed_np, true, strict=True):
            joint[int(c), int(t)] += 1
        print(f"  batch {i + 1}/{N_BATCHES}")

    n_total = sum(claimed_total.values())
    print(f"\ntotal states sampled: {n_total:,}")

    print("\n=== claimed-depth histogram ===")
    print(f"{'depth':>5}  {'count':>8}  {'pct':>6}")
    for d in range(0, MAX_DEPTH + 1):
        c = claimed_total.get(d, 0)
        pct = 100 * c / n_total
        print(f"{d:>5}  {c:>8}  {pct:>5.2f}%")

    print(f"\n=== true-V* depth histogram (V* ≤ {V_STAR_K}; overflow bucket = > K) ===")
    print(f"{'depth':>5}  {'count':>8}  {'pct':>6}")
    for d in range(0, V_STAR_K + 2):
        c = true_total.get(d, 0)
        pct = 100 * c / n_total
        label = "> K" if d == overflow_bucket else f"{d}"
        print(f"{label:>5}  {c:>8}  {pct:>5.2f}%")

    print(
        "\n=== joint (rows=claimed, cols=true V* depth, last col=> K) — "
        "% within claimed ==="
    )
    cols_label = "  ".join(
        [f"d{t:02d}".rjust(5) for t in range(0, V_STAR_K + 1)] + [">K".rjust(5)]
    )
    print(f"{'claimed':>8}  {cols_label}")
    for c in range(1, MAX_DEPTH + 1):
        row_total = joint[c].sum()
        if row_total == 0:
            continue
        cells = "  ".join(
            [f"{100 * joint[c, t] / row_total:>5.1f}" for t in range(0, V_STAR_K + 2)]
        )
        print(f"{c:>8}  {cells}")


if __name__ == "__main__":
    main()
