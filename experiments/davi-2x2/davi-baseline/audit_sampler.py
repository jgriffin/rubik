"""Audit the actual V* depth distribution of `generate_adi_batch` output.

The sampler claims per-depth balance: B=4096, max_depth=K, each claimed
depth d in [1, K] gets ~B/K states. But "claimed depth = d" means
"random walk of length d with same-face pruning" — the true V* depth
of those states may differ from d, because random walks can cycle
back to shallow states (e.g. R U R' is a length-3 walk that ends at
true depth 3, but more complex sequences may revisit shallower V*
shells).

This script generates several full batches at the production config
and reports:
- Claimed-depth histogram (should match per-depth balance)
- True-V* depth histogram (the actual distribution the model sees)
- 2D heatmap of (claimed_depth, true_depth) — how much each claimed
  bucket spreads across true depths.

Usage:
    uv run python experiments/davi-2x2/davi-baseline/audit_sampler.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rubik.cube.spec import CUBE_2X2
from rubik.oracle.v_star_2x2 import load_v_star, lookup_v_star_batch
from rubik.training.scrambles import generate_adi_batch

REPO_ROOT = Path(__file__).resolve().parents[3]
V_STAR_CACHE = REPO_ROOT / "data" / "v_star_2x2.npz"

BATCH_SIZE = 4096
MAX_DEPTH = 18
N_BATCHES = 8  # 8 * 4096 = 32k states — tight statistics
SEED = 0


def main() -> None:
    print(f"loading V* cache from {V_STAR_CACHE.relative_to(REPO_ROOT)}...")
    v_star = load_v_star(V_STAR_CACHE)
    print(f"  {len(v_star):,} states; max depth = {max(v_star.values())}")

    spec = CUBE_2X2
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    claimed_total: Counter[int] = Counter()
    true_total: Counter[int] = Counter()
    joint = np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 1), dtype=np.int64)

    print(
        f"\ngenerating {N_BATCHES} batches of B={BATCH_SIZE} at K_max={MAX_DEPTH}..."
    )
    for i in range(N_BATCHES):
        states, depths, _ = generate_adi_batch(
            spec,
            batch_size=BATCH_SIZE,
            max_depth=MAX_DEPTH,
            generator=generator,
        )
        true = lookup_v_star_batch(states, v_star)
        claimed_np = depths.numpy()
        claimed_total.update(claimed_np.tolist())
        true_total.update(true.tolist())
        for c, t in zip(claimed_np, true, strict=True):
            joint[int(c), int(t)] += 1
        print(f"  batch {i + 1}/{N_BATCHES}")

    n_total = sum(claimed_total.values())
    print(f"\ntotal states sampled: {n_total:,}")
    print(f"max true depth observed: {max(true_total)}")

    print("\n=== claimed-depth histogram ===")
    print(f"{'depth':>5}  {'count':>8}  {'pct':>6}")
    for d in range(0, MAX_DEPTH + 1):
        c = claimed_total.get(d, 0)
        pct = 100 * c / n_total
        print(f"{d:>5}  {c:>8}  {pct:>5.2f}%")

    print("\n=== true-V* depth histogram ===")
    print(f"{'depth':>5}  {'count':>8}  {'pct':>6}")
    for d in range(0, MAX_DEPTH + 1):
        c = true_total.get(d, 0)
        pct = 100 * c / n_total
        print(f"{d:>5}  {c:>8}  {pct:>5.2f}%")

    print("\n=== joint (rows=claimed, cols=true V* depth) — % within claimed ===")
    header = "  ".join([f"d{t:02d}".rjust(5) for t in range(0, MAX_DEPTH + 1)])
    print(f"{'claimed':>8}  {header}")
    for c in range(1, MAX_DEPTH + 1):
        row_total = joint[c].sum()
        if row_total == 0:
            continue
        cells = "  ".join(
            [f"{100 * joint[c, t] / row_total:>5.1f}" for t in range(0, MAX_DEPTH + 1)]
        )
        print(f"{c:>8}  {cells}")

    # Effective deep-tail coverage: fraction of states at true depth >= 12
    deep_count = sum(true_total[d] for d in range(12, MAX_DEPTH + 1))
    deep_pct = 100 * deep_count / n_total
    shallow_count = sum(true_total[d] for d in range(0, 6))
    shallow_pct = 100 * shallow_count / n_total
    print(f"\nstates at true depth <  6: {shallow_count:>6} ({shallow_pct:.2f}%)")
    print(f"states at true depth >=12: {deep_count:>6} ({deep_pct:.2f}%)")


if __name__ == "__main__":
    main()
