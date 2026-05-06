"""Build a depth-stratified V* eval set for DAVI training evaluation.

Methodology pre-tier deliverable — see
``plans/m5-davi-methodology.md`` §"Pre-tier deliverable". The supervised
T1c run revealed that a random val subset of V* under-samples the rare
tail depths (depth 0/1/2/14), so macro-MAE is computed only over the
depths that happened to land in the sample. A frozen depth-stratified
eval set fixes this: macro-MAE is always over the same 14 depths.

Spec:
- 100 canonical states per depth d in {1, ..., 14}.
- For depths whose bucket size >= 100, sample without replacement.
- For depths whose bucket size < 100, sample with replacement
  (depth 1 has 6 states, depth 2 has 27, depth 14 has 276 — only the
  first two need with-replacement at this size).
- Depth 0 is excluded (it's the solved state, exactly one of them; the
  network should never need to predict it during DAVI).
- Frozen seed = 0.

Output: ``experiments/davi-2x2/eval_set.npz`` with keys
``states`` (1400, 24) int8, ``depths`` (1400,) int8, ``seed`` scalar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
V_STAR_CACHE = REPO_ROOT / "data" / "v_star_2x2.npz"
EVAL_SET_PATH = REPO_ROOT / "experiments" / "davi-2x2" / "eval_set.npz"

EVAL_DEPTHS = tuple(range(1, 15))  # depths 1..14
N_PER_DEPTH = 100
SEED = 0


def main() -> None:
    if not V_STAR_CACHE.exists():
        raise FileNotFoundError(
            f"V* cache not found at {V_STAR_CACHE}. Run "
            "`uv run python -m rubik.oracle.v_star_2x2` first."
        )

    with np.load(V_STAR_CACHE) as data:
        all_states = data["states"]  # (3674160, 24) int8
        all_depths = data["depths"]  # (3674160,) int8

    rng = np.random.default_rng(SEED)
    sampled_states_parts: list[np.ndarray] = []
    sampled_depths_parts: list[np.ndarray] = []

    for d in EVAL_DEPTHS:
        bucket_idx = np.flatnonzero(all_depths == d)
        bucket_size = bucket_idx.size
        replace = bucket_size < N_PER_DEPTH
        chosen = rng.choice(bucket_idx, size=N_PER_DEPTH, replace=replace)
        sampled_states_parts.append(all_states[chosen])
        sampled_depths_parts.append(np.full(N_PER_DEPTH, d, dtype=np.int8))
        print(
            f"depth {d:>2d}: bucket={bucket_size:>7d} "
            f"sampled={N_PER_DEPTH} (with_replacement={replace})"
        )

    states = np.concatenate(sampled_states_parts, axis=0).astype(np.int8)
    depths = np.concatenate(sampled_depths_parts, axis=0).astype(np.int8)

    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EVAL_SET_PATH,
        states=states,
        depths=depths,
        seed=np.int64(SEED),
    )
    size_kb = EVAL_SET_PATH.stat().st_size / 1024
    print()
    print(f"wrote {EVAL_SET_PATH.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")
    print(f"shape: states={states.shape}, depths={depths.shape}")


if __name__ == "__main__":
    main()
