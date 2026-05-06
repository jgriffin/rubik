"""Bounded V* oracle tests — layer counts, lookup, move-shift identity.

The full K=6 oracle is built once per test module via fixture and reused
across every test (~5s build on M4 Max — would explode CI time if rebuilt
per test). Tests prefer the on-disk cache at ``data/v_star_bounded_3x3_k6.npz``
when present (built by ``scripts/build_v_star_bounded_3x3.py``); falls back
to building inline.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from rubik.cube.env import apply_moves, random_scrambles
from rubik.cube.spec import CUBE_3X3
from rubik.oracle.v_star_bounded_3x3 import (
    compute_v_star_bounded_3x3,
    load_v_star_bounded_3x3,
    load_v_star_bounded_3x3_arrays,
    lookup_v_star_bounded_3x3,
    lookup_v_star_bounded_3x3_batch,
    save_v_star_bounded_3x3,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"

# Well-known QTM layer counts from cube literature; sentinel for K=6.
EXPECTED_LAYER_COUNTS = (1, 12, 114, 1068, 10011, 93840, 878880)
EXPECTED_CUMULATIVE_K6 = sum(EXPECTED_LAYER_COUNTS)  # 983,926


@pytest.fixture(scope="module")
def v_star_k6() -> dict[bytes, int]:
    """Load (or build once) the K=6 bounded V* oracle for the whole module."""
    if CACHE_PATH.exists():
        return load_v_star_bounded_3x3(CACHE_PATH)
    return compute_v_star_bounded_3x3(CUBE_3X3, max_depth=6, verbose=False)


# ---------------------------------------------------------------------------
# Layer-count sentinels
# ---------------------------------------------------------------------------


def test_v_star_bounded_3x3_total_state_count_at_k6(v_star_k6):
    """Sentinel: K=6 BFS sees exactly 983,926 distinct states."""
    assert len(v_star_k6) == EXPECTED_CUMULATIVE_K6


def test_v_star_bounded_3x3_per_layer_counts_at_k6(v_star_k6):
    """Each BFS layer matches the cube-literature QTM count."""
    profile = [0] * 7
    for d in v_star_k6.values():
        profile[d] += 1
    assert tuple(profile) == EXPECTED_LAYER_COUNTS


# ---------------------------------------------------------------------------
# Solved-state and basic lookup invariants
# ---------------------------------------------------------------------------


def test_solved_state_has_v_star_zero(v_star_k6):
    solved_bytes = CUBE_3X3.solved_state.numpy().astype(np.int8).tobytes()
    assert v_star_k6[solved_bytes] == 0


def test_single_move_states_have_v_star_one(v_star_k6):
    """Every single-move scramble lands on a V*=1 state."""
    for m in range(CUBE_3X3.n_moves):
        child = apply_moves(CUBE_3X3.solved_state, m, CUBE_3X3)
        assert lookup_v_star_bounded_3x3(child.numpy(), v_star_k6) == 1


# ---------------------------------------------------------------------------
# Random-walk round-trip: V*[endpoint] ≤ d (often <, due to walk redundancy)
# ---------------------------------------------------------------------------


def test_v_star_bounded_3x3_random_walk_round_trip(v_star_k6):
    rng = torch.Generator().manual_seed(42)
    for d in (1, 2, 3, 4, 5, 6):
        states, _ = random_scrambles(CUBE_3X3, batch_size=50, depth=d, generator=rng)
        for s in states:
            v = lookup_v_star_bounded_3x3(s.numpy(), v_star_k6)
            assert v <= d, (
                f"walk depth {d} produced state with V*={v} > {d} —"
                " bounded oracle inconsistent with walk semantics"
            )


# ---------------------------------------------------------------------------
# Move-shift identity: applying any QTM move to V*=k state lands in {k-1,k,k+1}
# ---------------------------------------------------------------------------


def test_move_shifts_v_star_by_at_most_one(v_star_k6):
    """For V*=k ∈ {1..5}, all 12 children must have V* ∈ {k-1, k, k+1}.

    Skip k=0 (only 12 specific neighbors are at distance 1 — covered by
    `test_single_move_states_have_v_star_one`). Skip k=6 (children may
    leave the bounded oracle — V* lookup raises KeyError, which is the
    expected behavior tested separately).
    """
    if CACHE_PATH.exists():
        states, depths = load_v_star_bounded_3x3_arrays(CACHE_PATH)
    else:
        states, depths = _arrays_from_dict(v_star_k6)
    rng = np.random.default_rng(0)
    for k in (1, 2, 3, 4, 5):
        bucket_idx = np.flatnonzero(depths == k)
        # Sample a handful per bucket — exercising every state would be 10k+ at
        # k=5 × 12 children = 1.1M lookups for one bucket.
        chosen = rng.choice(bucket_idx, size=min(30, bucket_idx.size), replace=False)
        for idx in chosen:
            state = torch.from_numpy(states[idx].copy())
            for m in range(CUBE_3X3.n_moves):
                child = apply_moves(state, m, CUBE_3X3)
                v_child = lookup_v_star_bounded_3x3(child.numpy(), v_star_k6)
                assert v_child in (k - 1, k, k + 1), (
                    f"V*={k} state moved by m={m} landed on V*={v_child}"
                    f" — expected k-1, k, or k+1"
                )


def _arrays_from_dict(v_star: dict[bytes, int]) -> tuple[np.ndarray, np.ndarray]:
    """Materialize (states, depths) arrays from a dict — fallback when no cache."""
    n = len(v_star)
    states = np.empty((n, 54), dtype=np.int8)
    depths = np.empty(n, dtype=np.int8)
    for i, (key, d) in enumerate(v_star.items()):
        states[i] = np.frombuffer(key, dtype=np.int8)
        depths[i] = d
    return states, depths


# ---------------------------------------------------------------------------
# Save / load round-trip and cache short-circuit
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    """At small K, save + load reproduces the dict exactly."""
    v = compute_v_star_bounded_3x3(CUBE_3X3, max_depth=2, verbose=False)
    cache = tmp_path / "v_star_bounded_3x3.npz"
    save_v_star_bounded_3x3(v, cache)
    assert cache.exists()
    loaded = load_v_star_bounded_3x3(cache)
    assert loaded == v


def test_load_arrays_round_trip(tmp_path):
    v = compute_v_star_bounded_3x3(CUBE_3X3, max_depth=2, verbose=False)
    cache = tmp_path / "v_star_bounded_3x3.npz"
    save_v_star_bounded_3x3(v, cache)
    states, depths = load_v_star_bounded_3x3_arrays(cache)
    assert states.shape == (len(v), 54)
    assert depths.shape == (len(v),)
    # Reconstruct dict from arrays — must match.
    reconstructed = {states[i].tobytes(): int(depths[i]) for i in range(len(v))}
    assert reconstructed == v


def test_compute_v_star_uses_cache(tmp_path):
    """If cache_path exists, compute returns the cached dict instead of
    rebuilding (mirrors the v_star_2x2 short-circuit)."""
    cache = tmp_path / "v_star_bounded_3x3.npz"
    v_first = compute_v_star_bounded_3x3(
        CUBE_3X3, max_depth=2, cache_path=cache, verbose=False
    )
    assert cache.exists()
    # max_depth=99 would normally rebuild deeper, but cache short-circuits.
    v_second = compute_v_star_bounded_3x3(
        CUBE_3X3, max_depth=99, cache_path=cache, verbose=False
    )
    assert v_first == v_second


# ---------------------------------------------------------------------------
# Validation / API edge cases
# ---------------------------------------------------------------------------


def test_compute_v_star_rejects_non_3x3():
    from dataclasses import replace

    spec_2x2_stub = replace(CUBE_3X3, name="2x2", size=2, stickers_per_face=4)
    with pytest.raises(ValueError, match="3x3"):
        compute_v_star_bounded_3x3(spec_2x2_stub, max_depth=1)


def test_compute_v_star_rejects_negative_depth():
    with pytest.raises(ValueError, match="non-negative"):
        compute_v_star_bounded_3x3(CUBE_3X3, max_depth=-1)


def test_lookup_v_star_rejects_wrong_shape():
    v = compute_v_star_bounded_3x3(CUBE_3X3, max_depth=1, verbose=False)
    with pytest.raises(ValueError, match=r"\(54,\)"):
        lookup_v_star_bounded_3x3(np.zeros(50, dtype=np.int8), v)


def test_lookup_v_star_batch_rejects_wrong_shape():
    v = compute_v_star_bounded_3x3(CUBE_3X3, max_depth=1, verbose=False)
    with pytest.raises(ValueError, match=r"\(B, 54\)"):
        lookup_v_star_bounded_3x3_batch(np.zeros((5, 50), dtype=np.int8), v)


def test_lookup_batch_returns_sentinel_for_missing(v_star_k6):
    """States outside the bounded oracle (V*>K=6) get the -1 sentinel from
    the batch lookup — this is the contract relied on by ``eval_set_3x3.npz``
    (P1c) for walk_depth>K endpoints."""
    rng = torch.Generator().manual_seed(123)
    # Length-15 walks on 3x3 land on states with V* well beyond 6 with
    # high probability — at least most of the batch should be missing.
    states, _ = random_scrambles(CUBE_3X3, batch_size=20, depth=15, generator=rng)
    out = lookup_v_star_bounded_3x3_batch(states.numpy(), v_star_k6)
    assert out.dtype == np.int8
    assert out.shape == (20,)
    # We expect most or all to be missing (-1); demanding "any" keeps the
    # test deterministic-ish without baking in a specific count.
    assert (out == -1).any(), (
        "expected at least one length-15 walk endpoint to be outside K=6 oracle"
    )


def test_lookup_batch_finds_known_states(v_star_k6):
    """Single-move children are V*=1 in the oracle — batch lookup finds them."""
    children_list = []
    for m in range(CUBE_3X3.n_moves):
        children_list.append(
            apply_moves(CUBE_3X3.solved_state, m, CUBE_3X3).numpy().astype(np.int8)
        )
    children = np.stack(children_list)
    out = lookup_v_star_bounded_3x3_batch(children, v_star_k6)
    assert (out == 1).all()
