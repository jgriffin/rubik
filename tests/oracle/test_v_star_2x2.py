"""V* enumerator tests — canonicalization, BFS depth profile, lookup helpers."""

import numpy as np
import pytest
import torch

from rubik.cube.env import apply_moves, random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.oracle.v_star_2x2 import (
    _CUBE_ROT_PERM,
    canonicalize_state,
    canonicalize_states_batch,
    compute_v_star_2x2,
    load_v_star,
    lookup_v_star,
    lookup_v_star_batch,
    save_v_star,
)


def test_cube_rotation_group_has_24_elements():
    assert _CUBE_ROT_PERM.shape == (24, 24)
    # Each row is a permutation of 0..23.
    for r in _CUBE_ROT_PERM:
        assert sorted(r.tolist()) == list(range(24))
    # Identity is among the 24.
    has_identity = any(np.array_equal(r, np.arange(24)) for r in _CUBE_ROT_PERM)
    assert has_identity


def test_canonicalize_idempotent():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.integers(0, 6, size=24, dtype=np.int8)
        c1 = canonicalize_state(s)
        c2 = canonicalize_state(c1)
        assert np.array_equal(c1, c2)


def test_canonicalize_invariant_under_rotation():
    """All 24 rotations of a state share the same canonical form."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.integers(0, 6, size=24, dtype=np.int8)
        canon = canonicalize_state(s)
        for rot in _CUBE_ROT_PERM:
            rotated = s[rot]
            assert np.array_equal(canonicalize_state(rotated), canon)


def test_canonicalize_states_batch_matches_per_state():
    rng = np.random.default_rng(2)
    states = rng.integers(0, 6, size=(50, 24), dtype=np.int8)
    batch_canon = canonicalize_states_batch(states)
    for i, s in enumerate(states):
        assert np.array_equal(canonicalize_state(s), batch_canon[i])


def test_canonicalize_accepts_torch_tensor():
    s = CUBE_2X2.solved_state
    np_canon = canonicalize_state(s.numpy())
    torch_canon = canonicalize_state(s)
    assert np.array_equal(np_canon, torch_canon)


def test_v_star_smoke_depth_5():
    """Depth-5 BFS covers a meaningful slice of the QTM state space.

    Profile is the canonical 2x2-QTM-mod-rotations sequence — verified
    against the full BFS in `experiments/davi-2x2/` runs.
    """
    v = compute_v_star_2x2(CUBE_2X2, max_depth=5)
    profile = [0] * 6
    for d in v.values():
        profile[d] += 1
    assert profile == [1, 6, 27, 120, 534, 2256]
    assert len(v) == sum(profile)


def test_v_star_solved_has_depth_zero():
    v = compute_v_star_2x2(CUBE_2X2, max_depth=2)
    canon = canonicalize_state(CUBE_2X2.solved_state)
    assert v[canon.tobytes()] == 0


def test_v_star_single_move_has_depth_one():
    v = compute_v_star_2x2(CUBE_2X2, max_depth=2)
    for m in range(CUBE_2X2.n_moves):
        child = apply_moves(CUBE_2X2.solved_state, m, CUBE_2X2)
        assert lookup_v_star(child.numpy(), v) == 1


def test_v_star_random_scramble_bounded_by_depth():
    """For a depth-d scramble, V* ≤ d (V* is the optimal cost)."""
    v = compute_v_star_2x2(CUBE_2X2, max_depth=5)
    rng = torch.Generator().manual_seed(42)
    for d in (1, 2, 3, 4, 5):
        states, _ = random_scrambles(CUBE_2X2, batch_size=20, depth=d, generator=rng)
        for s in states:
            assert lookup_v_star(s.numpy(), v) <= d


def test_lookup_v_star_batch_matches_per_state():
    v = compute_v_star_2x2(CUBE_2X2, max_depth=4)
    rng = torch.Generator().manual_seed(7)
    states, _ = random_scrambles(CUBE_2X2, batch_size=30, depth=3, generator=rng)
    batch_depths = lookup_v_star_batch(states, v)
    per_state = np.array([lookup_v_star(s, v) for s in states.numpy()], dtype=np.int8)
    assert np.array_equal(batch_depths, per_state)


def test_save_load_roundtrip(tmp_path):
    v = compute_v_star_2x2(CUBE_2X2, max_depth=3)
    cache = tmp_path / "v_star.npz"
    save_v_star(v, cache)
    assert cache.exists()
    loaded = load_v_star(cache)
    assert loaded == v


def test_compute_v_star_uses_cache(tmp_path):
    """cache_path: if exists, load instead of recomputing."""
    cache = tmp_path / "v_star.npz"
    v_first = compute_v_star_2x2(CUBE_2X2, max_depth=3, cache_path=cache)
    assert cache.exists()
    # Second call should load from disk; verify by checking equality.
    v_second = compute_v_star_2x2(CUBE_2X2, max_depth=99, cache_path=cache)
    # max_depth=99 would normally do a full BFS — but cache short-circuits it.
    assert v_first == v_second


def test_compute_v_star_rejects_non_2x2():
    from dataclasses import replace

    spec_3x3_stub = replace(CUBE_2X2, name="3x3", size=3)
    with pytest.raises(ValueError, match="2x2"):
        compute_v_star_2x2(spec_3x3_stub, max_depth=1)


def test_canonicalize_rejects_wrong_shape():
    with pytest.raises(ValueError, match="\\(24,\\)"):
        canonicalize_state(np.zeros(20, dtype=np.int8))


def test_canonicalize_states_batch_rejects_wrong_shape():
    with pytest.raises(ValueError, match="\\(B, 24\\)"):
        canonicalize_states_batch(np.zeros((5, 20), dtype=np.int8))
