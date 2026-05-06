"""V* enumerator tests — canonicalization, BFS depth profile, lookup helpers."""

from pathlib import Path

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
    load_v_star_arrays,
    lookup_v_star,
    lookup_v_star_batch,
    sample_states_at_v_star,
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


# ---------------------------------------------------------------------------
# V*-stratified sampling — load_v_star_arrays + sample_states_at_v_star.
# These tests use the real on-disk cache when available; otherwise they
# build a tiny synthetic table so they remain runnable in any environment.
# ---------------------------------------------------------------------------


@pytest.fixture
def synth_v_star(tmp_path):
    """Synthetic V*-table fixture: 3 distinct depths, 5 states each."""
    rng = np.random.default_rng(42)
    n_per_depth = 5
    depths_list = [0, 5, 10]
    states_list, depth_arr_list = [], []
    for d in depths_list:
        # Pick canonical states by canonicalizing random states (the
        # "canonical" property is what `lookup_v_star_batch` exercises).
        rand = rng.integers(0, 6, size=(n_per_depth, 24), dtype=np.int8)
        canon = canonicalize_states_batch(rand)
        states_list.append(canon)
        depth_arr_list.append(np.full(n_per_depth, d, dtype=np.int8))
    states = np.concatenate(states_list, axis=0)
    depths = np.concatenate(depth_arr_list, axis=0)
    return states, depths


def test_load_v_star_arrays_round_trip(tmp_path, synth_v_star):
    states, depths = synth_v_star
    cache_path = tmp_path / "synth.npz"
    np.savez_compressed(cache_path, states=states, depths=depths)
    loaded_states, loaded_depths = load_v_star_arrays(cache_path)
    assert np.array_equal(loaded_states, states)
    assert np.array_equal(loaded_depths, depths)


def test_sample_states_at_v_star_returns_correct_depth(synth_v_star):
    states, depths = synth_v_star
    rng = np.random.default_rng(0)
    for d in (0, 5, 10):
        sample = sample_states_at_v_star(states, depths, d, n=8, rng=rng, rotate=False)
        assert sample.shape == (8, 24)
        for row in sample:
            canon = canonicalize_state(row)
            # canonical sample row must match one of the d-depth seeds.
            seeds = states[depths == d]
            assert any(np.array_equal(canon, seed) for seed in seeds)


def test_sample_states_at_v_star_rotate_preserves_v_star(synth_v_star):
    """Rotation preserves V* — rotated rows still canonicalize to a depth-d seed."""
    states, depths = synth_v_star
    rng = np.random.default_rng(1)
    sample = sample_states_at_v_star(states, depths, 5, n=20, rng=rng, rotate=True)
    seeds_at_5 = states[depths == 5]
    for row in sample:
        canon = canonicalize_state(row)
        assert any(np.array_equal(canon, seed) for seed in seeds_at_5)


def test_sample_states_at_v_star_with_replacement(synth_v_star):
    """Asking for more than the orbit count returns with-replacement samples."""
    states, depths = synth_v_star
    rng = np.random.default_rng(2)
    # Synth table has 5 states per depth; ask for 50.
    sample = sample_states_at_v_star(states, depths, 0, n=50, rng=rng, rotate=False)
    assert sample.shape == (50, 24)


def test_sample_states_at_v_star_zero_n(synth_v_star):
    states, depths = synth_v_star
    rng = np.random.default_rng(0)
    sample = sample_states_at_v_star(states, depths, 5, n=0, rng=rng)
    assert sample.shape == (0, 24)


def test_sample_states_at_v_star_missing_depth_raises(synth_v_star):
    states, depths = synth_v_star
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="V\\*=99"):
        sample_states_at_v_star(states, depths, 99, n=4, rng=rng)


def test_sample_states_at_v_star_negative_n_raises(synth_v_star):
    states, depths = synth_v_star
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n must be non-negative"):
        sample_states_at_v_star(states, depths, 5, n=-1, rng=rng)


def test_sample_states_at_v_star_deterministic_under_seed(synth_v_star):
    states, depths = synth_v_star
    sample_a = sample_states_at_v_star(
        states, depths, 5, n=10, rng=np.random.default_rng(7)
    )
    sample_b = sample_states_at_v_star(
        states, depths, 5, n=10, rng=np.random.default_rng(7)
    )
    assert np.array_equal(sample_a, sample_b)


def test_sample_states_at_v_star_lookup_matches_real_table():
    """End-to-end check against the real V* npz: every sampled state's V*
    matches the requested target. Skipped if the cache isn't built."""
    cache_path = Path(__file__).resolve().parents[2] / "data" / "v_star_2x2.npz"
    if not cache_path.exists():
        pytest.skip(f"V* cache not built at {cache_path}; skipping real-table test")
    from rubik.oracle.v_star_2x2 import load_v_star

    v_star = load_v_star(cache_path)
    states_arr, depths_arr = load_v_star_arrays(cache_path)
    rng = np.random.default_rng(0)
    for target in (0, 5, 10, 14):
        sample = sample_states_at_v_star(
            states_arr, depths_arr, target, n=20, rng=rng, rotate=True
        )
        looked_up = lookup_v_star_batch(sample, v_star)
        assert (looked_up == target).all(), (
            f"V*-stratified sample at target={target} returned states with "
            f"V* values {sorted(set(looked_up.tolist()))}"
        )
