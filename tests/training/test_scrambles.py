"""Tests for `generate_adi_batch` — ADI batch shape, depth balance, determinism."""

from collections import Counter

import pytest
import torch

from rubik.cube.env import is_solved
from rubik.cube.spec import CUBE_2X2
from rubik.training.scrambles import generate_adi_batch


def test_returns_correct_shapes():
    states, depths, last_faces = generate_adi_batch(
        CUBE_2X2, batch_size=84, max_depth=14
    )
    assert states.shape == (84, CUBE_2X2.n_stickers)
    assert states.dtype == torch.int8
    assert depths.shape == (84,)
    assert depths.dtype == torch.int64
    assert last_faces.shape == (84,)
    assert last_faces.dtype == torch.int64


def test_depths_balanced_when_divisible():
    """B = max_depth × k → exactly k states per depth."""
    states, depths, last_faces = generate_adi_batch(
        CUBE_2X2, batch_size=14 * 5, max_depth=14
    )
    counts = Counter(depths.tolist())
    assert all(counts[d] == 5 for d in range(1, 15))


def test_depths_remainder_goes_to_early():
    """B = 14 × k + r → r extra states at depths 1..r."""
    _, depths, _ = generate_adi_batch(CUBE_2X2, batch_size=14 * 3 + 5, max_depth=14)
    counts = Counter(depths.tolist())
    for d in range(1, 6):
        assert counts[d] == 4
    for d in range(6, 15):
        assert counts[d] == 3


def test_depths_in_range():
    _, depths, _ = generate_adi_batch(CUBE_2X2, batch_size=200, max_depth=10)
    assert depths.min() >= 1
    assert depths.max() <= 10


def test_last_faces_in_range():
    _, _, last_faces = generate_adi_batch(CUBE_2X2, batch_size=200, max_depth=14)
    assert last_faces.min() >= 0
    assert last_faces.max() < CUBE_2X2.n_faces


def test_states_not_solved():
    """Every row has depth >= 1, so no state should be solved (with same-face
    pruning, no scramble cancels back to solved)."""
    states, _, _ = generate_adi_batch(CUBE_2X2, batch_size=200, max_depth=14)
    assert not is_solved(states, CUBE_2X2).any()


def test_deterministic_under_seed():
    rng1 = torch.Generator().manual_seed(123)
    rng2 = torch.Generator().manual_seed(123)
    s1, d1, lf1 = generate_adi_batch(
        CUBE_2X2, batch_size=140, max_depth=14, generator=rng1
    )
    s2, d2, lf2 = generate_adi_batch(
        CUBE_2X2, batch_size=140, max_depth=14, generator=rng2
    )
    assert torch.equal(s1, s2)
    assert torch.equal(d1, d2)
    assert torch.equal(lf1, lf2)


def test_different_seeds_give_different_batches():
    rng1 = torch.Generator().manual_seed(0)
    rng2 = torch.Generator().manual_seed(1)
    s1, _, _ = generate_adi_batch(
        CUBE_2X2, batch_size=140, max_depth=14, generator=rng1
    )
    s2, _, _ = generate_adi_batch(
        CUBE_2X2, batch_size=140, max_depth=14, generator=rng2
    )
    assert not torch.equal(s1, s2)


def test_empty_batch():
    states, depths, last_faces = generate_adi_batch(
        CUBE_2X2, batch_size=0, max_depth=14
    )
    assert states.shape == (0, CUBE_2X2.n_stickers)
    assert depths.shape == (0,)
    assert last_faces.shape == (0,)


def test_max_depth_one():
    """Edge case: max_depth=1 → all rows at depth 1."""
    _, depths, _ = generate_adi_batch(CUBE_2X2, batch_size=20, max_depth=1)
    assert (depths == 1).all()


def test_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        generate_adi_batch(CUBE_2X2, batch_size=-1, max_depth=14)


def test_rejects_invalid_max_depth():
    with pytest.raises(ValueError, match="max_depth"):
        generate_adi_batch(CUBE_2X2, batch_size=10, max_depth=0)


def test_states_consistent_with_depth_via_v_star():
    """Each generated state's V* lookup should be ≤ its claimed depth.

    V* is the optimal cost-to-go; a depth-d scramble can be optimal in fewer
    moves but never more. Cross-validates scramble pipeline + V* + canon.
    """
    from rubik.oracle.v_star_2x2 import compute_v_star_2x2, lookup_v_star_batch

    v = compute_v_star_2x2(CUBE_2X2, max_depth=5)
    states, depths, _ = generate_adi_batch(CUBE_2X2, batch_size=70, max_depth=5)
    v_star_depths = lookup_v_star_batch(states, v)
    # claimed depth >= optimal depth
    assert (torch.from_numpy(v_star_depths) <= depths).all()
