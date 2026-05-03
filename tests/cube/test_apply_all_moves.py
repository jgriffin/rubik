"""Tests for `apply_all_moves` — DAVI / beam-search children expansion."""

import pytest
import torch

from rubik.cube.env import apply_all_moves, apply_moves, is_solved
from rubik.cube.spec import CUBE_2X2


def test_apply_all_moves_single_state_shape():
    out = apply_all_moves(CUBE_2X2.solved_state, CUBE_2X2)
    assert out.shape == (CUBE_2X2.n_moves, CUBE_2X2.n_stickers)
    assert out.dtype == torch.int8


def test_apply_all_moves_batched_shape():
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(5, -1).clone()
    out = apply_all_moves(states, CUBE_2X2)
    assert out.shape == (5, CUBE_2X2.n_moves, CUBE_2X2.n_stickers)


def test_apply_all_moves_equivalent_to_per_move_single():
    s = CUBE_2X2.solved_state
    out = apply_all_moves(s, CUBE_2X2)
    for m in range(CUBE_2X2.n_moves):
        assert torch.equal(out[m], apply_moves(s, m, CUBE_2X2))


def test_apply_all_moves_equivalent_to_per_move_batched():
    rng = torch.Generator().manual_seed(0)
    from rubik.cube.env import random_scrambles

    states, _ = random_scrambles(CUBE_2X2, batch_size=8, depth=5, generator=rng)
    out = apply_all_moves(states, CUBE_2X2)
    for m in range(CUBE_2X2.n_moves):
        per_move = apply_moves(states, torch.full((8,), m, dtype=torch.int64), CUBE_2X2)
        assert torch.equal(out[:, m], per_move)


@pytest.mark.parametrize("face_idx", range(6))
def test_apply_all_moves_inverse_round_trip(face_idx):
    """For each face, applying the CW move then CCW returns to start."""
    cw = face_idx * 2
    ccw = face_idx * 2 + 1
    s = CUBE_2X2.solved_state
    children = apply_all_moves(s, CUBE_2X2)
    after_cw = children[cw]
    grandchildren = apply_all_moves(after_cw, CUBE_2X2)
    assert torch.equal(grandchildren[ccw], s)


def test_apply_all_moves_solved_check_consistency():
    """is_solved across all 12 children of solved should be False (no
    single move solves the cube from solved)."""
    children = apply_all_moves(CUBE_2X2.solved_state, CUBE_2X2)
    assert not is_solved(children, CUBE_2X2).any()


def test_apply_all_moves_rejects_wrong_sticker_dim():
    bad = torch.zeros((3, CUBE_2X2.n_stickers - 1), dtype=torch.int8)
    with pytest.raises(ValueError, match="last dim"):
        apply_all_moves(bad, CUBE_2X2)


def test_apply_all_moves_preserves_device():
    """Stays on whatever device the input is on (CPU here, MPS at training)."""
    s = CUBE_2X2.solved_state.to("cpu")
    out = apply_all_moves(s, CUBE_2X2)
    assert out.device == s.device
