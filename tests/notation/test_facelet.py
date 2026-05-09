import pytest
import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_2X2, CUBE_3X3
from rubik.notation.facelet import facelet_to_state, state_to_facelet


def test_solved_3x3_exact_string():
    expected = (
        "UUUUUUUUU"
        + "RRRRRRRRR"
        + "FFFFFFFFF"
        + "DDDDDDDDD"
        + "LLLLLLLLL"
        + "BBBBBBBBB"
    )
    assert state_to_facelet(CUBE_3X3.solved_state, CUBE_3X3) == expected


def test_solved_2x2_exact_string():
    expected = "UUUU" + "RRRR" + "FFFF" + "DDDD" + "LLLL" + "BBBB"
    assert state_to_facelet(CUBE_2X2.solved_state, CUBE_2X2) == expected


def test_round_trip_3x3_random_scrambles():
    rng = torch.Generator().manual_seed(0)
    states, _ = random_scrambles(CUBE_3X3, batch_size=64, depth=14, generator=rng)
    for i in range(states.shape[0]):
        s = states[i]
        back = facelet_to_state(state_to_facelet(s, CUBE_3X3), CUBE_3X3)
        assert torch.equal(back, s)


def test_round_trip_2x2_random_scrambles():
    rng = torch.Generator().manual_seed(0)
    states, _ = random_scrambles(CUBE_2X2, batch_size=64, depth=8, generator=rng)
    for i in range(states.shape[0]):
        s = states[i]
        back = facelet_to_state(state_to_facelet(s, CUBE_2X2), CUBE_2X2)
        assert torch.equal(back, s)


def test_facelet_to_state_length_mismatch_raises():
    with pytest.raises(ValueError, match="54"):
        facelet_to_state("UUU", CUBE_3X3)


def test_facelet_to_state_unknown_character_raises():
    bad = "X" + "U" * 53
    with pytest.raises(ValueError, match="unknown"):
        facelet_to_state(bad, CUBE_3X3)


def test_state_to_facelet_wrong_shape_raises():
    bad = torch.zeros((2, CUBE_3X3.n_stickers), dtype=torch.int8)
    with pytest.raises(ValueError):
        state_to_facelet(bad, CUBE_3X3)


@pytest.mark.parametrize("move_idx", range(12))
def test_facelet_letter_counts_preserved_under_each_qtm_move_3x3(move_idx):
    """Each QTM move on a solved cube must keep stickers_per_face of each
    letter — catches off-by-one errors in the face-block permutation."""
    from rubik.cube.env import apply_moves

    state = apply_moves(
        CUBE_3X3.solved_state,
        torch.tensor(move_idx, dtype=torch.int64),
        CUBE_3X3,
    )
    facelet = state_to_facelet(state, CUBE_3X3)
    for face in CUBE_3X3.faces:
        assert facelet.count(face) == CUBE_3X3.stickers_per_face


@pytest.mark.parametrize("move_idx", range(12))
def test_facelet_letter_counts_preserved_under_each_qtm_move_2x2(move_idx):
    from rubik.cube.env import apply_moves

    state = apply_moves(
        CUBE_2X2.solved_state,
        torch.tensor(move_idx, dtype=torch.int64),
        CUBE_2X2,
    )
    facelet = state_to_facelet(state, CUBE_2X2)
    for face in CUBE_2X2.faces:
        assert facelet.count(face) == CUBE_2X2.stickers_per_face
