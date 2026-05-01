import pytest
import torch

from rubik.cube.spec import CUBE_2X2
from rubik.notation.state import dict_to_state, state_to_dict


def test_solved_round_trip():
    state = CUBE_2X2.solved_state
    d = state_to_dict(state, CUBE_2X2)
    assert set(d.keys()) == set(CUBE_2X2.faces)
    assert d["U"] == [0, 0, 0, 0]
    assert d["D"] == [5, 5, 5, 5]
    back = dict_to_state(d, CUBE_2X2)
    assert torch.equal(back, state)


def test_arbitrary_state_round_trip():
    state = (torch.arange(24, dtype=torch.int8) % 6).to(torch.int8)
    d = state_to_dict(state, CUBE_2X2)
    back = dict_to_state(d, CUBE_2X2)
    assert torch.equal(back, state)


def test_dict_face_order_independent():
    state = CUBE_2X2.solved_state
    d = state_to_dict(state, CUBE_2X2)
    shuffled = {face: d[face] for face in reversed(CUBE_2X2.faces)}
    back = dict_to_state(shuffled, CUBE_2X2)
    assert torch.equal(back, state)


def test_missing_face_raises():
    d = state_to_dict(CUBE_2X2.solved_state, CUBE_2X2)
    del d["U"]
    with pytest.raises(ValueError):
        dict_to_state(d, CUBE_2X2)


def test_unknown_face_raises():
    d = state_to_dict(CUBE_2X2.solved_state, CUBE_2X2)
    d["X"] = d.pop("U")
    with pytest.raises(ValueError):
        dict_to_state(d, CUBE_2X2)


def test_wrong_face_length_raises():
    d = state_to_dict(CUBE_2X2.solved_state, CUBE_2X2)
    d["U"] = [0, 0, 0]
    with pytest.raises(ValueError):
        dict_to_state(d, CUBE_2X2)


def test_state_wrong_shape_raises():
    bad = torch.zeros(23, dtype=torch.int8)
    with pytest.raises(ValueError):
        state_to_dict(bad, CUBE_2X2)
