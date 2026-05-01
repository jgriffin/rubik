import dataclasses

import pytest
import torch

from rubik.cube.spec import CUBE_2X2


def test_cube_2x2_basic_fields():
    assert CUBE_2X2.faces == ("U", "L", "F", "R", "B", "D")
    assert len(CUBE_2X2.faces) == 6
    assert CUBE_2X2.stickers_per_face == 4
    assert CUBE_2X2.n_stickers == 24
    assert CUBE_2X2.n_faces == 6
    assert CUBE_2X2.n_moves == 12
    assert CUBE_2X2.n_colors == 6
    assert CUBE_2X2.size == 2
    assert CUBE_2X2.name == "2x2"


def test_solved_state_layout():
    state = CUBE_2X2.solved_state
    assert state.dtype == torch.int8
    assert state.shape == (24,)
    expected = torch.tensor([c for c in range(6) for _ in range(4)], dtype=torch.int8)
    assert torch.equal(state, expected)


def test_solved_state_returns_fresh_tensor():
    a = CUBE_2X2.solved_state
    a[0] = 99
    b = CUBE_2X2.solved_state
    assert b[0] == 0


def test_cube_spec_is_frozen():
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        CUBE_2X2.size = 3  # type: ignore[misc]


def test_cube_spec_is_hashable():
    hash(CUBE_2X2)
