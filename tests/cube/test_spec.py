import dataclasses

import pytest
import torch

from rubik.cube.spec import CUBE_2X2, CUBE_3X3


def test_cube_basic_fields_2x2():
    assert CUBE_2X2.faces == ("U", "L", "F", "R", "B", "D")
    assert len(CUBE_2X2.faces) == 6
    assert CUBE_2X2.stickers_per_face == 4
    assert CUBE_2X2.n_stickers == 24
    assert CUBE_2X2.n_faces == 6
    assert CUBE_2X2.n_moves == 12
    assert CUBE_2X2.n_colors == 6
    assert CUBE_2X2.size == 2
    assert CUBE_2X2.name == "2x2"


def test_solved_state_layout_2x2():
    state = CUBE_2X2.solved_state
    assert state.dtype == torch.int8
    assert state.shape == (24,)
    expected = torch.tensor([c for c in range(6) for _ in range(4)], dtype=torch.int8)
    assert torch.equal(state, expected)


def test_solved_state_returns_fresh_tensor_2x2():
    a = CUBE_2X2.solved_state
    a[0] = 99
    b = CUBE_2X2.solved_state
    assert b[0] == 0


def test_cube_spec_is_frozen_2x2():
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        CUBE_2X2.size = 3  # type: ignore[misc]


def test_cube_spec_is_hashable_2x2():
    hash(CUBE_2X2)


def test_cube_basic_fields_3x3():
    assert CUBE_3X3.faces == ("U", "L", "F", "R", "B", "D")
    assert len(CUBE_3X3.faces) == 6
    assert CUBE_3X3.stickers_per_face == 9
    assert CUBE_3X3.n_stickers == 54
    assert CUBE_3X3.n_faces == 6
    assert CUBE_3X3.n_moves == 12
    assert CUBE_3X3.n_colors == 6
    assert CUBE_3X3.size == 3
    assert CUBE_3X3.name == "3x3"


def test_solved_state_layout_3x3():
    state = CUBE_3X3.solved_state
    assert state.dtype == torch.int8
    assert state.shape == (54,)
    expected = torch.arange(6, dtype=torch.int8).repeat_interleave(9)
    assert torch.equal(state, expected)
    # Each color appears exactly 9 times.
    for c in range(6):
        assert (state == c).sum().item() == 9


def test_solved_state_returns_fresh_tensor_3x3():
    a = CUBE_3X3.solved_state
    a[0] = 99
    b = CUBE_3X3.solved_state
    assert b[0] == 0


def test_cube_spec_is_frozen_3x3():
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        CUBE_3X3.size = 2  # type: ignore[misc]


def test_cube_spec_is_hashable_3x3():
    hash(CUBE_3X3)
