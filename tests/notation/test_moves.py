import pytest

from rubik.notation.moves import (
    FACE_NAMES,
    MOVE_STRINGS,
    move_to_str,
    move_to_tuple,
    str_to_move,
    tuple_to_move,
)


def test_constants():
    assert FACE_NAMES == ("U", "L", "F", "R", "B", "D")
    assert len(MOVE_STRINGS) == 12


def test_specific_mappings():
    assert move_to_str(0) == "U"
    assert move_to_str(1) == "U'"
    assert move_to_str(11) == "D'"
    assert str_to_move("U") == 0
    assert str_to_move("U'") == 1
    assert str_to_move("D'") == 11


@pytest.mark.parametrize("idx", range(12))
def test_str_round_trip(idx):
    assert str_to_move(move_to_str(idx)) == idx


@pytest.mark.parametrize("idx", range(12))
def test_tuple_round_trip(idx):
    face_idx, direction = move_to_tuple(idx)
    assert tuple_to_move(face_idx, direction) == idx


def test_tuple_layout():
    assert move_to_tuple(0) == (0, 0)
    assert move_to_tuple(1) == (0, 1)
    assert move_to_tuple(11) == (5, 1)


def test_bad_index_raises():
    with pytest.raises(ValueError):
        move_to_str(12)
    with pytest.raises(ValueError):
        move_to_str(-1)
    with pytest.raises(ValueError):
        move_to_tuple(12)
    with pytest.raises(ValueError):
        move_to_tuple(-1)


def test_bad_string_raises():
    with pytest.raises(ValueError):
        str_to_move("Z")
    with pytest.raises(ValueError):
        str_to_move("U2")
    with pytest.raises(ValueError):
        str_to_move("")


def test_bad_tuple_raises():
    with pytest.raises(ValueError):
        tuple_to_move(6, 0)
    with pytest.raises(ValueError):
        tuple_to_move(-1, 0)
    with pytest.raises(ValueError):
        tuple_to_move(0, 2)
