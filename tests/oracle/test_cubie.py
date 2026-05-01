"""Cubie oracle tests — algebraic identities and invariants on the 2x2."""

import random
from collections import Counter

import pytest
import torch

from rubik.cube.spec import CUBE_2X2
from rubik.notation.moves import MOVE_STRINGS, str_to_move
from rubik.oracle.cubie import SOLVED, apply_move, cubie_to_tensor


def apply_sequence(state, moves):
    for m in moves:
        state = apply_move(state, m)
    return state


def parse(seq):
    return [str_to_move(s) for s in seq.split()]


def test_solved_state():
    assert SOLVED.positions == tuple(range(8))
    assert SOLVED.orientations == (0,) * 8


@pytest.mark.parametrize("m", range(12))
def test_move_quartic_identity(m):
    state = SOLVED
    for _ in range(4):
        state = apply_move(state, m)
    assert state == SOLVED


@pytest.mark.parametrize("face_idx", range(6))
def test_inverse_relation(face_idx):
    cw = face_idx * 2
    ccw = face_idx * 2 + 1
    triple_cw = apply_sequence(SOLVED, [cw, cw, cw])
    single_ccw = apply_move(SOLVED, ccw)
    assert triple_cw == single_ccw


def test_sexy_move_identity():
    seq = parse("R U R' U'") * 6
    assert apply_sequence(SOLVED, seq) == SOLVED


def test_sune_identity():
    seq = parse("R U R' U R U U R'") * 6
    assert apply_sequence(SOLVED, seq) == SOLVED


def test_color_multiset_preservation():
    rng = random.Random(42)
    seq = [rng.randrange(12) for _ in range(50)]
    state = apply_sequence(SOLVED, seq)
    tensor = cubie_to_tensor(state, CUBE_2X2)
    counts = Counter(tensor.tolist())
    assert counts == {c: 4 for c in range(6)}


def test_random_walk_no_divergence():
    rng = random.Random(0)
    state = SOLVED
    for _ in range(1000):
        m = rng.randrange(12)
        state = apply_move(state, m)
        assert set(state.positions) == set(range(8))
        assert all(o in {0, 1, 2} for o in state.orientations)
        assert sum(state.orientations) % 3 == 0


def test_round_trip_via_inverse_sequence():
    rng = random.Random(7)
    seq = [rng.randrange(12) for _ in range(30)]
    scrambled = apply_sequence(SOLVED, seq)
    assert scrambled != SOLVED
    inverse_seq = [m ^ 1 for m in reversed(seq)]
    assert apply_sequence(scrambled, inverse_seq) == SOLVED


def test_solved_renders_to_spec_solved_state():
    assert torch.equal(cubie_to_tensor(SOLVED, CUBE_2X2), CUBE_2X2.solved_state)


@pytest.mark.parametrize("m", range(12))
def test_after_one_move_changes_8_stickers(m):
    moved = apply_move(SOLVED, m)
    solved_t = cubie_to_tensor(SOLVED, CUBE_2X2)
    moved_t = cubie_to_tensor(moved, CUBE_2X2)
    diffs = (solved_t != moved_t).sum().item()
    assert diffs == 8, (
        f"move {MOVE_STRINGS[m]}: expected 8 changed stickers, got {diffs}"
    )
