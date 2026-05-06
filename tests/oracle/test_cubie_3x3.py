"""3x3 cubie oracle tests — algebraic identities + invariants on the 3x3.

Mirrors `test_cubie.py` (the 2x2 oracle suite) but extended for the 12
edges + 6 fixed centers introduced in M8-B2. The U/D-axis edge-flip
convention is verified indirectly through Sune^6, T-perm^2, and the
random round-trip — these compositions exercise both edge permutation
and edge orientation correctness.
"""

import random
from collections import Counter

import pytest
import torch

from rubik.cube.spec import CUBE_3X3
from rubik.notation.moves import MOVE_STRINGS, str_to_move
from rubik.oracle.cubie import (
    SOLVED_3X3,
    apply_move_3x3,
    cubie_to_tensor_3x3,
)


def apply_sequence(state, moves):
    for m in moves:
        state = apply_move_3x3(state, m)
    return state


def parse(seq):
    return [str_to_move(s) for s in seq.split()]


def test_solved_state_corners():
    assert SOLVED_3X3.corners.positions == tuple(range(8))
    assert SOLVED_3X3.corners.orientations == (0,) * 8


def test_solved_state_edges():
    assert SOLVED_3X3.edges.positions == tuple(range(12))
    assert SOLVED_3X3.edges.orientations == (0,) * 12


def test_solved_renders_to_spec_solved_state():
    rendered = cubie_to_tensor_3x3(SOLVED_3X3, CUBE_3X3)
    assert torch.equal(rendered, CUBE_3X3.solved_state)


def test_tensor_shape_and_dtype():
    t = cubie_to_tensor_3x3(SOLVED_3X3, CUBE_3X3)
    assert t.shape == (54,)
    assert t.dtype == torch.int8


@pytest.mark.parametrize("m", range(12))
def test_move_quartic_identity(m):
    state = SOLVED_3X3
    for _ in range(4):
        state = apply_move_3x3(state, m)
    assert state == SOLVED_3X3


@pytest.mark.parametrize("m", range(12))
def test_move_inverse_identity(m):
    """M . M' == identity for every move."""
    inverse = m ^ 1
    state = apply_move_3x3(SOLVED_3X3, m)
    state = apply_move_3x3(state, inverse)
    assert state == SOLVED_3X3


@pytest.mark.parametrize("face_idx", range(6))
def test_inverse_relation_is_triple_cw(face_idx):
    """M' == M^3 for every face."""
    cw = face_idx * 2
    ccw = face_idx * 2 + 1
    triple_cw = apply_sequence(SOLVED_3X3, [cw, cw, cw])
    single_ccw = apply_move_3x3(SOLVED_3X3, ccw)
    assert triple_cw == single_ccw


def test_sexy_move_identity():
    """(R U R' U')^6 == identity."""
    seq = parse("R U R' U'") * 6
    assert apply_sequence(SOLVED_3X3, seq) == SOLVED_3X3


def test_sune_identity():
    """(R U R' U R U U R')^6 == identity. UU = U applied twice (no U2 in QTM)."""
    seq = parse("R U R' U R U U R'") * 6
    assert apply_sequence(SOLVED_3X3, seq) == SOLVED_3X3


def test_t_perm_identity():
    """(T-perm)^2 == identity. T-perm is its own inverse."""
    seq = parse("R U R' U' R' F R R U' R' U' R U R' F'") * 2
    assert apply_sequence(SOLVED_3X3, seq) == SOLVED_3X3


@pytest.mark.parametrize("m", range(12))
def test_after_one_move_changes_12_stickers(m):
    """A QTM face turn changes exactly 12 stickers on the 3x3 (4 corner
    side stickers × 2 side faces + 4 edge side stickers × 1 side face per
    cycle... actually: 4 corners × 2 side stickers + 4 edges × 1 side
    sticker = 12). The face's own stickers don't change color."""
    moved = apply_move_3x3(SOLVED_3X3, m)
    solved_t = cubie_to_tensor_3x3(SOLVED_3X3, CUBE_3X3)
    moved_t = cubie_to_tensor_3x3(moved, CUBE_3X3)
    diffs = (solved_t != moved_t).sum().item()
    assert diffs == 12, (
        f"move {MOVE_STRINGS[m]}: expected 12 changed stickers, got {diffs}"
    )


def test_color_multiset_preservation_random_sequences():
    """100 random sequences (depth 1..30): sticker tensor has exactly 9 of
    each color."""
    rng = random.Random(42)
    for trial in range(100):
        depth = rng.randint(1, 30)
        seq = [rng.randrange(12) for _ in range(depth)]
        state = apply_sequence(SOLVED_3X3, seq)
        tensor = cubie_to_tensor_3x3(state, CUBE_3X3)
        counts = Counter(tensor.tolist())
        assert counts == {c: 9 for c in range(6)}, (
            f"trial {trial} (seq={seq}): colors {counts}"
        )


def test_random_walk_invariants():
    """1000-step random walk: positions remain a permutation, orientations
    in valid range, and orientation parity invariants hold."""
    rng = random.Random(0)
    state = SOLVED_3X3
    for _ in range(1000):
        m = rng.randrange(12)
        state = apply_move_3x3(state, m)
        # Corner invariants.
        assert set(state.corners.positions) == set(range(8))
        assert all(o in {0, 1, 2} for o in state.corners.orientations)
        assert sum(state.corners.orientations) % 3 == 0
        # Edge invariants.
        assert set(state.edges.positions) == set(range(12))
        assert all(o in {0, 1} for o in state.edges.orientations)
        # Edge orientation parity is invariant under QTM (each face turn
        # flips 0 or 4 edges, both even).
        assert sum(state.edges.orientations) % 2 == 0


def test_round_trip_via_inverse_sequence():
    """100 random sequences: applying inverse-reversed sequence returns
    to SOLVED_3X3."""
    rng = random.Random(7)
    for trial in range(100):
        depth = rng.randint(1, 30)
        seq = [rng.randrange(12) for _ in range(depth)]
        scrambled = apply_sequence(SOLVED_3X3, seq)
        inverse_seq = [m ^ 1 for m in reversed(seq)]
        recovered = apply_sequence(scrambled, inverse_seq)
        assert recovered == SOLVED_3X3, f"trial {trial} (seq={seq}): round-trip failed"


def test_centers_dont_move():
    """Center stickers stay at their face's center index regardless of
    moves. Verified by checking the within-face index 4 of each face
    block always equals the face's color index, after arbitrary
    sequences."""
    rng = random.Random(123)
    for trial in range(50):
        depth = rng.randint(1, 30)
        seq = [rng.randrange(12) for _ in range(depth)]
        state = apply_sequence(SOLVED_3X3, seq)
        tensor = cubie_to_tensor_3x3(state, CUBE_3X3)
        for face_idx in range(6):
            center_sticker_idx = face_idx * 9 + 4
            assert tensor[center_sticker_idx].item() == face_idx, (
                f"trial {trial} (seq={seq}): center of face {face_idx} "
                f"has color {tensor[center_sticker_idx].item()}"
            )
