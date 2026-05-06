"""Tensor cube tests for 3x3 — per-move oracle equivalence + identities.

Block 4 scope: lock the (12, 54) `MOVE_PERM_3X3` snapshot against the
oracle on a per-move basis, plus a few cheap algebraic identities that
the table must satisfy (M^4 = I, inverse = CW^3, 30-move random
round-trip via the tensor cube). The big oracle ↔ tensor random-corpus
equivalence test (10k sequences) is Block 5's job.
"""

import random

import pytest
import torch

from rubik.cube.env import (
    MOVE_PERM_3X3,
    apply_all_moves,
    apply_move_sequence,
    apply_moves,
    is_solved,
)
from rubik.cube.spec import CUBE_3X3
from rubik.notation.moves import FACE_NAMES, str_to_move
from rubik.oracle.cubie import (
    SOLVED_3X3,
    apply_move_3x3,
    cubie_to_tensor_3x3,
)

# ---------------------------------------------------------------------------
# Snapshot shape + structural invariants
# ---------------------------------------------------------------------------


def test_move_perm_3x3_shape():
    assert MOVE_PERM_3X3.shape == (12, 54)
    assert MOVE_PERM_3X3.dtype == torch.int64


@pytest.mark.parametrize("m", range(12))
def test_move_perm_3x3_is_valid_permutation(m):
    row = MOVE_PERM_3X3[m].tolist()
    assert set(row) == set(range(54)), f"move {m}: row is not a permutation of 0..53"


@pytest.mark.parametrize("m", range(12))
@pytest.mark.parametrize("face_idx", range(6))
def test_move_perm_3x3_centers_fixed(m, face_idx):
    # Center within-face index is 4; flat sticker idx = face_idx * 9 + 4.
    center_idx = face_idx * 9 + 4
    assert MOVE_PERM_3X3[m, center_idx].item() == center_idx, (
        f"move {m}, face {face_idx}: center sticker not fixed"
    )


# ---------------------------------------------------------------------------
# Per-move oracle equivalence — the load-bearing snapshot guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", range(12))
def test_apply_moves_matches_oracle_per_move(m):
    """Single-move tensor result equals oracle render for all 12 QTM moves."""
    tensor_result = apply_moves(CUBE_3X3.solved_state, torch.tensor(m), CUBE_3X3)
    oracle_result = cubie_to_tensor_3x3(apply_move_3x3(SOLVED_3X3, m), CUBE_3X3)
    assert torch.equal(tensor_result, oracle_result), (
        f"move {m}: tensor cube disagrees with oracle"
    )


@pytest.mark.parametrize("face", list(FACE_NAMES))
def test_cw_direction_tensor_3x3(face):
    # Direction-sensitive ground truth: CW move on solved must render the
    # same tensor the oracle produces. Catches R<->R' style inversion.
    move_idx = str_to_move(face)
    tensor_result = apply_moves(CUBE_3X3.solved_state, torch.tensor(move_idx), CUBE_3X3)
    oracle_result = cubie_to_tensor_3x3(apply_move_3x3(SOLVED_3X3, move_idx), CUBE_3X3)
    assert torch.equal(tensor_result, oracle_result)


# ---------------------------------------------------------------------------
# is_solved + dispatch dropping in cleanly for 3x3
# ---------------------------------------------------------------------------


def test_is_solved_3x3():
    assert is_solved(CUBE_3X3.solved_state, CUBE_3X3).item() is True
    # Any single move breaks solvedness.
    scrambled = apply_moves(CUBE_3X3.solved_state, torch.tensor(6), CUBE_3X3)
    assert is_solved(scrambled, CUBE_3X3).item() is False


def test_solved_state_shape_3x3():
    s = CUBE_3X3.solved_state
    assert s.shape == (54,)
    assert s.dtype == torch.int8


# ---------------------------------------------------------------------------
# apply_all_moves: shape + per-move match against the oracle
# ---------------------------------------------------------------------------


def test_apply_all_moves_3x3_shape():
    out = apply_all_moves(CUBE_3X3.solved_state, CUBE_3X3)
    assert out.shape == (12, 54)
    assert out.dtype == torch.int8


def test_apply_all_moves_3x3_matches_oracle_per_child():
    out = apply_all_moves(CUBE_3X3.solved_state, CUBE_3X3)
    for m in range(12):
        oracle_child = cubie_to_tensor_3x3(apply_move_3x3(SOLVED_3X3, m), CUBE_3X3)
        assert torch.equal(out[m], oracle_child), f"child {m} mismatched oracle"


# ---------------------------------------------------------------------------
# Batched apply: 1-d move_idxs over a batch of states
# ---------------------------------------------------------------------------


def test_apply_moves_batched_3x3():
    moves = torch.tensor([3, 5])  # L', F'
    batch = CUBE_3X3.solved_state.unsqueeze(0).expand(2, -1).clone()
    out = apply_moves(batch, moves, CUBE_3X3)
    assert out.shape == (2, 54)
    assert out.dtype == torch.int8
    for i, m in enumerate(moves.tolist()):
        oracle_row = cubie_to_tensor_3x3(apply_move_3x3(SOLVED_3X3, m), CUBE_3X3)
        assert torch.equal(out[i], oracle_row), f"row {i} (move {m}) mismatched"


# ---------------------------------------------------------------------------
# Algebraic identities verified through the tensor cube
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", range(12))
def test_quartic_identity_via_tensor_3x3(m):
    """M^4 = I for every QTM move via the tensor cube."""
    state = CUBE_3X3.solved_state
    for _ in range(4):
        state = apply_moves(state, torch.tensor(m), CUBE_3X3)
    assert torch.equal(state, CUBE_3X3.solved_state)


@pytest.mark.parametrize("face_idx", range(6))
def test_inverse_relation_via_tensor_3x3(face_idx):
    """CW^3 == CCW for every face via the tensor cube."""
    cw = face_idx * 2
    ccw = face_idx * 2 + 1
    triple_cw = apply_move_sequence(CUBE_3X3.solved_state, [cw, cw, cw], CUBE_3X3)
    single_ccw = apply_moves(CUBE_3X3.solved_state, torch.tensor(ccw), CUBE_3X3)
    assert torch.equal(triple_cw, single_ccw)


def test_round_trip_via_inverse_sequence_3x3():
    """30-move random scramble via tensor + inverse-reversed sequence
    returns to SOLVED. Probes the perm table along a long path, not just
    single-move correctness."""
    rng = random.Random(0)
    moves = [rng.randrange(12) for _ in range(30)]
    scrambled = apply_move_sequence(CUBE_3X3.solved_state, moves, CUBE_3X3)
    assert not is_solved(scrambled, CUBE_3X3).item()
    inverse_seq = [m ^ 1 for m in reversed(moves)]
    recovered = apply_move_sequence(scrambled, inverse_seq, CUBE_3X3)
    assert is_solved(recovered, CUBE_3X3).item()
