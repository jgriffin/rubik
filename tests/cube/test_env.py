"""Tensor cube tests — identities, snapshot drift, 10k oracle equivalence."""

import random
from collections import Counter

import pytest
import torch

from rubik.cube.env import (
    MOVE_PERM_2X2,
    apply_move_sequence,
    apply_moves,
    is_solved,
    random_scrambles,
    valid_next_moves_mask,
)
from rubik.cube.spec import CUBE_2X2
from rubik.notation.moves import FACE_NAMES, str_to_move
from rubik.oracle.cubie import (
    CORNER_COLORS,
    FACE_SLOTS,
    SLOT_FACETS,
    SOLVED,
    apply_move as oracle_apply_move,
    cubie_to_tensor,
)


def parse(seq: str) -> list[int]:
    return [str_to_move(s) for s in seq.split()]


def test_solved_is_solved():
    assert is_solved(CUBE_2X2.solved_state, CUBE_2X2).item() is True
    scrambled = apply_moves(CUBE_2X2.solved_state, torch.tensor(6), CUBE_2X2)
    assert is_solved(scrambled, CUBE_2X2).item() is False


def test_move_perm_matches_oracle():
    # Re-derive the (12, 24) sticker permutation from the oracle and assert
    # equality with the embedded snapshot. Drift catcher: any change to the
    # oracle's sticker geometry that isn't reflected in env.py fails here.
    facet_to_sticker: dict[tuple[int, int], int] = {}
    for face_idx, face in enumerate(FACE_NAMES):
        for within_idx, slot in enumerate(FACE_SLOTS[face]):
            k = SLOT_FACETS[slot].index(face)
            facet_to_sticker[(slot, k)] = face_idx * CUBE_2X2.stickers_per_face + within_idx

    expected = torch.zeros((12, CUBE_2X2.n_stickers), dtype=torch.int64)
    for move_idx in range(CUBE_2X2.n_moves):
        moved = oracle_apply_move(SOLVED, move_idx)
        for face_idx, face in enumerate(FACE_NAMES):
            for within_idx, slot in enumerate(FACE_SLOTS[face]):
                cubie_id = moved.positions[slot]
                ori = moved.orientations[slot]
                k = (SLOT_FACETS[slot].index(face) - ori) % 3
                sticker_idx = face_idx * CUBE_2X2.stickers_per_face + within_idx
                expected[move_idx, sticker_idx] = facet_to_sticker[(cubie_id, k)]
    # Touch CORNER_COLORS so the import is exercised (sanity that module
    # geometry constants haven't drifted in shape).
    assert len(CORNER_COLORS) == 8
    assert torch.equal(MOVE_PERM_2X2, expected)


@pytest.mark.parametrize("m", range(12))
def test_apply_moves_quartic_identity(m):
    state = CUBE_2X2.solved_state
    for _ in range(4):
        state = apply_moves(state, torch.tensor(m), CUBE_2X2)
    assert torch.equal(state, CUBE_2X2.solved_state)


@pytest.mark.parametrize("face_idx", range(6))
def test_apply_moves_inverse_relation(face_idx):
    cw = face_idx * 2
    ccw = face_idx * 2 + 1
    triple_cw = apply_move_sequence(CUBE_2X2.solved_state, [cw, cw, cw], CUBE_2X2)
    single_ccw = apply_moves(CUBE_2X2.solved_state, torch.tensor(ccw), CUBE_2X2)
    assert torch.equal(triple_cw, single_ccw)


def test_apply_moves_sexy_identity():
    seq = parse("R U R' U'") * 6
    result = apply_move_sequence(CUBE_2X2.solved_state, seq, CUBE_2X2)
    assert torch.equal(result, CUBE_2X2.solved_state)


def test_apply_moves_sune_identity():
    seq = parse("R U R' U R U U R'") * 6
    result = apply_move_sequence(CUBE_2X2.solved_state, seq, CUBE_2X2)
    assert torch.equal(result, CUBE_2X2.solved_state)


# Direction-sensitive ground truth: applying each face's CW move to solved
# must render the same tensor the oracle produces. Catches R<->R' inversion.
@pytest.mark.parametrize("face", list(FACE_NAMES))
def test_cw_direction_tensor(face):
    move_idx = str_to_move(face)
    tensor_result = apply_moves(CUBE_2X2.solved_state, torch.tensor(move_idx), CUBE_2X2)
    oracle_result = cubie_to_tensor(oracle_apply_move(SOLVED, move_idx), CUBE_2X2)
    assert torch.equal(tensor_result, oracle_result), (
        f"{face} CW: tensor cube disagrees with oracle"
    )


def test_oracle_equivalence_10k():
    # Acceptance sweep. 10k random sequences of depth [1, 30]. Tensor result
    # must equal cubie_to_tensor(oracle_state) exactly.
    rng = random.Random(0)
    for _ in range(10_000):
        depth = rng.randrange(1, 31)
        seq = [rng.randrange(12) for _ in range(depth)]
        oracle_state = SOLVED
        for m in seq:
            oracle_state = oracle_apply_move(oracle_state, m)
        tensor_state = apply_move_sequence(CUBE_2X2.solved_state, seq, CUBE_2X2)
        assert torch.equal(tensor_state, cubie_to_tensor(oracle_state, CUBE_2X2))


def test_round_trip_via_inverse_sequence():
    g = torch.Generator().manual_seed(0)
    moves = torch.randint(low=0, high=12, size=(30,), generator=g)
    scrambled = apply_move_sequence(CUBE_2X2.solved_state, moves.tolist(), CUBE_2X2)
    assert not is_solved(scrambled, CUBE_2X2).item()
    inverse_seq = [int(m ^ 1) for m in reversed(moves.tolist())]
    recovered = apply_move_sequence(scrambled, inverse_seq, CUBE_2X2)
    assert is_solved(recovered, CUBE_2X2).item()


def test_color_multiset_preservation_tensor():
    rng = random.Random(42)
    seq = [rng.randrange(12) for _ in range(50)]
    state = apply_move_sequence(CUBE_2X2.solved_state, seq, CUBE_2X2)
    counts = Counter(state.tolist())
    assert counts == {c: 4 for c in range(6)}


def test_apply_moves_batched():
    # 4 solved states, apply 4 different moves in one call; spot-check
    # against per-state scalar calls.
    moves = torch.tensor([0, 4, 6, 10])  # U, F, R, D
    batch = CUBE_2X2.solved_state.unsqueeze(0).expand(4, -1).clone()
    out = apply_moves(batch, moves, CUBE_2X2)
    assert out.shape == (4, 24)
    assert out.dtype == torch.int8
    for i, m in enumerate(moves.tolist()):
        single = apply_moves(CUBE_2X2.solved_state, torch.tensor(m), CUBE_2X2)
        assert torch.equal(out[i], single), f"row {i} (move {m}) mismatched"


def test_random_scrambles_shape_and_legality():
    g = torch.Generator().manual_seed(0)
    states, moves = random_scrambles(
        CUBE_2X2, batch_size=64, depth=20, generator=g, prune_same_face=True
    )
    assert states.shape == (64, 24)
    assert moves.shape == (64, 20)
    assert states.dtype == torch.int8
    assert moves.dtype == torch.int64
    # Color multiset preserved per row.
    for row in states:
        counts = Counter(row.tolist())
        assert counts == {c: 4 for c in range(6)}
    # Same-face prune honored: no two consecutive moves share a face.
    same_face = ((moves[:, 1:] >> 1) == (moves[:, :-1] >> 1)).any().item()
    assert not same_face


def test_random_scrambles_no_prune_allows_same_face():
    # Without the prune flag, depth=2 can produce same-face pairs.
    g = torch.Generator().manual_seed(0)
    _, moves = random_scrambles(
        CUBE_2X2, batch_size=512, depth=2, generator=g, prune_same_face=False
    )
    same_face_count = ((moves[:, 1] >> 1) == (moves[:, 0] >> 1)).sum().item()
    assert same_face_count > 0


def test_valid_next_moves_mask():
    # None -> all True.
    mask = valid_next_moves_mask(None, CUBE_2X2)
    assert mask.shape == (12,)
    assert mask.all().item()

    # Scalar prev=R (idx 6): R (6) and R' (7) are masked out, others True.
    mask_r = valid_next_moves_mask(6, CUBE_2X2)
    expected = [True] * 12
    expected[6] = False
    expected[7] = False
    assert mask_r.tolist() == expected


def test_device_preservation():
    state = CUBE_2X2.solved_state
    out = apply_moves(state, torch.tensor(6), CUBE_2X2)
    assert out.device == state.device  # CPU only at M2; MPS lives in M4


def test_dtype_preservation():
    state = CUBE_2X2.solved_state
    assert state.dtype == torch.int8
    out = apply_moves(state, torch.tensor(6), CUBE_2X2)
    assert out.dtype == torch.int8
