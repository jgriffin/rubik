"""3x3 oracle ↔ tensor equivalence — the M2 acceptance corpus, 3x3 edition.

Mirrors `tests/cube/test_env.py::test_oracle_equivalence_10k` for the 3x3
stack: 10k random move sequences (depth 1..30, seed 42), the cubie
oracle's render must match the tensor cube's state byte-for-byte. Block 4
already verified per-move equivalence; this is the long-path corpus that
would catch any bug propagating across composition.

Identity tests (M^4=I, sexy^6=I, Sune^6=I, T-perm^2=I) are replicated
here against the *tensor* cube — Block 2 verified them against the
oracle. Belt-and-suspenders against any bug in `MOVE_PERM_3X3` that
single-move equivalence would miss but composition would expose.

`apply_all_moves` cross-check + random-batch equivalence are likely
redundant with the 10k corpus, but they're cheap regression boundaries.
"""

import random
from collections import Counter

import pytest
import torch

from rubik.cube.env import (
    apply_all_moves,
    apply_move_sequence,
    apply_moves,
)
from rubik.cube.spec import CUBE_3X3
from rubik.notation.moves import str_to_move
from rubik.oracle.cubie import (
    SOLVED_3X3,
    apply_move_3x3,
    cubie_to_tensor_3x3,
)


def parse(seq: str) -> list[int]:
    return [str_to_move(s) for s in seq.split()]


# ---------------------------------------------------------------------------
# The big corpus — 10k random sequences
# ---------------------------------------------------------------------------


def test_oracle_equivalence_10k_3x3():
    """10k random sequences (depth 1..30, seed 42). For each, the tensor
    cube's final state must equal the cubie oracle's render. One
    end-of-sequence comparison per trial is sufficient: any single bug in
    `MOVE_PERM_3X3` would propagate through composition and surface here.
    """
    rng = random.Random(42)
    for _ in range(10_000):
        depth = rng.randrange(1, 31)
        seq = [rng.randrange(12) for _ in range(depth)]
        oracle_state = SOLVED_3X3
        for m in seq:
            oracle_state = apply_move_3x3(oracle_state, m)
        tensor_state = apply_move_sequence(CUBE_3X3.solved_state, seq, CUBE_3X3)
        assert torch.equal(tensor_state, cubie_to_tensor_3x3(oracle_state, CUBE_3X3))


# ---------------------------------------------------------------------------
# Identities verified against the tensor cube
# (Block 2 verified these against the oracle; replicating here guards
# against any bug in MOVE_PERM_3X3 that would flip M^4=I or sexy^6=I.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", range(12))
def test_quartic_identity_tensor_3x3(m):
    """M^4 == I via the tensor cube, every QTM move."""
    state = CUBE_3X3.solved_state
    for _ in range(4):
        state = apply_moves(state, torch.tensor(m), CUBE_3X3)
    assert torch.equal(state, CUBE_3X3.solved_state)


@pytest.mark.parametrize("m", range(12))
def test_inverse_relation_tensor_3x3(m):
    """M . M' == I via the tensor cube, every QTM move."""
    inverse = m ^ 1
    state = apply_moves(CUBE_3X3.solved_state, torch.tensor(m), CUBE_3X3)
    state = apply_moves(state, torch.tensor(inverse), CUBE_3X3)
    assert torch.equal(state, CUBE_3X3.solved_state)


def test_sexy_move_identity_tensor_3x3():
    """(R U R' U')^6 == I via the tensor cube."""
    seq = parse("R U R' U'") * 6
    result = apply_move_sequence(CUBE_3X3.solved_state, seq, CUBE_3X3)
    assert torch.equal(result, CUBE_3X3.solved_state)


def test_sune_identity_tensor_3x3():
    """(R U R' U R U U R')^6 == I via the tensor cube. UU = U applied
    twice (no U2 in QTM)."""
    seq = parse("R U R' U R U U R'") * 6
    result = apply_move_sequence(CUBE_3X3.solved_state, seq, CUBE_3X3)
    assert torch.equal(result, CUBE_3X3.solved_state)


def test_t_perm_identity_tensor_3x3():
    """(T-perm)^2 == I via the tensor cube. T-perm is its own inverse."""
    seq = parse("R U R' U' R' F R R U' R' U' R U R' F'") * 2
    result = apply_move_sequence(CUBE_3X3.solved_state, seq, CUBE_3X3)
    assert torch.equal(result, CUBE_3X3.solved_state)


def test_color_multiset_preservation_tensor_3x3():
    """100 random sequences via the tensor cube: exactly 9 stickers of
    each color."""
    rng = random.Random(42)
    for trial in range(100):
        depth = rng.randint(1, 30)
        seq = [rng.randrange(12) for _ in range(depth)]
        state = apply_move_sequence(CUBE_3X3.solved_state, seq, CUBE_3X3)
        counts = Counter(state.tolist())
        assert counts == {c: 9 for c in range(6)}, (
            f"trial {trial} (seq={seq}): colors {counts}"
        )


# ---------------------------------------------------------------------------
# apply_all_moves cross-check vs. per-move apply_moves
# ---------------------------------------------------------------------------


def test_apply_all_moves_3x3_matches_per_move_random_states():
    """50 random states. For each, `apply_all_moves(state)` rows match
    `apply_moves(state, [m])` for every m."""
    rng = random.Random(0)
    for trial in range(50):
        depth = rng.randint(1, 30)
        seq = [rng.randrange(12) for _ in range(depth)]
        state = apply_move_sequence(CUBE_3X3.solved_state, seq, CUBE_3X3)
        all_children = apply_all_moves(state, CUBE_3X3)
        assert all_children.shape == (12, 54)
        for m in range(12):
            single = apply_moves(state, torch.tensor(m), CUBE_3X3)
            assert torch.equal(all_children[m], single), (
                f"trial {trial} move {m}: apply_all_moves disagrees with apply_moves"
            )


# ---------------------------------------------------------------------------
# Random-batch equivalence: batched apply matches per-state oracle apply
# ---------------------------------------------------------------------------


def test_random_batch_equivalence_3x3():
    """64 random states × 12 moves: batched `apply_moves` agrees with the
    per-state oracle render for each move."""
    rng = random.Random(123)
    # Build 64 random states via the tensor cube (Block 4 already locked
    # tensor ↔ oracle agreement on these); the equivalence probe is on the
    # *batched* apply path that DAVI / beam search use.
    states = []
    oracle_states = []
    for _ in range(64):
        depth = rng.randint(1, 30)
        seq = [rng.randrange(12) for _ in range(depth)]
        states.append(apply_move_sequence(CUBE_3X3.solved_state, seq, CUBE_3X3))
        oracle = SOLVED_3X3
        for mv in seq:
            oracle = apply_move_3x3(oracle, mv)
        oracle_states.append(oracle)
    batch = torch.stack(states)  # (64, 54)

    for m in range(12):
        moves = torch.full((64,), m, dtype=torch.int64)
        out = apply_moves(batch, moves, CUBE_3X3)
        assert out.shape == (64, 54)
        for i, oracle in enumerate(oracle_states):
            expected = cubie_to_tensor_3x3(apply_move_3x3(oracle, m), CUBE_3X3)
            assert torch.equal(out[i], expected), (
                f"move {m}, row {i}: batched tensor apply disagrees with oracle"
            )
