"""Generate the MOVE_PERM_2X2 sticker-permutation snapshot from the M1 oracle.

For each of the 12 QTM moves, derives the (24,) sticker permutation
`perm[i] = source_position_of_sticker_i_after_move` such that
`s_new[i] = s[perm[i]]`. Method: build a "fingerprint" rendering of the
solved state where each sticker carries its own index 0..23, apply the
move via the oracle, render again, and read off the permutation.

Prints the (12, 24) tuple-of-tuples as a Python literal for paste into
`src/rubik/cube/env.py`. Re-run whenever the oracle's sticker geometry
changes (drift caught in CI by `test_move_perm_matches_oracle`).

Run:
    uv run python scripts/move_perm_from_oracle_2x2.py
"""

import torch

from rubik.cube.spec import CUBE_2X2
from rubik.notation.moves import MOVE_STRINGS
from rubik.oracle.cubie import (
    FACE_NAMES,
    FACE_SLOTS,
    SLOT_FACETS,
    SOLVED,
    apply_move,
)


def _solved_sticker_index(face_idx: int, within_idx: int) -> int:
    return face_idx * CUBE_2X2.stickers_per_face + within_idx


def _cubie_to_position_tensor(state) -> torch.Tensor:
    """Render a CornerState carrying sticker-position indices, not colors.

    Mirrors `cubie_to_tensor` exactly but emits the SOLVED sticker index of
    each facet rather than its color. The result is a `(24,)` int64 tensor
    where `out[i]` = the sticker index that sits at slot `i` after the move,
    i.e. the permutation source-index for slot `i`.
    """
    # (cubie_id, facet_position_k) -> SOLVED sticker index. Built once from
    # the SOLVED render: walk every (face, within-face slot) and record
    # which (cubie_id, k) produced its color.
    facet_to_sticker: dict[tuple[int, int], int] = {}
    for face_idx, face in enumerate(FACE_NAMES):
        for within_idx, slot in enumerate(FACE_SLOTS[face]):
            # In SOLVED, ori=0 so k = SLOT_FACETS[slot].index(face).
            k = SLOT_FACETS[slot].index(face)
            cubie_id = slot  # SOLVED: cubie_id == slot
            facet_to_sticker[(cubie_id, k)] = _solved_sticker_index(
                face_idx, within_idx
            )

    tensor = torch.zeros(CUBE_2X2.n_stickers, dtype=torch.int64)
    for face_idx, face in enumerate(FACE_NAMES):
        for within_idx, slot in enumerate(FACE_SLOTS[face]):
            cubie_id = state.positions[slot]
            ori = state.orientations[slot]
            k = (SLOT_FACETS[slot].index(face) - ori) % 3
            sticker_idx = _solved_sticker_index(face_idx, within_idx)
            tensor[sticker_idx] = facet_to_sticker[(cubie_id, k)]
    return tensor


def build_move_perm() -> tuple[tuple[int, ...], ...]:
    perms: list[tuple[int, ...]] = []
    for move_idx in range(CUBE_2X2.n_moves):
        moved = apply_move(SOLVED, move_idx)
        perm = tuple(_cubie_to_position_tensor(moved).tolist())
        # Sanity: must be a permutation of range(24).
        assert len(perm) == CUBE_2X2.n_stickers, (
            f"move {move_idx}: wrong length {len(perm)}"
        )
        assert set(perm) == set(range(CUBE_2X2.n_stickers)), (
            f"move {move_idx}: not a permutation of 0..23"
        )
        perms.append(perm)
    return tuple(perms)


def format_literal(perms: tuple[tuple[int, ...], ...]) -> str:
    lines = ["MOVE_PERM_2X2 = ("]
    for move_idx, perm in enumerate(perms):
        row = ", ".join(f"{x:2d}" for x in perm)
        lines.append(f"    ({row}),  # {move_idx:2d}: {MOVE_STRINGS[move_idx]}")
    lines.append(")")
    return "\n".join(lines)


# Quick correctness check: applying perm to SOLVED via gather must equal the
# oracle's render. Algebraically redundant with the test but cheap to keep.
def _self_check(perms: tuple[tuple[int, ...], ...]) -> None:
    from rubik.oracle.cubie import cubie_to_tensor

    solved_t = cubie_to_tensor(SOLVED, CUBE_2X2)
    for move_idx, perm in enumerate(perms):
        moved_t = cubie_to_tensor(apply_move(SOLVED, move_idx), CUBE_2X2)
        gathered = solved_t[torch.tensor(perm)]
        assert torch.equal(gathered, moved_t), (
            f"move {move_idx}: gather(solved, perm) != oracle render"
        )


def main() -> None:
    perms = build_move_perm()
    _self_check(perms)
    print(format_literal(perms))


if __name__ == "__main__":
    main()
