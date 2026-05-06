"""Generate the MOVE_PERM_3X3 sticker-permutation snapshot from the M8 oracle.

For each of the 12 QTM moves, derives the (54,) sticker permutation
`perm[i] = source_position_of_sticker_i_after_move` such that
`s_new[i] = s[perm[i]]`. Method: build a "fingerprint" rendering of the
solved state where each sticker carries its own index 0..53 (corner /
edge / center facets all distinguished), apply the move via the oracle,
render again, and read off the permutation.

Prints the (12, 54) tuple-of-tuples as a Python literal for paste into
`src/rubik/cube/env.py`. Re-run whenever the oracle's sticker geometry
changes (drift caught in CI by `test_move_perm_3x3_matches_oracle`).

Run:
    uv run python scripts/move_perm_from_oracle_3x3.py
"""

import torch

from rubik.cube.spec import CUBE_3X3
from rubik.notation.moves import FACE_NAMES, MOVE_STRINGS
from rubik.oracle.cubie import (
    _CENTER_INDEX_3X3,
    _CORNER_INDICES_3X3,
    _EDGE_INDICES_3X3,
    EDGE_SLOT_FACETS,
    EDGE_SLOTS_3X3,
    FACE_SLOTS,
    SLOT_FACETS,
    SOLVED_3X3,
    apply_move_3x3,
    cubie_to_tensor_3x3,
)


def _solved_sticker_index(face_idx: int, within_idx: int) -> int:
    return face_idx * CUBE_3X3.stickers_per_face + within_idx


def _cubie_to_position_tensor_3x3(state) -> torch.Tensor:
    """Render a Cube3x3State carrying sticker-position indices, not colors.

    Mirrors `cubie_to_tensor_3x3` exactly but emits the SOLVED sticker
    index of each facet rather than its color. The result is a `(54,)`
    int64 tensor where `out[i]` = the sticker index that sits at slot `i`
    after the move, i.e. the permutation source-index for slot `i`.
    """
    # (kind, cubie_id, facet_position_k) -> SOLVED sticker index. Built
    # once from the SOLVED render: walk every (face, within-face slot)
    # and record which (cubie_id, k) produced its color. Centers are
    # keyed only by face (no cubie/orientation degree of freedom).
    corner_facet_to_sticker: dict[tuple[int, int], int] = {}
    edge_facet_to_sticker: dict[tuple[int, int], int] = {}
    center_face_to_sticker: dict[int, int] = {}

    for face_idx, face in enumerate(FACE_NAMES):
        face_base = face_idx * CUBE_3X3.stickers_per_face
        # Center: fixed at within-face index 4.
        center_face_to_sticker[face_idx] = face_base + _CENTER_INDEX_3X3
        # Corners: 4 per face at within-face indices (0, 2, 6, 8).
        for corner_pos, slot in enumerate(FACE_SLOTS[face]):
            # In SOLVED, ori=0 so k = SLOT_FACETS[slot].index(face).
            k = SLOT_FACETS[slot].index(face)
            cubie_id = slot  # SOLVED: cubie_id == slot
            within_idx = _CORNER_INDICES_3X3[corner_pos]
            corner_facet_to_sticker[(cubie_id, k)] = face_base + within_idx
        # Edges: 4 per face at within-face indices (1, 3, 5, 7).
        for edge_pos, slot in enumerate(EDGE_SLOTS_3X3[face]):
            k = EDGE_SLOT_FACETS[slot].index(face)
            cubie_id = slot  # SOLVED: cubie_id == slot
            within_idx = _EDGE_INDICES_3X3[edge_pos]
            edge_facet_to_sticker[(cubie_id, k)] = face_base + within_idx

    tensor = torch.zeros(CUBE_3X3.n_stickers, dtype=torch.int64)
    for face_idx, face in enumerate(FACE_NAMES):
        face_base = face_idx * CUBE_3X3.stickers_per_face
        # Center: source = same center sticker (face-fixed under QTM).
        tensor[face_base + _CENTER_INDEX_3X3] = center_face_to_sticker[face_idx]
        # Corner facets.
        for corner_pos, slot in enumerate(FACE_SLOTS[face]):
            cubie_id = state.corners.positions[slot]
            ori = state.corners.orientations[slot]
            k = (SLOT_FACETS[slot].index(face) - ori) % 3
            within_idx = _CORNER_INDICES_3X3[corner_pos]
            tensor[face_base + within_idx] = corner_facet_to_sticker[(cubie_id, k)]
        # Edge facets.
        for edge_pos, slot in enumerate(EDGE_SLOTS_3X3[face]):
            cubie_id = state.edges.positions[slot]
            ori = state.edges.orientations[slot]
            k = (EDGE_SLOT_FACETS[slot].index(face) - ori) % 2
            within_idx = _EDGE_INDICES_3X3[edge_pos]
            tensor[face_base + within_idx] = edge_facet_to_sticker[(cubie_id, k)]
    return tensor


def build_move_perm() -> tuple[tuple[int, ...], ...]:
    perms: list[tuple[int, ...]] = []
    for move_idx in range(CUBE_3X3.n_moves):
        moved = apply_move_3x3(SOLVED_3X3, move_idx)
        perm = tuple(_cubie_to_position_tensor_3x3(moved).tolist())
        # Sanity: must be a permutation of range(54).
        assert len(perm) == CUBE_3X3.n_stickers, (
            f"move {move_idx}: wrong length {len(perm)}"
        )
        assert set(perm) == set(range(CUBE_3X3.n_stickers)), (
            f"move {move_idx}: not a permutation of 0..53"
        )
        # Centers are fixed under any QTM face turn.
        for face_idx in range(CUBE_3X3.n_faces):
            center_idx = face_idx * CUBE_3X3.stickers_per_face + _CENTER_INDEX_3X3
            assert perm[center_idx] == center_idx, (
                f"move {move_idx}: center at face {face_idx} not fixed"
            )
        perms.append(perm)
    return tuple(perms)


def format_literal(perms: tuple[tuple[int, ...], ...]) -> str:
    lines = ["MOVE_PERM_3X3 = ("]
    for move_idx, perm in enumerate(perms):
        row = ", ".join(f"{x:2d}" for x in perm)
        lines.append(f"    ({row}),  # {move_idx:2d}: {MOVE_STRINGS[move_idx]}")
    lines.append(")")
    return "\n".join(lines)


# Quick correctness check: applying perm to SOLVED via gather must equal the
# oracle's render. Algebraically redundant with the test but cheap to keep.
def _self_check(perms: tuple[tuple[int, ...], ...]) -> None:
    solved_t = cubie_to_tensor_3x3(SOLVED_3X3, CUBE_3X3)
    for move_idx, perm in enumerate(perms):
        moved_t = cubie_to_tensor_3x3(apply_move_3x3(SOLVED_3X3, move_idx), CUBE_3X3)
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
