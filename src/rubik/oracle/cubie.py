"""Slow, hand-rolled cubie oracle for the 2x2 cube.

This is one of the two correctness witnesses defined in `SPEC.md`: the
unmistakably-correct, human-readable ground truth that M2's fast tensor
cube is verified against. Optimized for clarity, not speed.

State representation: corners as `(positions, orientations)` 8-tuples where
`positions[slot]` is the cubie identity currently at that slot, and
`orientations[slot]` is the corresponding twist (0/1/2). Slots are
bit-packed `(U/D, L/R, F/B)` — slot 0=ULF, 1=ULB, 2=URF, 3=URB,
4=DLF, 5=DLB, 6=DRF, 7=DRB.

Move tables are hand-written CW for the 6 faces; CCW is derived as CW^3.
The within-face sticker geometry pinned by `cubie_to_tensor` is the M1
decision that all downstream consumers (M2 tensor cube, M3 renderer)
inherit.
"""

from dataclasses import dataclass

import torch

from rubik.cube.spec import CubeSpec
from rubik.notation.moves import FACE_NAMES


@dataclass(frozen=True)
class CornerState:
    # length 8: positions[slot] = cubie identity at that slot
    positions: tuple[int, ...]
    # length 8: orientations[slot] = twist of cubie at that slot (0/1/2)
    orientations: tuple[int, ...]


SOLVED = CornerState(
    positions=(0, 1, 2, 3, 4, 5, 6, 7),
    orientations=(0,) * 8,
)


# CW move tables. Each entry is (slot_before, slot_after, orientation_delta)
# for one of the 4 affected corners. The move "moves the cubie at
# slot_before to slot_after, bumping its orientation by orientation_delta
# (mod 3)".
#
# Position cycles (verified by hand against the slot/face geometry):
#   U: ULF→ULB→URB→URF→ULF, i.e. 0→1→3→2→0 (CW from above)
#   D: DLF→DRF→DRB→DLB→DLF, i.e. 4→6→7→5→4 (CW from below)
#   R: URF→DRF→DRB→URB→URF, i.e. 2→6→7→3→2 (CW from right)
#   L: ULB→ULF→DLF→DLB→ULB, i.e. 1→0→4→5→1 (CW from left)
#   F: ULF→URF→DRF→DLF→ULF, i.e. 0→2→6→4→0 (CW from front)
#   B: URB→ULB→DLB→DRB→URB, i.e. 3→1→5→7→3 (CW from back)
#
# Orientation deltas: U/D moves rotate around the U/D axis so they preserve
# orientation. The 4 side moves alternate +1/+2 around their cycle (must
# sum to 0 mod 3). The (1, 2, 1, 2) assignment around each side-face cycle
# matches the SLOT_FACETS canonical labeling — verified by Sune^6 = I,
# (R U R' U')^6 = I, and the 8-stickers-changed-per-move sanity check.
_CW_MOVES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "U": ((0, 1, 0), (1, 3, 0), (3, 2, 0), (2, 0, 0)),
    "L": ((1, 0, 1), (0, 4, 2), (4, 5, 1), (5, 1, 2)),
    "F": ((0, 2, 1), (2, 6, 2), (6, 4, 1), (4, 0, 2)),
    "R": ((2, 6, 1), (6, 7, 2), (7, 3, 1), (3, 2, 2)),
    "B": ((3, 1, 1), (1, 5, 2), (5, 7, 1), (7, 3, 2)),
    "D": ((4, 6, 0), (6, 7, 0), (7, 5, 0), (5, 4, 0)),
}


# Module-import validation: catches typos in the move tables.
for _face, _cycle in _CW_MOVES.items():
    _befores = {entry[0] for entry in _cycle}
    _afters = {entry[1] for entry in _cycle}
    assert len(_befores) == 4 and _befores == _afters, (
        f"{_face}: cycle is not a permutation"
    )
    assert sum(entry[2] for entry in _cycle) % 3 == 0, (
        f"{_face}: orientation deltas don't sum to 0 mod 3"
    )


def _apply_cw(state: CornerState, face: str) -> CornerState:
    new_positions = list(state.positions)
    new_orientations = list(state.orientations)
    for slot_before, slot_after, ori_delta in _CW_MOVES[face]:
        cubie = state.positions[slot_before]
        ori = state.orientations[slot_before]
        new_positions[slot_after] = cubie
        new_orientations[slot_after] = (ori + ori_delta) % 3
    return CornerState(tuple(new_positions), tuple(new_orientations))


def _build_move_tables() -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Build a length-12 move-table array.

    Index follows the M0 convention `move_idx = face_idx * 2 + direction`
    where face order is FACE_NAMES and direction 0=CW, 1=CCW. CCW is
    derived as CW^3.
    """
    tables: list[tuple[tuple[int, int, int], ...]] = []
    for face in FACE_NAMES:
        cw = _CW_MOVES[face]
        tables.append(cw)
        # Build CCW by composing CW thrice from SOLVED.
        s = SOLVED
        for _ in range(3):
            s = _apply_cw(s, face)
        # Diff against SOLVED: any slot whose position changed gives us a
        # (slot_before, slot_after) edge; orientation_delta is the twist.
        # cubie identity `c` was at slot `c` originally and is now at slot
        # s.positions.index(c). We need slot_before = original position of
        # the cubie now sitting at slot_after.
        ccw_entries: list[tuple[int, int, int]] = []
        for slot_after in range(8):
            cubie = s.positions[slot_after]
            slot_before = cubie  # cubie identity == original slot
            if slot_before != slot_after:
                ori_delta = s.orientations[slot_after]
                ccw_entries.append((slot_before, slot_after, ori_delta))
        tables.append(tuple(ccw_entries))
    return tuple(tables)


_MOVE_TABLES: tuple[tuple[tuple[int, int, int], ...], ...] = _build_move_tables()


def apply_move(state: CornerState, move_idx: int) -> CornerState:
    """Apply the given QTM move to `state`, returning a new `CornerState`."""
    if not 0 <= move_idx < 12:
        raise ValueError(f"move index out of range: {move_idx}")
    new_positions = list(state.positions)
    new_orientations = list(state.orientations)
    for slot_before, slot_after, ori_delta in _MOVE_TABLES[move_idx]:
        cubie = state.positions[slot_before]
        ori = state.orientations[slot_before]
        new_positions[slot_after] = cubie
        new_orientations[slot_after] = (ori + ori_delta) % 3
    return CornerState(tuple(new_positions), tuple(new_orientations))


# Within-face sticker geometry (THE M1 DECISION).
#
# For each face, view from outside the cube with the per-face "up" axis as
# specified in plans/m1-cubie-oracle.md. Sticker positions 0,1,2,3 are
# top-left, top-right, bottom-left, bottom-right. Each entry below is the
# corner slot at (TL, TR, BL, BR) for that face.
FACE_SLOTS: dict[str, tuple[int, int, int, int]] = {
    "U": (1, 3, 0, 2),
    "L": (1, 0, 5, 4),
    "F": (0, 2, 4, 6),
    "R": (2, 3, 6, 7),
    "B": (3, 1, 7, 5),
    "D": (4, 6, 5, 7),
}


# Canonical facet order at each corner slot. Position 0 is always the
# U/D-axis facet (orientation 0 means the cubie's U/D-color sits there).
# Positions 1 and 2 are the two side facets, ordered so that orientation
# acts as a clean cyclic shift across all moves: with `(k - ori) % 3`
# indexing, orientation deltas of (1, 2, 1, 2) around each side-face cycle
# match the rendering. This specific labeling was derived empirically by
# searching for a SLOT_FACETS / orientation-delta combination that
# satisfies the algebraic identities AND renders single moves to exactly
# 8 changed stickers — equivalent to the standard speedcubing convention
# up to the choice of "1st side facet" at each slot.
SLOT_FACETS: tuple[tuple[str, str, str], ...] = (
    ("U", "F", "L"),  # 0: ULF
    ("U", "L", "B"),  # 1: ULB
    ("U", "R", "F"),  # 2: URF
    ("U", "B", "R"),  # 3: URB
    ("D", "L", "F"),  # 4: DLF
    ("D", "B", "L"),  # 5: DLB
    ("D", "F", "R"),  # 6: DRF
    ("D", "R", "B"),  # 7: DRB
)


def _build_corner_colors() -> tuple[tuple[int, int, int], ...]:
    # Color = face index per FACE_NAMES = ("U","L","F","R","B","D").
    face_color = {face: i for i, face in enumerate(FACE_NAMES)}
    return tuple(tuple(face_color[f] for f in SLOT_FACETS[c]) for c in range(8))


# CORNER_COLORS[cubie_id] = the cubie's 3 colors in canonical facet order.
CORNER_COLORS: tuple[tuple[int, int, int], ...] = _build_corner_colors()


def cubie_to_tensor(state: CornerState, spec: CubeSpec) -> torch.Tensor:
    """Render a CornerState into a flat sticker tensor of shape (n_stickers,).

    Sticker layout matches `state_to_dict`: face blocks in `spec.faces`
    order, each block of length `spec.stickers_per_face` in the
    within-face geometric order pinned by `FACE_SLOTS`.
    """
    if spec.size != 2:
        raise NotImplementedError("M1 oracle is 2x2 only; M5 will extend to 3x3.")
    tensor = torch.zeros(spec.n_stickers, dtype=torch.int8)
    for face_idx, face in enumerate(FACE_NAMES):
        for within_idx, slot in enumerate(FACE_SLOTS[face]):
            cubie_id = state.positions[slot]
            ori = state.orientations[slot]
            # Facet position k at this slot for this face direction.
            k = SLOT_FACETS[slot].index(face)
            color = CORNER_COLORS[cubie_id][(k - ori) % 3]
            sticker_idx = face_idx * spec.stickers_per_face + within_idx
            tensor[sticker_idx] = color
    return tensor
