"""Slow, hand-rolled cubie oracle for the 2x2 and 3x3 cubes.

This is one of the two correctness witnesses defined in `SPEC.md`: the
unmistakably-correct, human-readable ground truth that M2's fast tensor
cube is verified against. Optimized for clarity, not speed.

State representations:
- 2x2: corners as `(positions, orientations)` 8-tuples where
  `positions[slot]` is the cubie identity currently at that slot, and
  `orientations[slot]` is the corresponding twist (0/1/2). Slots are
  bit-packed `(U/D, L/R, F/B)` — slot 0=ULF, 1=ULB, 2=URF, 3=URB,
  4=DLF, 5=DLB, 6=DRF, 7=DRB.
- 3x3: same 8 corners, plus 12 edges with 2 orientations each (U/D-axis
  flip convention — F/B turns flip, U/D/L/R don't), plus 6 centers which
  are positionally fixed under QTM face turns (no whole-cube rotations).

Move tables are hand-written CW for the 6 faces; CCW is derived as CW^3.
The within-face sticker geometry pinned by `cubie_to_tensor` /
`cubie_to_tensor_3x3` is the M1 / M8 decision that all downstream
consumers (tensor cube, renderer) inherit.
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
# Position cycles (CW viewed from outside the moved face — verified
# physically and by `test_cw_direction`):
#   U: ULF→ULB→URB→URF→ULF, i.e. 0→1→3→2→0 (CW from above)
#   D: DLF→DRF→DRB→DLB→DLF, i.e. 4→6→7→5→4 (CW from below)
#   R: URF→URB→DRB→DRF→URF, i.e. 2→3→7→6→2 (CW from right)
#   L: ULB→ULF→DLF→DLB→ULB, i.e. 1→0→4→5→1 (CW from left)
#   F: ULF→URF→DRF→DLF→ULF, i.e. 0→2→6→4→0 (CW from front)
#   B: URB→ULB→DLB→DRB→URB, i.e. 3→1→5→7→3 (CW from back)
#
# Orientation deltas: U/D moves rotate around the U/D axis so they preserve
# orientation. The 4 side moves alternate +1/+2 around their cycle (must
# sum to 0 mod 3). The (1, 2, 1, 2) assignment around each side-face cycle
# matches the SLOT_FACETS canonical labeling — verified by Sune^6 = I,
# (R U R' U')^6 = I, and the per-move directional checks.
_CW_MOVES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "U": ((0, 1, 0), (1, 3, 0), (3, 2, 0), (2, 0, 0)),
    "L": ((1, 0, 1), (0, 4, 2), (4, 5, 1), (5, 1, 2)),
    "F": ((0, 2, 1), (2, 6, 2), (6, 4, 1), (4, 0, 2)),
    "R": ((2, 3, 1), (3, 7, 2), (7, 6, 1), (6, 2, 2)),
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
        raise NotImplementedError(
            "cubie_to_tensor is 2x2 only; use cubie_to_tensor_3x3 for 3x3."
        )
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


# ============================================================================
# 3x3 oracle — corners (reused) + 12 edges + 6 (fixed) centers.
# ============================================================================


@dataclass(frozen=True)
class EdgeState:
    # length 12: positions[slot] = edge cubie identity at that slot
    positions: tuple[int, ...]
    # length 12: orientations[slot] = flip of edge at that slot (0 or 1)
    orientations: tuple[int, ...]


@dataclass(frozen=True)
class Cube3x3State:
    """Composite 3x3 state: corners + edges. Centers are positionally
    fixed under QTM face turns and are encoded directly in
    `cubie_to_tensor_3x3` from face index — no state needed."""

    corners: CornerState
    edges: EdgeState


# Edge slot indexing (0..11):
#   0=UF, 1=UR, 2=UB, 3=UL    (top layer, CW from front when viewed from above)
#   4=DF, 5=DR, 6=DB, 7=DL    (bottom layer, same CW order)
#   8=FR, 9=BR, 10=BL, 11=FL  (middle slice — front-right, back-right, etc.)
#
# Edge orientation convention (U/D-axis flip):
#   Each edge has 2 facets. A "primary" facet is picked per slot:
#     - top/bottom edges (UF, UR, UB, UL, DF, DR, DB, DL): primary = U or D
#     - middle slice edges (FR, BR, BL, FL): primary = F or B
#   Orientation 0 = primary facet sits at the slot's primary position.
#   Orientation 1 = flipped.
#   Under U/U'/D/D'/L/L'/R/R': orientation unchanged.
#   Under F/F'/B/B': orientation flipped (xor 1) for all 4 affected edges.

# CW edge cycles per face. (slot_before, slot_after, ori_delta).
# Position cycles (CW viewed from outside the moved face):
#   U: UF→UL→UB→UR→UF, i.e. 0→3→2→1→0
#   D: DF→DR→DB→DL→DF, i.e. 4→5→6→7→4
#   R: UR→BR→DR→FR→UR, i.e. 1→9→5→8→1
#   L: UL→FL→DL→BL→UL, i.e. 3→11→7→10→3
#   F: UF→FR→DF→FL→UF, i.e. 0→8→4→11→0  (orient flipped)
#   B: UB→BL→DB→BR→UB, i.e. 2→10→6→9→2  (orient flipped)
_EDGE_CW_MOVES_3X3: dict[str, tuple[tuple[int, int, int], ...]] = {
    "U": ((0, 3, 0), (3, 2, 0), (2, 1, 0), (1, 0, 0)),
    "L": ((3, 11, 0), (11, 7, 0), (7, 10, 0), (10, 3, 0)),
    "F": ((0, 8, 1), (8, 4, 1), (4, 11, 1), (11, 0, 1)),
    "R": ((1, 9, 0), (9, 5, 0), (5, 8, 0), (8, 1, 0)),
    "B": ((2, 10, 1), (10, 6, 1), (6, 9, 1), (9, 2, 1)),
    "D": ((4, 5, 0), (5, 6, 0), (6, 7, 0), (7, 4, 0)),
}


# Module-import validation: catches typos in the edge move tables.
for _face, _cycle in _EDGE_CW_MOVES_3X3.items():
    _befores = {entry[0] for entry in _cycle}
    _afters = {entry[1] for entry in _cycle}
    assert len(_befores) == 4 and _befores == _afters, (
        f"{_face}: edge cycle is not a permutation"
    )
    # In the U/D-axis convention, deltas sum to 0 mod 2 trivially since
    # each face's deltas are all 0 or all 1.
    assert all(entry[2] in (0, 1) for entry in _cycle), (
        f"{_face}: edge orientation delta out of range"
    )


SOLVED_3X3 = Cube3x3State(
    corners=SOLVED,
    edges=EdgeState(
        positions=tuple(range(12)),
        orientations=(0,) * 12,
    ),
)


def _apply_cw_edges(state: EdgeState, face: str) -> EdgeState:
    new_positions = list(state.positions)
    new_orientations = list(state.orientations)
    for slot_before, slot_after, ori_delta in _EDGE_CW_MOVES_3X3[face]:
        cubie = state.positions[slot_before]
        ori = state.orientations[slot_before]
        new_positions[slot_after] = cubie
        new_orientations[slot_after] = (ori + ori_delta) % 2
    return EdgeState(tuple(new_positions), tuple(new_orientations))


def _build_edge_move_tables() -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Build length-12 edge move-table array (parallel to corners)."""
    tables: list[tuple[tuple[int, int, int], ...]] = []
    solved_edges = SOLVED_3X3.edges
    for face in FACE_NAMES:
        cw = _EDGE_CW_MOVES_3X3[face]
        tables.append(cw)
        # CCW = CW^3 derived from SOLVED.
        s = solved_edges
        for _ in range(3):
            s = _apply_cw_edges(s, face)
        ccw_entries: list[tuple[int, int, int]] = []
        for slot_after in range(12):
            cubie = s.positions[slot_after]
            slot_before = cubie  # cubie identity == original slot
            if slot_before != slot_after:
                ori_delta = s.orientations[slot_after]
                ccw_entries.append((slot_before, slot_after, ori_delta))
        tables.append(tuple(ccw_entries))
    return tuple(tables)


_EDGE_MOVE_TABLES_3X3: tuple[tuple[tuple[int, int, int], ...], ...] = (
    _build_edge_move_tables()
)


def _apply_edge_move(state: EdgeState, move_idx: int) -> EdgeState:
    new_positions = list(state.positions)
    new_orientations = list(state.orientations)
    for slot_before, slot_after, ori_delta in _EDGE_MOVE_TABLES_3X3[move_idx]:
        cubie = state.positions[slot_before]
        ori = state.orientations[slot_before]
        new_positions[slot_after] = cubie
        new_orientations[slot_after] = (ori + ori_delta) % 2
    return EdgeState(tuple(new_positions), tuple(new_orientations))


def apply_move_3x3(state: Cube3x3State, move_idx: int) -> Cube3x3State:
    """Apply the given QTM move to a 3x3 state, returning a new state.

    Centers don't permute under face turns, so they aren't tracked.
    """
    if not 0 <= move_idx < 12:
        raise ValueError(f"move index out of range: {move_idx}")
    return Cube3x3State(
        corners=apply_move(state.corners, move_idx),
        edges=_apply_edge_move(state.edges, move_idx),
    )


# Edge slots adjacent to each face (4 per face), ordered by within-face
# sticker position (T, L, R, B) — corresponds to within-face indices
# (1, 3, 5, 7) on the row-major 3x3 grid (idx 4 is the center).
#
# Hand-derived from the per-face viewing orientation pinned by 2x2
# FACE_SLOTS:
#   U: viewed from above, top row = back row of cube. T=UB, L=UL, R=UR, B=UF.
#   L: viewed from left, top=U, B is on viewer's left. T=UL, L=BL, R=FL, B=DL.
#   F: viewed from front, top=U, L on viewer's left.   T=UF, L=FL, R=FR, B=DF.
#   R: viewed from right, top=U, F on viewer's left.   T=UR, L=FR, R=BR, B=DR.
#   B: viewed from behind, top=U, R on viewer's left.  T=UB, L=BR, R=BL, B=DB.
#   D: viewed from below, top row = front row of cube. T=DF, L=DL, R=DR, B=DB.
EDGE_SLOTS_3X3: dict[str, tuple[int, int, int, int]] = {
    "U": (2, 3, 1, 0),  # T=UB, L=UL, R=UR, B=UF
    "L": (3, 10, 11, 7),  # T=UL, L=BL, R=FL, B=DL
    "F": (0, 11, 8, 4),  # T=UF, L=FL, R=FR, B=DF
    "R": (1, 8, 9, 5),  # T=UR, L=FR, R=BR, B=DR
    "B": (2, 9, 10, 6),  # T=UB, L=BR, R=BL, B=DB
    "D": (4, 7, 5, 6),  # T=DF, L=DL, R=DR, B=DB
}


# Canonical facet order at each edge slot: (primary, secondary).
# orientation 0: cubie's primary color sits at the slot's primary face.
# orientation 1: flipped.
# Cubie ID == home slot, so EDGE_SLOT_FACETS doubles as the per-cubie
# color tuple.
EDGE_SLOT_FACETS: tuple[tuple[str, str], ...] = (
    ("U", "F"),  # 0: UF
    ("U", "R"),  # 1: UR
    ("U", "B"),  # 2: UB
    ("U", "L"),  # 3: UL
    ("D", "F"),  # 4: DF
    ("D", "R"),  # 5: DR
    ("D", "B"),  # 6: DB
    ("D", "L"),  # 7: DL
    ("F", "R"),  # 8: FR
    ("B", "R"),  # 9: BR
    ("B", "L"),  # 10: BL
    ("F", "L"),  # 11: FL
)


def _build_edge_colors() -> tuple[tuple[int, int], ...]:
    face_color = {face: i for i, face in enumerate(FACE_NAMES)}
    return tuple(
        (face_color[primary], face_color[secondary])
        for primary, secondary in EDGE_SLOT_FACETS
    )


# EDGE_COLORS_3X3[cubie_id] = (primary_color, secondary_color).
EDGE_COLORS_3X3: tuple[tuple[int, int], ...] = _build_edge_colors()


# Within-face indices on a row-major 3x3 face (0..8):
#   0 1 2
#   3 4 5
#   6 7 8
# Corners at 0/2/6/8, edges at 1/3/5/7, center at 4.
_CORNER_INDICES_3X3: tuple[int, int, int, int] = (0, 2, 6, 8)
_EDGE_INDICES_3X3: tuple[int, int, int, int] = (1, 3, 5, 7)
_CENTER_INDEX_3X3: int = 4


def cubie_to_tensor_3x3(state: Cube3x3State, spec: CubeSpec) -> torch.Tensor:
    """Render a Cube3x3State into a flat sticker tensor of shape (54,).

    Sticker layout: face blocks in `spec.faces` order, each block of
    length 9 in row-major within-face order. Per-face viewing orientation
    matches `FACE_SLOTS` (re-used for corner placement).
    """
    if spec.size != 3:
        raise ValueError(
            f"cubie_to_tensor_3x3 requires a 3x3 spec, got size={spec.size}"
        )
    tensor = torch.zeros(spec.n_stickers, dtype=torch.int8)
    for face_idx, face in enumerate(FACE_NAMES):
        face_base = face_idx * spec.stickers_per_face
        # Center sticker — fixed color = face_idx (no permutation under QTM).
        tensor[face_base + _CENTER_INDEX_3X3] = face_idx
        # Corner stickers (4 per face): same FACE_SLOTS / SLOT_FACETS
        # mapping as the 2x2 oracle.
        for corner_pos, slot in enumerate(FACE_SLOTS[face]):
            cubie_id = state.corners.positions[slot]
            ori = state.corners.orientations[slot]
            k = SLOT_FACETS[slot].index(face)
            color = CORNER_COLORS[cubie_id][(k - ori) % 3]
            within_idx = _CORNER_INDICES_3X3[corner_pos]
            tensor[face_base + within_idx] = color
        # Edge stickers (4 per face).
        for edge_pos, slot in enumerate(EDGE_SLOTS_3X3[face]):
            cubie_id = state.edges.positions[slot]
            ori = state.edges.orientations[slot]
            k = EDGE_SLOT_FACETS[slot].index(face)
            color = EDGE_COLORS_3X3[cubie_id][(k - ori) % 2]
            within_idx = _EDGE_INDICES_3X3[edge_pos]
            tensor[face_base + within_idx] = color
    return tensor
