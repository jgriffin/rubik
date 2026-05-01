"""QTM move notation.

12 quarter-turn moves indexed `face_idx * 2 + direction` where direction 0 is
CW and 1 is CCW (prime'). This shape makes same-face pruning a single
`move >> 1` later. Independent of `CubeSpec` — the QTM mapping is identical
for any cube size.
"""

FACE_NAMES: tuple[str, ...] = ("U", "L", "F", "R", "B", "D")

MOVE_STRINGS: tuple[str, ...] = tuple(
    f"{face}{suffix}" for face in FACE_NAMES for suffix in ("", "'")
)

_STR_TO_MOVE: dict[str, int] = {s: i for i, s in enumerate(MOVE_STRINGS)}


def move_to_str(idx: int) -> str:
    if not 0 <= idx < len(MOVE_STRINGS):
        raise ValueError(f"move index out of range: {idx}")
    return MOVE_STRINGS[idx]


def str_to_move(s: str) -> int:
    if s not in _STR_TO_MOVE:
        raise ValueError(f"unknown move string: {s!r}")
    return _STR_TO_MOVE[s]


def move_to_tuple(idx: int) -> tuple[int, int]:
    if not 0 <= idx < len(MOVE_STRINGS):
        raise ValueError(f"move index out of range: {idx}")
    return idx // 2, idx % 2


def tuple_to_move(face_idx: int, direction: int) -> int:
    if not 0 <= face_idx < len(FACE_NAMES):
        raise ValueError(f"face index out of range: {face_idx}")
    if direction not in (0, 1):
        raise ValueError(f"direction must be 0 or 1: {direction}")
    return face_idx * 2 + direction
