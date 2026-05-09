"""Kociemba facelet-string <-> internal-state converter.

Internal state is `(spec.n_stickers,) int8` with face order `spec.faces`
(`(U, L, F, R, B, D)` for both 2x2 and 3x3) and row-major top-left ->
bottom-right ordering within each face. Each int names the face the
sticker's color belongs to: 0=U, 1=L, 2=F, 3=R, 4=B, 5=D — i.e. the
letter is `spec.faces[color_int]`. The color-letter map therefore uses
internal face order, not kociemba's.

Kociemba's facelet string (https://kociemba.org/cube.htm) is `URFDLB`
face order, row-major within each face, characters `U/R/F/D/L/B` for the
face that each sticker's color belongs to. Within-face ordering is
identical to ours, so conversion is a pure face-block permutation.

The permutation is computed at call time from `spec.faces` against the
module-level `_KOCIEMBA_FACE_ORDER` so the same code generalizes across
`CubeSpec` sizes (2x2 -> 24 chars, 3x3 -> 54 chars).
"""

import torch

from rubik.cube.spec import CubeSpec

_KOCIEMBA_FACE_ORDER: tuple[str, ...] = ("U", "R", "F", "D", "L", "B")


def _internal_to_kociemba_perm(spec: CubeSpec) -> list[int]:
    """For each kociemba face slot, the internal face index that fills it."""
    internal_idx = {face: i for i, face in enumerate(spec.faces)}
    return [internal_idx[face] for face in _KOCIEMBA_FACE_ORDER]


def state_to_facelet(state: torch.Tensor, spec: CubeSpec) -> str:
    if state.shape != (spec.n_stickers,):
        raise ValueError(
            f"state shape {tuple(state.shape)} does not match spec.n_stickers"
            f" ({spec.n_stickers},)"
        )
    n = spec.stickers_per_face
    ints = state.tolist()
    chars = [spec.faces[c] for c in ints]
    perm = _internal_to_kociemba_perm(spec)
    out: list[str] = []
    for src in perm:
        start = src * n
        out.extend(chars[start : start + n])
    return "".join(out)


def facelet_to_state(facelet: str, spec: CubeSpec) -> torch.Tensor:
    if len(facelet) != spec.n_stickers:
        raise ValueError(
            f"facelet length {len(facelet)} does not match spec.n_stickers"
            f" ({spec.n_stickers})"
        )
    letter_to_int = {face: i for i, face in enumerate(spec.faces)}
    unknown = set(facelet) - set(letter_to_int)
    if unknown:
        raise ValueError(
            f"facelet has unknown characters {sorted(unknown)};"
            f" expected one of {list(spec.faces)}"
        )
    n = spec.stickers_per_face
    perm = _internal_to_kociemba_perm(spec)
    # `perm[k]` is the internal face index for kociemba slot k. Invert to
    # know, for each internal face, which kociemba slot it came from.
    kociemba_slot_for_internal = [0] * len(spec.faces)
    for kociemba_slot, internal_face_idx in enumerate(perm):
        kociemba_slot_for_internal[internal_face_idx] = kociemba_slot
    flat: list[int] = []
    for internal_face_idx in range(len(spec.faces)):
        ks = kociemba_slot_for_internal[internal_face_idx]
        block = facelet[ks * n : (ks + 1) * n]
        flat.extend(letter_to_int[c] for c in block)
    return torch.tensor(flat, dtype=torch.int8)
