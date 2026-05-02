"""ASCII renderer for the unfolded-cross cube layout.

Consumes the face-dict notation form (``dict[str, list[int]]``) produced by
``rubik.notation.state.state_to_dict`` — never a raw tensor. Each face is
laid out row-major TL/TR/BL/BR per the M1 ``FACE_SLOTS`` contract; the
renderer just reshapes each face's flat block into an ``size x size`` grid.

The cross shape (parametric on ``spec.size``):

::

         U
       L F R B
         D

For a 2x2 cube each face is 2 rows by 2 cols of stickers; total height is
``3 * size`` rows. Cells are single characters — the face letter
corresponding to the sticker's color index — separated by single spaces
within a row. The U and D rows are indented by ``2 * size`` spaces (the
width of the L face plus its trailing inter-face separator) so their
columns line up with the F face below.
"""

from rubik.cube.spec import CubeSpec


def render_ascii(face_dict: dict[str, list[int]], spec: CubeSpec) -> str:
    """Render a face-dict as an unfolded-cross ASCII string (no trailing newline)."""
    if set(face_dict.keys()) != set(spec.faces):
        raise ValueError(
            f"face_dict faces {sorted(face_dict.keys())} do not match"
            f" spec.faces {list(spec.faces)}"
        )
    n = spec.size
    expected_per_face = spec.stickers_per_face
    for face in spec.faces:
        if len(face_dict[face]) != expected_per_face:
            raise ValueError(
                f"face {face!r} has {len(face_dict[face])} stickers,"
                f" expected {expected_per_face}"
            )

    def row_of(face: str, r: int) -> list[str]:
        # FACE_NAMES is the ordering. spec.faces[idx] is the letter for that color.
        block = face_dict[face]
        return [spec.faces[block[r * n + c]] for c in range(n)]

    # 2 leading spaces per "skipped" face cell (1 sticker char + 1 space).
    indent = " " * (2 * n)

    lines: list[str] = []
    # Top band — U.
    for r in range(n):
        cells = row_of("U", r)
        lines.append(indent + " ".join(cells))
    # Middle band — L F R B.
    for r in range(n):
        cells: list[str] = []
        for face in ("L", "F", "R", "B"):
            cells.extend(row_of(face, r))
        lines.append(" ".join(cells))
    # Bottom band — D.
    for r in range(n):
        cells = row_of("D", r)
        lines.append(indent + " ".join(cells))

    return "\n".join(lines)
