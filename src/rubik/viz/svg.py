"""SVG renderers for cube states.

Three functions, each returning an SVG fragment string (no XML preamble) so
callers can splice the result directly into an HTML document.

- ``render_svg_state`` — single state, unfolded cross.
- ``render_svg_sequence`` — horizontal row of states with optional labels.
- ``render_svg_compare`` — two stacked sequences with optional row labels.

All consume the face-dict notation (``dict[str, list[int]]``) per
``rubik.notation.state.state_to_dict``. Geometry is parametric on
``spec.size`` and matches the M1 ``FACE_SLOTS`` row-major TL/TR/BL/BR
contract — each face's flat 4-block reshapes as ``[[TL, TR], [BL, BR]]``
with no further rotation.

Cross layout (in face-units, where each face is ``size`` x ``size`` stickers):

::

    .  U  .  .
    L  F  R  B
    .  D  .  .

For a 2x2 the sticker grid is 8 cols x 6 rows; for an N x N it is 4N x 3N.
Output is byte-stable across runs (no whitespace inside ``<svg>`` beyond
single spaces between attributes; deterministic attribute order; no
trailing newline).
"""

from collections.abc import Sequence

from rubik.cube.spec import CubeSpec
from rubik.viz.colors import FACE_COLORS

# Face position in the unfolded cross, in face-units (col, row).
_FACE_GRID: dict[str, tuple[int, int]] = {
    "U": (1, 0),
    "L": (0, 1),
    "F": (1, 1),
    "R": (2, 1),
    "B": (3, 1),
    "D": (1, 2),
}

_STROKE = "#0a0a0a"
_LABEL_FILL = "#f5f5f5"
_LABEL_FONT = "'SF Mono', ui-monospace, Menlo, monospace"
_LABEL_FONT_SIZE = 12
_LABEL_HEIGHT = 18  # vertical space reserved above each frame when labelled.


def _validate(face_dict: dict[str, list[int]], spec: CubeSpec) -> None:
    if set(face_dict.keys()) != set(spec.faces):
        raise ValueError(
            f"face_dict faces {sorted(face_dict.keys())} do not match"
            f" spec.faces {list(spec.faces)}"
        )
    for face in spec.faces:
        if len(face_dict[face]) != spec.stickers_per_face:
            raise ValueError(
                f"face {face!r} has {len(face_dict[face])} stickers,"
                f" expected {spec.stickers_per_face}"
            )


def _state_dims(spec: CubeSpec, sticker_size: int, gap: int) -> tuple[int, int]:
    n = spec.size
    cell = sticker_size + gap
    width = 4 * n * cell + gap
    height = 3 * n * cell + gap
    return width, height


def _state_rects(
    face_dict: dict[str, list[int]],
    spec: CubeSpec,
    *,
    sticker_size: int,
    gap: int,
    x_offset: int = 0,
    y_offset: int = 0,
) -> list[str]:
    """Emit one ``<rect>`` per sticker. No wrapping ``<svg>`` or ``<g>``."""
    n = spec.size
    cell = sticker_size + gap
    parts: list[str] = []
    for face in spec.faces:
        face_col, face_row = _FACE_GRID[face]
        block = face_dict[face]
        for i in range(spec.stickers_per_face):
            r = i // n
            c = i % n
            x = x_offset + (face_col * n + c) * cell + gap
            y = y_offset + (face_row * n + r) * cell + gap
            color = FACE_COLORS[block[i]]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{sticker_size}"'
                f' height="{sticker_size}" rx="1.5"'
                f' fill="{color}" stroke="{_STROKE}"/>'
            )
    return parts


def _label_text(text: str, x: int, y: int) -> str:
    # Anchor middle, baseline ~12px below the y coordinate.
    return (
        f'<text x="{x}" y="{y}"'
        f' font-family="{_LABEL_FONT}" font-size="{_LABEL_FONT_SIZE}"'
        f' fill="{_LABEL_FILL}" text-anchor="middle">{text}</text>'
    )


def render_svg_state(
    face_dict: dict[str, list[int]],
    spec: CubeSpec,
    *,
    sticker_size: int = 20,
    gap: int = 1,
) -> str:
    """Render a single cube state as an unfolded-cross SVG fragment."""
    _validate(face_dict, spec)
    width, height = _state_dims(spec, sticker_size, gap)
    rects = _state_rects(face_dict, spec, sticker_size=sticker_size, gap=gap)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}"'
        f' height="{height}" viewBox="0 0 {width} {height}">'
        f"{''.join(rects)}"
        "</svg>"
    )


def render_svg_sequence(
    face_dicts: Sequence[dict[str, list[int]]],
    spec: CubeSpec,
    *,
    labels: Sequence[str] | None = None,
    sticker_size: int = 20,
    gap: int = 1,
    frame_gap: int = 24,
) -> str:
    """Render a horizontal row of states with optional labels above each frame."""
    if labels is not None and len(labels) != len(face_dicts):
        raise ValueError(
            f"labels length {len(labels)} != face_dicts length {len(face_dicts)}"
        )
    for fd in face_dicts:
        _validate(fd, spec)

    frame_w, frame_h = _state_dims(spec, sticker_size, gap)
    label_offset = _LABEL_HEIGHT if labels is not None else 0
    total_w = (
        len(face_dicts) * frame_w + max(0, len(face_dicts) - 1) * frame_gap
        if face_dicts
        else 0
    )
    total_h = frame_h + label_offset

    parts: list[str] = []
    for i, fd in enumerate(face_dicts):
        x_offset = i * (frame_w + frame_gap)
        if labels is not None:
            label_y = _LABEL_FONT_SIZE + 2  # small top padding
            parts.append(_label_text(labels[i], x_offset + frame_w // 2, label_y))
        parts.extend(
            _state_rects(
                fd,
                spec,
                sticker_size=sticker_size,
                gap=gap,
                x_offset=x_offset,
                y_offset=label_offset,
            )
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}"'
        f' height="{total_h}" viewBox="0 0 {total_w} {total_h}">'
        f"{''.join(parts)}"
        "</svg>"
    )


def render_svg_compare(
    top_dicts: Sequence[dict[str, list[int]]],
    bottom_dicts: Sequence[dict[str, list[int]]],
    spec: CubeSpec,
    *,
    top_labels: Sequence[str] | None = None,
    bottom_labels: Sequence[str] | None = None,
    sticker_size: int = 20,
    gap: int = 1,
    frame_gap: int = 24,
    row_gap: int = 24,
) -> str:
    """Render two stacked sequences of cube states with optional per-frame labels."""
    if top_labels is not None and len(top_labels) != len(top_dicts):
        raise ValueError(
            f"top_labels length {len(top_labels)} != top_dicts length {len(top_dicts)}"
        )
    if bottom_labels is not None and len(bottom_labels) != len(bottom_dicts):
        raise ValueError(
            f"bottom_labels length {len(bottom_labels)} !="
            f" bottom_dicts length {len(bottom_dicts)}"
        )
    for fd in (*top_dicts, *bottom_dicts):
        _validate(fd, spec)

    frame_w, frame_h = _state_dims(spec, sticker_size, gap)
    top_label_offset = _LABEL_HEIGHT if top_labels is not None else 0
    bottom_label_offset = _LABEL_HEIGHT if bottom_labels is not None else 0
    top_row_h = frame_h + top_label_offset
    bottom_row_h = frame_h + bottom_label_offset

    top_count = len(top_dicts)
    bottom_count = len(bottom_dicts)
    top_w = top_count * frame_w + max(0, top_count - 1) * frame_gap if top_count else 0
    bottom_w = (
        bottom_count * frame_w + max(0, bottom_count - 1) * frame_gap
        if bottom_count
        else 0
    )
    total_w = max(top_w, bottom_w)
    total_h = top_row_h + row_gap + bottom_row_h

    parts: list[str] = []
    # Top row.
    for i, fd in enumerate(top_dicts):
        x_offset = i * (frame_w + frame_gap)
        if top_labels is not None:
            parts.append(
                _label_text(
                    top_labels[i],
                    x_offset + frame_w // 2,
                    _LABEL_FONT_SIZE + 2,
                )
            )
        parts.extend(
            _state_rects(
                fd,
                spec,
                sticker_size=sticker_size,
                gap=gap,
                x_offset=x_offset,
                y_offset=top_label_offset,
            )
        )
    # Bottom row.
    bottom_y0 = top_row_h + row_gap
    for i, fd in enumerate(bottom_dicts):
        x_offset = i * (frame_w + frame_gap)
        if bottom_labels is not None:
            parts.append(
                _label_text(
                    bottom_labels[i],
                    x_offset + frame_w // 2,
                    bottom_y0 + _LABEL_FONT_SIZE + 2,
                )
            )
        parts.extend(
            _state_rects(
                fd,
                spec,
                sticker_size=sticker_size,
                gap=gap,
                x_offset=x_offset,
                y_offset=bottom_y0 + bottom_label_offset,
            )
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}"'
        f' height="{total_h}" viewBox="0 0 {total_w} {total_h}">'
        f"{''.join(parts)}"
        "</svg>"
    )
