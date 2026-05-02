"""Visual sanity-check for the 2x2 cubie oracle rotations.

Renders the cube as an unfolded-cross HTML page at every step of several
move sequences. Open the resulting HTML in a browser to spot-check that
rotations move stickers the way you'd expect.

What you see is exactly what `cubie_to_tensor` produces — there is no
parallel rendering logic. If a state looks wrong here, the oracle is
wrong.

Run:
    uv run python visuals/scripts/generate_cubie_2x2_rotations.py
    open visuals/cubie_2x2_rotations.html
"""

import random
from pathlib import Path

from rubik.cube.spec import CUBE_2X2
from rubik.notation.moves import move_to_str, str_to_move
from rubik.oracle.cubie import SOLVED, apply_move, cubie_to_tensor
from rubik.viz.colors import FACE_COLORS

# Position of each face in the unfolded-cross grid (rows × cols of stickers).
# Cross layout (each face is 2x2, so the sticker grid is 8 cols × 6 rows):
#     . . U U . . . .
#     . . U U . . . .
#     L L F F R R B B
#     L L F F R R B B
#     . . D D . . . .
#     . . D D . . . .
FACE_TOP_LEFT = {
    0: (1, 3),  # U
    1: (3, 1),  # L
    2: (3, 3),  # F
    3: (3, 5),  # R
    4: (3, 7),  # B
    5: (5, 3),  # D
}


CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a1a;
    color: #e8e8e8;
    margin: 0;
    padding: 24px 32px 48px;
}
h1 { font-size: 1.4em; margin: 0 0 8px; font-weight: 600; }
h2 {
    font-size: 0.95em; margin: 36px 0 4px;
    color: #d8d8d8; font-weight: 600; letter-spacing: 0.02em;
}
.intro, .section-intro {
    color: #888; max-width: 760px; line-height: 1.5;
    font-size: 0.85em; margin: 4px 0 16px;
}
.cubes {
    display: flex;
    flex-wrap: wrap;
    gap: 24px 18px;
    align-items: flex-start;
}
.labeled-cube {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.label {
    font-family: 'SF Mono', ui-monospace, Menlo, monospace;
    font-size: 0.72em;
    color: #888;
    white-space: nowrap;
    letter-spacing: 0.02em;
}
.cube-net {
    display: grid;
    grid-template-columns: repeat(8, 13px);
    grid-template-rows: repeat(6, 13px);
    gap: 1px;
    background: #1a1a1a;
}
.sticker {
    border: 1px solid #0a0a0a;
    box-sizing: border-box;
    border-radius: 1.5px;
}
code {
    background: #2a2a2a;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.92em;
}
"""


def _render_cube_net(state) -> str:
    colors = cubie_to_tensor(state, CUBE_2X2).tolist()
    cells: list[str] = []
    for face_idx in range(6):
        top, left = FACE_TOP_LEFT[face_idx]
        offset = face_idx * 4
        for sticker_idx in range(4):
            row, col = sticker_idx // 2, sticker_idx % 2
            color = FACE_COLORS[colors[offset + sticker_idx]]
            cells.append(
                f'<div class="sticker" style="grid-row:{top + row};'
                f'grid-column:{left + col};background:{color}"></div>'
            )
    return f'<div class="cube-net">{"".join(cells)}</div>'


def _labeled(label: str, state) -> str:
    return (
        f'<div class="labeled-cube"><div class="label">{label}</div>'
        f"{_render_cube_net(state)}</div>"
    )


def _sequence_section(title: str, seq: str, description: str = "") -> str:
    moves = [str_to_move(s) for s in seq.split()] if seq else []
    state = SOLVED
    cubes = [_labeled("solved", state)]
    accum: list[str] = []
    for m in moves:
        state = apply_move(state, m)
        accum.append(move_to_str(m))
        cubes.append(_labeled(" ".join(accum), state))
    intro = f'<p class="section-intro">{description}</p>' if description else ""
    return (
        f"<section><h2>{title}</h2>{intro}"
        f'<div class="cubes">{"".join(cubes)}</div></section>'
    )


def _singles_section() -> str:
    moves = ["U", "U'", "L", "L'", "F", "F'", "R", "R'", "B", "B'", "D", "D'"]
    cubes = [_labeled("solved", SOLVED)]
    for m_str in moves:
        state = apply_move(SOLVED, str_to_move(m_str))
        cubes.append(_labeled(m_str, state))
    return (
        "<section><h2>Each of the 12 moves applied to solved</h2>"
        '<p class="section-intro">Single-move sanity check. Each cube shows '
        "one move applied to solved. Compare each CW/CCW pair (e.g. R vs R'): "
        "they should mirror each other.</p>"
        f'<div class="cubes">{"".join(cubes)}</div></section>'
    )


def _scramble_section() -> str:
    rng = random.Random(123)
    forward = [rng.randrange(12) for _ in range(8)]
    inverse = [m ^ 1 for m in reversed(forward)]
    state = SOLVED
    cubes = [_labeled("solved", state)]
    accum: list[str] = []
    for m in forward:
        state = apply_move(state, m)
        accum.append(move_to_str(m))
        cubes.append(_labeled(" ".join(accum[-3:]) + " …", state))
    cubes[-1] = _labeled("scrambled", state)  # rename last forward step
    for m in inverse:
        state = apply_move(state, m)
        accum.append(move_to_str(m))
        cubes.append(_labeled(" ".join(accum[-3:]) + " …", state))
    cubes[-1] = _labeled("solved (back)", state)
    return (
        "<section><h2>Scramble + inverse — must return to solved</h2>"
        '<p class="section-intro">8-move random scramble (seed 123) followed '
        "by its inverse (each move's direction bit flipped, applied in reverse "
        "order). Final cube should match the leftmost solved cube exactly.</p>"
        f'<div class="cubes">{"".join(cubes)}</div></section>'
    )


def _color_legend() -> str:
    items = []
    names = ["U white", "L orange", "F green", "R red", "B blue", "D yellow"]
    for face_idx, name in enumerate(names):
        items.append(
            '<div class="labeled-cube"><div class="label">' + name + "</div>"
            f'<div class="sticker" style="width:24px;height:24px;'
            f'background:{FACE_COLORS[face_idx]}"></div></div>'
        )
    return (
        "<section><h2>Color legend</h2>"
        f'<div class="cubes">{"".join(items)}</div></section>'
    )


def main() -> None:
    sections = [
        _color_legend(),
        _singles_section(),
        _sequence_section(
            "Sexy move (R U R' U') — one iteration",
            "R U R' U'",
            "Six iterations of this sequence return to solved "
            "(see test_sexy_move_identity).",
        ),
        _sequence_section(
            "Sexy move ×3 — every step",
            "R U R' U' R U R' U' R U R' U'",
            "Three iterations chained. Watch how the cube cycles through "
            "states; ×6 total returns to solved.",
        ),
        _sequence_section(
            "Sune (R U R' U R U U R')",
            "R U R' U R U U R'",
            "QTM-expanded Sune (the standard HTM Sune is R U R' U R U2 R'). "
            "Six iterations should return to solved (see test_sune_identity).",
        ),
        _scramble_section(),
    ]

    body = (
        "<h1>2x2 — Cubie oracle rotations</h1>"
        '<p class="intro">Each cube is the unfolded cross: U on top, '
        "L | F | R | B in the middle row, D on the bottom. Within-face "
        "sticker layout matches the M1 <code>FACE_SLOTS</code> convention "
        "(row-major, viewed from outside, with per-face up-axes per the "
        "plan). Renders go through <code>cubie_to_tensor</code> directly — "
        "what you see is what the oracle says.</p>" + "".join(sections)
    )

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>2x2 — Cubie oracle rotations</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )

    # Script lives in visuals/scripts/; output goes one level up in visuals/.
    out = Path(__file__).parent.parent / "cubie_2x2_rotations.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
