"""Visual sanity-check for the 3x3 cubie oracle rotations.

Renders the cube as an unfolded-cross HTML page at every step of several
move sequences. Open the resulting HTML in a browser to spot-check that
rotations move stickers the way you'd expect.

What you see is exactly what `cubie_to_tensor_3x3` produces — there is no
parallel rendering logic. If a state looks wrong here, the oracle is
wrong.

Mirrors `render_oracle_rotations_2x2.py`. The SVG renderer is already
parametric on `spec.size`, so the only differences are the imports
(`CUBE_3X3` / `SOLVED_3X3` / `apply_move_3x3` / `cubie_to_tensor_3x3`)
and the addition of a 3x3-specific F-move edge-flip detail section
(2x2 has no edges, so no flip to visualize there).

Run:
    uv run python visuals/scripts/render_oracle_rotations_3x3.py
    open visuals/oracle_rotations_3x3.html
"""

import random
from pathlib import Path

from rubik.cube.spec import CUBE_3X3
from rubik.notation.moves import move_to_str, str_to_move
from rubik.notation.state import state_to_dict
from rubik.oracle import SOLVED_3X3, apply_move_3x3, cubie_to_tensor_3x3
from rubik.viz.colors import SPEEDCUBING_PALETTE
from rubik.viz.svg import render_svg_state

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
code {
    background: #2a2a2a;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.92em;
}
"""

SCRAMBLE_SEED = 42
SCRAMBLE_DEPTH = 10


def _render_cube_net(state) -> str:
    tensor = cubie_to_tensor_3x3(state, CUBE_3X3)
    return render_svg_state(state_to_dict(tensor, CUBE_3X3), CUBE_3X3)


def _labeled(label: str, state) -> str:
    return (
        f'<div class="labeled-cube"><div class="label">{label}</div>'
        f"{_render_cube_net(state)}</div>"
    )


def _solved_section() -> str:
    return (
        "<section><h2>Solved state</h2>"
        '<p class="section-intro">Reference unfolded cross for SOLVED_3X3. '
        "Each face is one solid color; centers (the middle sticker of each "
        "face) define face identity. Compare against the color legend below."
        "</p>"
        f'<div class="cubes">{_labeled("solved", SOLVED_3X3)}</div></section>'
    )


def _sequence_section(title: str, seq: str, description: str = "") -> str:
    moves = [str_to_move(s) for s in seq.split()] if seq else []
    state = SOLVED_3X3
    cubes = [_labeled("solved", state)]
    accum: list[str] = []
    for m in moves:
        state = apply_move_3x3(state, m)
        accum.append(move_to_str(m))
        cubes.append(_labeled(" ".join(accum), state))
    intro = f'<p class="section-intro">{description}</p>' if description else ""
    return (
        f"<section><h2>{title}</h2>{intro}"
        f'<div class="cubes">{"".join(cubes)}</div></section>'
    )


def _singles_section() -> str:
    moves = ["U", "U'", "L", "L'", "F", "F'", "R", "R'", "B", "B'", "D", "D'"]
    cubes = [_labeled("solved", SOLVED_3X3)]
    for m_str in moves:
        state = apply_move_3x3(SOLVED_3X3, str_to_move(m_str))
        cubes.append(_labeled(m_str, state))
    return (
        "<section><h2>Each of the 12 moves applied to solved</h2>"
        '<p class="section-intro">Single-move sanity check. Each cube shows '
        "one move applied to solved. Compare each CW/CCW pair (e.g. R vs R'): "
        "they should mirror each other. Centers stay put — only the layer "
        "around each face's axis rotates.</p>"
        f'<div class="cubes">{"".join(cubes)}</div></section>'
    )


def _f_flip_detail_section() -> str:
    """3x3-only: the F move is the only QTM move that flips edge orientation
    under our U/D-axis convention. The 4 affected edges are UF, FR, DF, FL.
    This section makes the flip visually obvious by placing F and F' next to
    solved at large size, and labels the edges to look at.
    """
    f_state = apply_move_3x3(SOLVED_3X3, str_to_move("F"))
    fp_state = apply_move_3x3(SOLVED_3X3, str_to_move("F'"))
    cubes = [
        _labeled("solved", SOLVED_3X3),
        _labeled("F", f_state),
        _labeled("F'", fp_state),
    ]
    return (
        "<section><h2>F-move edge-flip detail (U/D-axis convention)</h2>"
        '<p class="section-intro">Under the U/D-axis edge-orientation '
        "convention, the F and B moves are the only QTM moves that flip edge "
        "orientation. Look at the 4 edges around the F face — UF (top of F), "
        "FR (right of F), DF (bottom of F), FL (left of F). After F, each of "
        "those edges has its U/D-color sticker swapped to the side facet. "
        "On the unfolded net this shows up as: the top edge of F now wears "
        "the side-color (not the U-color); the bottom edge of F now wears "
        "the side-color (not the D-color); and the F-side stickers of L and "
        "R now show U/D colors at the F-adjacent rim. F' is the mirror.</p>"
        f'<div class="cubes">{"".join(cubes)}</div></section>'
    )


def _scramble_section() -> str:
    rng = random.Random(SCRAMBLE_SEED)
    forward = [rng.randrange(12) for _ in range(SCRAMBLE_DEPTH)]
    inverse = [m ^ 1 for m in reversed(forward)]
    state = SOLVED_3X3
    cubes = [_labeled("solved", state)]
    accum: list[str] = []
    for m in forward:
        state = apply_move_3x3(state, m)
        accum.append(move_to_str(m))
        cubes.append(_labeled(" ".join(accum[-3:]) + " …", state))
    cubes[-1] = _labeled("scrambled", state)  # rename last forward step
    for m in inverse:
        state = apply_move_3x3(state, m)
        accum.append(move_to_str(m))
        cubes.append(_labeled(" ".join(accum[-3:]) + " …", state))
    cubes[-1] = _labeled("solved (back)", state)
    return (
        "<section><h2>Scramble + inverse — must return to solved</h2>"
        f'<p class="section-intro">{SCRAMBLE_DEPTH}-move random scramble '
        f"(seed {SCRAMBLE_SEED}) followed by its inverse (each move's "
        "direction bit flipped, applied in reverse order). Final cube should "
        "match the leftmost solved cube exactly.</p>"
        f'<div class="cubes">{"".join(cubes)}</div></section>'
    )


def _color_legend() -> str:
    items = []
    names = ["U white", "L orange", "F green", "R red", "B blue", "D yellow"]
    for face_idx, name in enumerate(names):
        items.append(
            '<div class="labeled-cube"><div class="label">' + name + "</div>"
            f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"'
            f' viewBox="0 0 24 24"><rect x="0" y="0" width="24" height="24"'
            f' rx="1.5" fill="{SPEEDCUBING_PALETTE[face_idx]}"'
            f' stroke="#0a0a0a"/></svg></div>'
        )
    return (
        "<section><h2>Color legend</h2>"
        f'<div class="cubes">{"".join(items)}</div></section>'
    )


def main() -> None:
    sections = [
        _color_legend(),
        _solved_section(),
        _singles_section(),
        _f_flip_detail_section(),
        _sequence_section(
            "Sexy move (R U R' U') — one iteration",
            "R U R' U'",
            "Six iterations of this sequence return to solved "
            "(see test_sexy_move_identity_3x3).",
        ),
        _sequence_section(
            "Sexy move ×6 — every step (must end at solved)",
            " ".join(["R", "U", "R'", "U'"] * 6),
            "Six iterations chained — 24 steps total. The final cube on "
            "the right must be byte-identical to the leftmost solved cube. "
            "If any of the 24 intermediate states looks wrong, the oracle's "
            "rotation is wrong; if only the final cube is wrong, the "
            "round-trip identity is broken.",
        ),
        _sequence_section(
            "Sune (R U R' U R U U R')",
            "R U R' U R U U R'",
            "QTM-expanded Sune (the standard HTM Sune is R U R' U R U2 R'). "
            "Six iterations should return to solved (see "
            "test_sune_identity_3x3).",
        ),
        _scramble_section(),
    ]

    body = (
        "<h1>3x3 — Cubie oracle rotations</h1>"
        '<p class="intro">Each cube is the unfolded cross: U on top, '
        "L | F | R | B in the middle row, D on the bottom. Within-face "
        "sticker layout matches the M1 <code>FACE_SLOTS</code> convention "
        "(row-major, viewed from outside, with per-face up-axes per the "
        "plan). Renders go through <code>cubie_to_tensor_3x3</code> "
        "directly — what you see is what the oracle says. Centers are "
        "fixed (they only rotate in place); each face center is the "
        "ground-truth color of its face.</p>" + "".join(sections)
    )

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>3x3 — Cubie oracle rotations</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )

    # Script lives in visuals/scripts/; output goes one level up in visuals/.
    out = Path(__file__).parent.parent / "oracle_rotations_3x3.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
