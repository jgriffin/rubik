"""Static HTML preview for the M3 viz stack on a 2x2 cube.

Renders four sections:

1. Solved — single state.
2. Single-move — solved -> R, U, F as a 4-frame sequence.
3. Depth-10 canonical — step-by-step application of "R U R' U' R U R' U'"
   (eight quarter-turn moves, partial cycle of the sexy move).
4. Depth-20 scramble — top row: ``[solved, scrambled]``; bottom row:
   ``[scrambled, after-inverse]`` (must equal solved).

The renderers used here are exactly the ones the package ships with
(``rubik.viz.svg``); this file is the M3 acceptance "favorite-pattern"
preview — open it in a browser to eyeball the geometry.

Run:

    uv run python visuals/scripts/render_cube_examples_2x2.py
    open visuals/cube_examples_2x2.html
"""

from pathlib import Path

import torch

from rubik.cube.env import apply_moves, random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.notation.moves import move_to_str, str_to_move
from rubik.notation.state import state_to_dict
from rubik.viz.svg import render_svg_compare, render_svg_sequence, render_svg_state

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
.frame {
    display: inline-block;
    padding: 12px 16px;
    background: #1a1a1a;
}
code {
    background: #2a2a2a;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.92em;
    font-family: 'SF Mono', ui-monospace, Menlo, monospace;
}
"""


def _fd(state):
    return state_to_dict(state, CUBE_2X2)


def _section_solved() -> str:
    svg = render_svg_state(_fd(CUBE_2X2.solved_state), CUBE_2X2)
    return (
        "<section><h2>Solved</h2>"
        '<p class="section-intro">The 2x2 unfolded cross. Faces in '
        "<code>(U,L,F,R,B,D)</code> order; each face is a 2x2 block of "
        "stickers in row-major TL/TR/BL/BR per the M1 "
        "<code>FACE_SLOTS</code> contract.</p>"
        f'<div class="frame">{svg}</div></section>'
    )


def _section_single_move() -> str:
    solved = CUBE_2X2.solved_state
    moves = ["R", "U", "F"]
    states = [solved] + [apply_moves(solved, str_to_move(m), CUBE_2X2) for m in moves]
    labels = ["solved", *moves]
    svg = render_svg_sequence([_fd(s) for s in states], CUBE_2X2, labels=labels)
    return (
        "<section><h2>Single-move</h2>"
        '<p class="section-intro">Three single quarter-turns applied to '
        "solved. Compare each result with its CW sibling face — the "
        "sticker pattern should be the M1 rotation cycle for that face.</p>"
        f'<div class="frame">{svg}</div></section>'
    )


def _section_depth10_canonical() -> str:
    seq_str = ["R", "U", "R'", "U'", "R", "U", "R'", "U'"]
    state = CUBE_2X2.solved_state
    states = [state]
    for s in seq_str:
        state = apply_moves(state, str_to_move(s), CUBE_2X2)
        states.append(state)
    labels = ["solved"]
    accum: list[str] = []
    for s in seq_str:
        accum.append(s)
        labels.append(" ".join(accum))
    svg = render_svg_sequence([_fd(s) for s in states], CUBE_2X2, labels=labels)
    return (
        "<section><h2>Depth-8 canonical (sexy x2)</h2>"
        '<p class="section-intro">Two iterations of the sexy move '
        "<code>R U R&#39; U&#39;</code>. Six iterations form an identity; "
        "two iterations show partial scrambling that returns to solved "
        "after four more reps.</p>"
        f'<div class="frame">{svg}</div></section>'
    )


def _section_depth20_scramble() -> str:
    rng = torch.Generator()
    rng.manual_seed(42)
    states_batched, move_seqs = random_scrambles(
        CUBE_2X2, batch_size=1, depth=20, generator=rng
    )
    scrambled = states_batched[0]
    forward = move_seqs[0].tolist()
    # Inverse: flip direction bit, apply in reverse order.
    inverse = [m ^ 1 for m in reversed(forward)]
    state = scrambled
    for m in inverse:
        state = apply_moves(state, m, CUBE_2X2)
    after_inverse = state

    solved = CUBE_2X2.solved_state
    forward_str = " ".join(move_to_str(m) for m in forward)
    inverse_str = " ".join(move_to_str(m) for m in inverse)

    svg = render_svg_compare(
        [_fd(solved), _fd(scrambled)],
        [_fd(scrambled), _fd(after_inverse)],
        CUBE_2X2,
        top_labels=["solved", "scrambled (depth 20)"],
        bottom_labels=["scrambled", "after inverse"],
    )
    return (
        "<section><h2>Depth-20 scramble + inverse</h2>"
        '<p class="section-intro">'
        f"Forward (seed 42, depth 20): <code>{forward_str}</code><br>"
        f"Inverse: <code>{inverse_str}</code><br>"
        "The bottom-right cube must visually equal the top-left "
        "(solved) — that's the round-trip check.</p>"
        f'<div class="frame">{svg}</div></section>'
    )


def main() -> None:
    sections = [
        _section_solved(),
        _section_single_move(),
        _section_depth10_canonical(),
        _section_depth20_scramble(),
    ]

    body = (
        "<h1>2x2 — Cube preview (M3)</h1>"
        '<p class="intro">Rendered through <code>rubik.viz.svg</code> — '
        "the production renderer. Each cube is the unfolded cross: U on "
        "top, L | F | R | B in the middle row, D on the bottom. Within-face "
        "sticker layout matches the M1 <code>FACE_SLOTS</code> convention "
        "(row-major TL/TR/BL/BR, viewed from outside).</p>" + "".join(sections)
    )

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>2x2 — Cube preview (M3)</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )

    out = Path(__file__).parent.parent / "cube_examples_2x2.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
