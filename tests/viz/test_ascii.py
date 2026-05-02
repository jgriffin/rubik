"""Tests for the ASCII unfolded-cross renderer."""

from rubik.cube.env import apply_moves
from rubik.cube.spec import CUBE_2X2
from rubik.notation.moves import str_to_move
from rubik.notation.state import state_to_dict
from rubik.viz.ascii import render_ascii


def _render(state) -> str:
    return render_ascii(state_to_dict(state, CUBE_2X2), CUBE_2X2)


def test_render_ascii_solved() -> None:
    expected = "    U U\n    U U\nL L F F R R B B\nL L F F R R B B\n    D D\n    D D"
    assert _render(CUBE_2X2.solved_state) == expected


def test_render_ascii_after_R() -> None:
    # R cycles the right column of stickers through U -> B -> D -> F -> U
    # (CW viewed from +R) and rotates the R face among itself. In our
    # unfolded cross:
    #   - U's right column (col=1) takes on F's color.
    #   - F's right column takes on D's color.
    #   - D's right column takes on B's color.
    #   - B's left column (which is geometrically adjacent to R) takes on
    #     U's color (B is laid out flipped in the unfolded cross, so its
    #     "left column" sits next to R).
    #   - R face rotates 90 deg CW: TL,TR,BL,BR -> BL,TL,BR,TR — but the
    #     stickers all start as R so visually unchanged.
    state = apply_moves(CUBE_2X2.solved_state, str_to_move("R"), CUBE_2X2)
    expected = "    U F\n    U F\nL L F D R R U B\nL L F D R R U B\n    D B\n    D B"
    assert _render(state) == expected


def test_render_ascii_after_U() -> None:
    # U rotates the top layer CW (viewed from +U). Side faces' top rows
    # cycle: F-top -> L-top -> B-top -> R-top -> F-top. In our layout
    # that means:
    #   - L's top row takes on F's color
    #   - F's top row takes on R's color
    #   - R's top row takes on B's color
    #   - B's top row takes on L's color
    # U face itself rotates 90 deg CW (all U color, visually unchanged).
    # Bottom rows (and D) untouched.
    state = apply_moves(CUBE_2X2.solved_state, str_to_move("U"), CUBE_2X2)
    expected = "    U U\n    U U\nF F R R B B L L\nL L F F R R B B\n    D D\n    D D"
    assert _render(state) == expected


def test_render_ascii_sticker_count() -> None:
    # 24 face letters total for a 2x2 (6 faces * 4 stickers).
    out = _render(CUBE_2X2.solved_state)
    count = sum(1 for ch in out if ch in {"U", "L", "F", "R", "B", "D"})
    assert count == 24
