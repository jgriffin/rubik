"""Snapshot tests for SVG renderers.

First run: ``uv run pytest tests/viz/test_svg.py --snapshot-update`` to
materialize the expected SVG files. Thereafter, plain ``uv run pytest``
verifies byte-equality.
"""

from rubik.cube.env import apply_move_sequence, apply_moves
from rubik.cube.spec import CUBE_2X2
from rubik.notation.moves import str_to_move
from rubik.notation.state import state_to_dict
from rubik.viz.svg import render_svg_compare, render_svg_sequence, render_svg_state

SNAPSHOT_DIR = "tests/viz/snapshots"


def _fd(state):
    return state_to_dict(state, CUBE_2X2)


def test_svg_state_solved(snapshot) -> None:
    snapshot.snapshot_dir = SNAPSHOT_DIR
    svg = render_svg_state(_fd(CUBE_2X2.solved_state), CUBE_2X2)
    snapshot.assert_match(svg, "state_solved.svg")


def test_svg_state_after_R(snapshot) -> None:
    snapshot.snapshot_dir = SNAPSHOT_DIR
    state = apply_moves(CUBE_2X2.solved_state, str_to_move("R"), CUBE_2X2)
    svg = render_svg_state(_fd(state), CUBE_2X2)
    snapshot.assert_match(svg, "state_after_R.svg")


def test_svg_sequence_solved_R_U(snapshot) -> None:
    snapshot.snapshot_dir = SNAPSHOT_DIR
    solved = CUBE_2X2.solved_state
    after_r = apply_moves(solved, str_to_move("R"), CUBE_2X2)
    after_ru = apply_moves(after_r, str_to_move("U"), CUBE_2X2)
    svg = render_svg_sequence(
        [_fd(solved), _fd(after_r), _fd(after_ru)],
        CUBE_2X2,
        labels=["", "R", "U"],
    )
    snapshot.assert_match(svg, "sequence_solved_R_U.svg")


def test_svg_compare_scramble_vs_solved(snapshot) -> None:
    snapshot.snapshot_dir = SNAPSHOT_DIR
    solved = CUBE_2X2.solved_state
    # Deterministic 5-move forward sequence (not all on the same face).
    forward = [str_to_move(s) for s in ("R", "U", "F", "L", "B")]
    scrambled = apply_move_sequence(solved, forward, CUBE_2X2)
    # Inverse: flip direction bit on each move, apply in reverse order.
    inverse = [m ^ 1 for m in reversed(forward)]
    back_to_solved = apply_move_sequence(scrambled, inverse, CUBE_2X2)
    svg = render_svg_compare(
        [_fd(solved), _fd(scrambled), _fd(solved)],
        [_fd(solved), _fd(scrambled), _fd(back_to_solved)],
        CUBE_2X2,
        top_labels=["solved", "scrambled", "solved"],
        bottom_labels=["solved", "scrambled", "after inverse"],
    )
    snapshot.assert_match(svg, "compare_scramble_vs_solved.svg")
