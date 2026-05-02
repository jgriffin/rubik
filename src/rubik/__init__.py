"""rubik — deep RL solver for the Rubik's Cube on Apple Silicon (MPS).

See SPEC.md at the repo root for project decisions and milestones.
"""

from rubik.cube import (
    CUBE_2X2,
    CubeSpec,
    apply_move_sequence,
    apply_moves,
    is_solved,
    random_scrambles,
    valid_next_moves_mask,
)
from rubik.viz import (
    FACE_COLORS,
    render_ascii,
    render_svg_compare,
    render_svg_sequence,
    render_svg_state,
)

__version__ = "0.1.0"

__all__ = [
    "CUBE_2X2",
    "CubeSpec",
    "FACE_COLORS",
    "__version__",
    "apply_move_sequence",
    "apply_moves",
    "is_solved",
    "random_scrambles",
    "render_ascii",
    "render_svg_compare",
    "render_svg_sequence",
    "render_svg_state",
    "valid_next_moves_mask",
]
