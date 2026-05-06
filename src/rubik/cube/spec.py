"""Parameterized cube specification.

A `CubeSpec` is the single source of truth for cube size and topology so the
env, oracle, training, and search code paths stay shared between 2x2 and 3x3
(no side-by-side branches).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CubeSpec:
    name: str
    size: int
    faces: tuple[str, ...]
    stickers_per_face: int
    n_colors: int
    n_moves: int

    @property
    def n_stickers(self) -> int:
        return len(self.faces) * self.stickers_per_face

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def solved_state(self) -> torch.Tensor:
        # Fresh tensor each call; callers may mutate freely without aliasing.
        return torch.arange(self.n_faces, dtype=torch.int8).repeat_interleave(
            self.stickers_per_face
        )


CUBE_2X2 = CubeSpec(
    name="2x2",
    size=2,
    faces=("U", "L", "F", "R", "B", "D"),
    stickers_per_face=4,
    n_colors=6,
    n_moves=12,
)

CUBE_3X3 = CubeSpec(
    name="3x3",
    size=3,
    faces=("U", "L", "F", "R", "B", "D"),
    stickers_per_face=9,
    n_colors=6,
    n_moves=12,
)
