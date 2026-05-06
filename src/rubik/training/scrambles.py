"""ADI batch generation for DAVI training.

`generate_adi_batch` produces an ADI-shaped batch: ``(states, depths,
last_faces)`` where each row is a backward random scramble of some
depth in ``[1, max_depth]``. The DAVI loop consumes these to compute
the Bellman target ``min_a (1 + V_target(child))`` for each state.

**Balanced per-depth slicing.** Rather than per-row variable-depth
generation (which wastes work — a depth-2 row would still iterate
through max_depth-2 unused steps under the simplest impl), we slice the
batch into ``max_depth`` per-depth groups of size ``B / max_depth``,
call existing ``random_scrambles`` per group, concat, and shuffle. The
remainder ``B mod max_depth`` is distributed to the lowest depths.
Reuses the M2 ``random_scrambles`` and its same-face pruning.
"""

from __future__ import annotations

import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CubeSpec


def generate_adi_batch(
    spec: CubeSpec,
    batch_size: int,
    max_depth: int,
    *,
    generator: torch.Generator | None = None,
    prune_same_face: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate an ADI batch with balanced per-depth slicing.

    Args:
        spec: cube spec.
        batch_size: number of states (B).
        max_depth: maximum scramble depth; depths sampled from [1, max_depth].
        generator: optional torch RNG for determinism.
        prune_same_face: if True, scrambles avoid same-face consecutive moves
            (matches both env and search conventions).

    Returns:
        states: ``(B, n_stickers)`` int8.
        depths: ``(B,)`` int64 — depth in ``[1, max_depth]``.
        last_faces: ``(B,)`` int64 — face index of the last applied move per
            row (0..n_faces-1). Used for non-trivial-move pruning at
            search-time eval.
    """
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative; got {batch_size}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1; got {max_depth}")

    if batch_size == 0:
        return (
            torch.empty((0, spec.n_stickers), dtype=torch.int8),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )

    # Per-depth slice sizes; remainder distributed to early depths.
    base = batch_size // max_depth
    remainder = batch_size % max_depth
    sizes = [base + (1 if d <= remainder else 0) for d in range(1, max_depth + 1)]

    states_parts: list[torch.Tensor] = []
    depths_parts: list[torch.Tensor] = []
    last_faces_parts: list[torch.Tensor] = []

    for d, size in enumerate(sizes, start=1):
        if size == 0:
            continue
        states_d, move_seqs_d = random_scrambles(
            spec,
            batch_size=size,
            depth=d,
            generator=generator,
            prune_same_face=prune_same_face,
        )
        last_face_d = move_seqs_d[:, -1] >> 1  # face_idx = move_idx >> 1
        depth_d = torch.full((size,), d, dtype=torch.int64)
        states_parts.append(states_d)
        depths_parts.append(depth_d)
        last_faces_parts.append(last_face_d)

    states = torch.cat(states_parts, dim=0)
    depths = torch.cat(depths_parts, dim=0)
    last_faces = torch.cat(last_faces_parts, dim=0)

    perm = torch.randperm(batch_size, generator=generator)
    return states[perm], depths[perm], last_faces[perm]
