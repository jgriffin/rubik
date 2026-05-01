"""State <-> face-dict converters, parameterized on `CubeSpec`.

The dict form is the canonical human-readable handoff between the tensor
state and renderers / tests. Each face maps to a flat list of
`spec.stickers_per_face` ints; the renderer (M3) is what reshapes those
into 2D grids.
"""

import torch

from rubik.cube.spec import CubeSpec


def state_to_dict(state: torch.Tensor, spec: CubeSpec) -> dict[str, list[int]]:
    if state.shape != (spec.n_stickers,):
        raise ValueError(
            f"state shape {tuple(state.shape)} does not match spec.n_stickers"
            f" ({spec.n_stickers},)"
        )
    out: dict[str, list[int]] = {}
    for i, face in enumerate(spec.faces):
        start = i * spec.stickers_per_face
        end = start + spec.stickers_per_face
        out[face] = state[start:end].tolist()
    return out


def dict_to_state(d: dict[str, list[int]], spec: CubeSpec) -> torch.Tensor:
    if set(d.keys()) != set(spec.faces):
        raise ValueError(
            f"dict faces {sorted(d.keys())} do not match spec.faces {list(spec.faces)}"
        )
    flat: list[int] = []
    for face in spec.faces:
        block = d[face]
        if len(block) != spec.stickers_per_face:
            raise ValueError(
                f"face {face!r} has {len(block)} stickers, expected"
                f" {spec.stickers_per_face}"
            )
        flat.extend(block)
    return torch.tensor(flat, dtype=torch.int8)
