"""Bounded BFS V* oracle for the 3x3 cube — depths 0..K only.

The 3x3 has ~4.3×10^19 reachable states — full BFS enumeration like the
2x2 oracle (`v_star_2x2`) is out of reach. This module instead caches a
*bounded* slice: every state at QTM-distance ≤ K from solved.

For K=6 the cumulative count is exactly **983,926** states (well-known
from cube literature; see plan ``m8-3x3-davi.md`` Phase-1 P1b layer table):

    d=0:           1
    d=1:          12   (cumulative      13)
    d=2:         114   (cumulative     127)
    d=3:       1,068   (cumulative   1,195)
    d=4:      10,011   (cumulative  11,206)
    d=5:      93,840   (cumulative 105,046)
    d=6:     878,880   (cumulative 983,926)

Used by the M8 eval pipeline as a sanity-check ground truth at d≤6 (true
V* via lookup) and as a hazard-analysis side channel for walk-depth
sampling beyond d=6 (where V* is unknown — sentinel -1).

**No canonicalization.** Unlike `v_star_2x2`, this module does NOT
collapse the 24-rotation orbit. The 3x3 sticker tensor encoding has a
fixed center on each face (the centers are sticker-permutation-invariant
under face turns), so a raw `state.tobytes()` key is already canonical
*for this specific encoding*. The 24-rotation collapse on 2x2 was needed
because the 2x2 has no centers and the encoding is rotation-degenerate;
3x3 doesn't need it. Layer counts above are the uncanonicalized counts.

Public API (mirrors `v_star_2x2.py`'s shape, replacing the 2x2 size suffix
with 3x3 and dropping the canonicalization helpers):

- ``compute_v_star_bounded_3x3(spec, *, max_depth, cache_path, verbose)``
- ``load_v_star_bounded_3x3(cache_path)``
- ``load_v_star_bounded_3x3_arrays(cache_path)``
- ``save_v_star_bounded_3x3(v_star, path)``
- ``lookup_v_star_bounded_3x3(state, v_star)``
- ``lookup_v_star_bounded_3x3_batch(states, v_star)``
- ``sample_states_at_v_star_3x3(states, depths, target_v_star, n, *, rng)``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rubik.cube.env import MOVE_PERM_3X3
from rubik.cube.spec import CubeSpec

_MOVE_PERM_NP: np.ndarray = MOVE_PERM_3X3.numpy().astype(np.int64)


def compute_v_star_bounded_3x3(
    spec: CubeSpec,
    *,
    max_depth: int,
    cache_path: Path | None = None,
    verbose: bool = False,
) -> dict[bytes, int]:
    """BFS the 3x3 state space up to ``max_depth`` from solved.

    Args:
        spec: must be the 3x3 spec — this module is 3x3-only.
        max_depth: BFS frontier depth bound (K). Required (no full BFS
            possible on 3x3). For K=6, expect ~983,926 states.
        cache_path: optional .npz path. If exists, load instead of
            recomputing. If given and we computed fresh, save the result.
        verbose: print per-layer progress.

    Returns:
        dict[bytes, int] mapping ``state.tobytes()`` → optimal QTM depth.
    """
    if cache_path is not None and cache_path.exists():
        if verbose:
            print(f"loading bounded V* from {cache_path}")
        return load_v_star_bounded_3x3(cache_path)

    if spec.name != "3x3":
        raise ValueError(
            f"bounded V* enumeration only supported for 3x3; got {spec.name!r}"
        )
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative; got {max_depth}")

    solved = spec.solved_state.numpy().astype(np.int8)
    visited: dict[bytes, int] = {solved.tobytes(): 0}
    frontier = solved[None, :]  # (1, 54) int8

    if verbose:
        print("depth 0: 1 new state, 1 total")

    for depth in range(1, max_depth + 1):
        # Batched expansion: (B, 54) -> (B, 12, 54) via fancy-index against
        # the (12, 54) move-perm table. State-bytes dedup is sequential.
        children = frontier[:, _MOVE_PERM_NP].reshape(-1, 54).astype(np.int8)

        new_states_list: list[np.ndarray] = []
        for s in children:
            key = s.tobytes()
            if key not in visited:
                visited[key] = depth
                new_states_list.append(s)

        if verbose:
            print(
                f"depth {depth}: {len(new_states_list)} new states,"
                f" {len(visited)} total"
            )

        if not new_states_list:
            break
        frontier = np.stack(new_states_list)

    if cache_path is not None:
        save_v_star_bounded_3x3(visited, cache_path)
        if verbose:
            print(f"saved bounded V* to {cache_path}")

    return visited


def save_v_star_bounded_3x3(v_star: dict[bytes, int], path: Path) -> None:
    """Save bounded V* dict to a compressed npz file.

    Layout: ``states: int8[N, 54]``, ``depths: int8[N]``. Mirrors
    ``v_star_2x2``'s storage layout (just with 54-wide rows).
    """
    n = len(v_star)
    states = np.empty((n, 54), dtype=np.int8)
    depths = np.empty(n, dtype=np.int8)
    for i, (key, d) in enumerate(v_star.items()):
        states[i] = np.frombuffer(key, dtype=np.int8)
        depths[i] = d
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=states, depths=depths)


def load_v_star_bounded_3x3(path: Path) -> dict[bytes, int]:
    """Load bounded V* dict from a compressed npz file."""
    with np.load(path) as data:
        states = data["states"]
        depths = data["depths"]
    return {states[i].tobytes(): int(depths[i]) for i in range(len(states))}


def load_v_star_bounded_3x3_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load bounded V* as (states, depths) arrays — needed for V*-stratified
    sampling and eval-set construction (P1c).

    Returns:
        states: ``(N, 54)`` int8.
        depths: ``(N,)`` int8 — optimal QTM depth for the corresponding row.
    """
    with np.load(path) as data:
        states = data["states"].copy()
        depths = data["depths"].copy()
    return states, depths


def sample_states_at_v_star_3x3(
    states: np.ndarray,
    depths: np.ndarray,
    target_v_star: int,
    n: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample ``n`` states with ``V*[state] == target_v_star``.

    Mirrors ``v_star_2x2.sample_states_at_v_star`` but **without the rotate
    option**: 3x3 states stored in this oracle are not canonicalized over
    a 24-rotation orbit (see module docstring) — they are raw sticker
    tensors, exactly the shape the solver consumes. There is no rotation
    step to optionally undo.

    Args:
        states: ``(N, 54)`` int8 — from ``load_v_star_bounded_3x3_arrays``.
        depths: ``(N,)`` int — same N as ``states``; optimal QTM depth.
        target_v_star: the V* depth to sample from. ``0`` is the (lone)
            solved state.
        n: number of states to return.
        rng: numpy random generator (deterministic when seeded).

    Returns:
        ``(n, 54)`` int8 array of states. Sampling is with replacement
        when ``n`` exceeds the layer count at this depth.

    Raises:
        ValueError: if no states have ``V* == target_v_star``, or if
            ``n < 0``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    if n == 0:
        return np.empty((0, 54), dtype=np.int8)
    mask = depths == target_v_star
    if not mask.any():
        raise ValueError(
            f"no states with V*={target_v_star} in the table "
            f"(table covers V* ∈ {{{int(depths.min())}, ..., {int(depths.max())}}})"
        )
    candidate_idx = np.flatnonzero(mask)
    n_candidates = candidate_idx.shape[0]
    replace = n > n_candidates
    chosen_local = rng.choice(n_candidates, size=n, replace=replace)
    chosen = candidate_idx[chosen_local]
    return states[chosen].astype(np.int8, copy=True)


def lookup_v_star_bounded_3x3(
    state: np.ndarray | torch.Tensor, v_star: dict[bytes, int]
) -> int:
    """Look up the optimal QTM distance for a single 3x3 state.

    Raises ``KeyError`` if the state is not in the bounded oracle (i.e.
    its true V* exceeds the K used to build the cache).
    """
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    if state.shape != (54,):
        raise ValueError(f"expected (54,) state; got shape {state.shape}")
    if state.dtype != np.int8:
        state = state.astype(np.int8)
    return v_star[state.tobytes()]


def lookup_v_star_bounded_3x3_batch(
    states: np.ndarray | torch.Tensor,
    v_star: dict[bytes, int],
    *,
    missing: int = -1,
) -> np.ndarray:
    """Vectorized V* lookup for shape (B, 54). Returns (B,) int8.

    States not in the bounded oracle return ``missing`` (default -1) — the
    sentinel used by ``eval_set_3x3.npz`` for walk-depth-d>K endpoints.
    """
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    if states.ndim != 2 or states.shape[1] != 54:
        raise ValueError(f"expected (B, 54) states; got shape {states.shape}")
    if states.dtype != np.int8:
        states = states.astype(np.int8)
    out = np.empty(states.shape[0], dtype=np.int8)
    for i, s in enumerate(states):
        out[i] = v_star.get(s.tobytes(), missing)
    return out
