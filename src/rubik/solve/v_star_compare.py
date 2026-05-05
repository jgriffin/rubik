"""Compare greedy solve lengths to BFS-optimal V* (2x2 only).

The 2x2's small canonical state space (3,674,160 states modulo cube
rotations) admits a full BFS, so for any state we have its true optimal
QTM-distance to solved. This module turns greedy ``solve_lens`` into a
per-attempt ``excess`` tensor — how many extra moves greedy took over
BFS-optimal — by looking up V*(state) for each starting (scrambled) state.

3x3 is intentionally not supported: its state space is too large for a
practical BFS oracle, and any value-net evaluation there has to fall back
on solve-rate and avg-solve-length alone.
"""

from __future__ import annotations

import numpy as np
import torch

from rubik.oracle.v_star_2x2 import lookup_v_star_batch


def compute_excess_vs_v_star(
    solve_lens: torch.Tensor | np.ndarray,
    states: torch.Tensor | np.ndarray,
    v_star: dict[bytes, int],
) -> np.ndarray:
    """Per-attempt suboptimality vs BFS-optimal: ``solve_len - V*(state)``.

    Args:
        solve_lens: ``(N,)`` int tensor / array of greedy solve lengths from
            ``greedy_solve_batch``. ``-1`` entries mark failed attempts.
        states: ``(N, 24)`` — the *scrambled starting states* (where greedy
            began). Anything torch / numpy with shape ``(N, 24)``;
            canonicalised internally to look up against ``v_star``.
        v_star: BFS table from ``compute_v_star_2x2`` /
            ``load_v_star`` — maps canonical-state-bytes → optimal depth.

    Returns:
        ``(N,)`` int64 array. For successful attempts: ``solve_len - V*(s)``
        ≥ 0 (``0`` = greedy was BFS-optimal, ``> 0`` = greedy took N extra
        moves). For failed attempts (``solve_lens == -1``): returns ``-1``,
        propagating the failure marker so summary stats can mask them out.
    """
    if isinstance(solve_lens, torch.Tensor):
        solve_lens_np = solve_lens.detach().cpu().numpy().astype(np.int64)
    else:
        solve_lens_np = np.asarray(solve_lens, dtype=np.int64)

    n_states = states.shape[0] if hasattr(states, "shape") else len(states)
    if solve_lens_np.shape[0] != n_states:
        raise ValueError(
            f"solve_lens length {solve_lens_np.shape[0]} != states length {n_states}"
        )

    truth = lookup_v_star_batch(states, v_star).astype(np.int64)
    excess = solve_lens_np - truth
    excess[solve_lens_np < 0] = -1
    return excess
