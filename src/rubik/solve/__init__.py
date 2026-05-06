"""Greedy solve evaluation primitives.

`greedy_solve_batch` is the workhorse: takes a value net + scrambled states,
runs greedy ``argmin V_θ(child)`` until each state is solved or a per-row
budget is exhausted, returns per-attempt solve lengths. Spec-agnostic — the
same code path serves 2x2 and 3x3.

`compute_excess_vs_v_star` measures suboptimality on the 2x2 cube: per-attempt
``greedy_solve_len - V*(scrambled_state)``. 2x2-only because the 3x3 has no
practical V* oracle.

`summarize_solve_lens` collapses raw per-attempt solve lengths into the
summary dict (``solve_rate``, ``avg_solve_len``) used in eval logs.
"""

from rubik.solve.greedy import greedy_solve_batch, summarize_solve_lens
from rubik.solve.v_star_compare import compute_excess_vs_v_star

__all__ = ["greedy_solve_batch", "summarize_solve_lens", "compute_excess_vs_v_star"]
