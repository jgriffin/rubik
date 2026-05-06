"""V*-comparison primitives — excess against BFS-optimal solve length."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.oracle.v_star_2x2 import compute_v_star_2x2, lookup_v_star_batch
from rubik.solve import compute_excess_vs_v_star


def test_solved_state_v_star_zero_excess_zero():
    v = compute_v_star_2x2(CUBE_2X2, max_depth=2)
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(3, -1).clone()
    solve_lens = torch.zeros(3, dtype=torch.int64)
    excess = compute_excess_vs_v_star(solve_lens, states, v)
    assert excess.tolist() == [0, 0, 0]


def test_failed_attempts_propagate_minus_one():
    v = compute_v_star_2x2(CUBE_2X2, max_depth=3)
    rng = torch.Generator().manual_seed(0)
    states, _ = random_scrambles(CUBE_2X2, batch_size=4, depth=2, generator=rng)
    solve_lens = torch.tensor([-1, 5, -1, 4], dtype=torch.int64)
    excess = compute_excess_vs_v_star(solve_lens, states, v)
    truth = lookup_v_star_batch(states, v).astype(np.int64)
    assert excess[0] == -1 and excess[2] == -1
    assert excess[1] == 5 - truth[1]
    assert excess[3] == 4 - truth[3]


def test_excess_matches_subtraction_against_lookup():
    """``compute_excess_vs_v_star`` is exactly ``solve_lens - lookup_v_star_batch``
    for successful attempts, with ``-1`` propagated on failures.
    """
    v = compute_v_star_2x2(CUBE_2X2, max_depth=3)
    rng = torch.Generator().manual_seed(13)
    states, _ = random_scrambles(
        CUBE_2X2, batch_size=8, depth=3, generator=rng, prune_same_face=True
    )
    truth = lookup_v_star_batch(states, v).astype(np.int64)
    # Mix of solved-in-various-lengths and one failure.
    solve_lens = torch.tensor([int(t) for t in truth[:-1]] + [-1], dtype=torch.int64)
    excess = compute_excess_vs_v_star(solve_lens, states, v)
    expected = (solve_lens.numpy() - truth).copy()
    expected[solve_lens.numpy() < 0] = -1
    assert np.array_equal(excess, expected)


def test_excess_nonnegative_for_successful_solves():
    """For any successful greedy solve_len, excess = solve_len - V* must be ≥ 0
    (V* is the *optimal* cost, so any policy needs ≥ V* moves).
    """
    v = compute_v_star_2x2(CUBE_2X2, max_depth=3)
    rng = torch.Generator().manual_seed(21)
    states, _ = random_scrambles(CUBE_2X2, batch_size=8, depth=3, generator=rng)
    truth = lookup_v_star_batch(states, v).astype(np.int64)
    # Construct a synthetic solve_lens always ≥ V*; sanity-check excess ≥ 0.
    solve_lens = torch.from_numpy(truth + 2).to(torch.int64)
    excess = compute_excess_vs_v_star(solve_lens, states, v)
    assert (excess == 2).all()


def test_accepts_numpy_inputs():
    v = compute_v_star_2x2(CUBE_2X2, max_depth=2)
    rng = torch.Generator().manual_seed(5)
    states, _ = random_scrambles(CUBE_2X2, batch_size=3, depth=1, generator=rng)
    states_np = states.numpy()
    solve_lens_np = np.array([1, 1, 1], dtype=np.int64)
    excess = compute_excess_vs_v_star(solve_lens_np, states_np, v)
    assert excess.tolist() == [0, 0, 0]


def test_length_mismatch_raises():
    v = compute_v_star_2x2(CUBE_2X2, max_depth=2)
    rng = torch.Generator().manual_seed(1)
    states, _ = random_scrambles(CUBE_2X2, batch_size=3, depth=1, generator=rng)
    solve_lens = torch.tensor([1, 1], dtype=torch.int64)  # 2 != 3
    with pytest.raises(ValueError, match="solve_lens length"):
        compute_excess_vs_v_star(solve_lens, states, v)
