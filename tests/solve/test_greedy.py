"""Greedy solve primitives — boundary cases + V*-perfect-net oracle."""

from __future__ import annotations

import numpy as np
import torch

from rubik.cube.env import MOVE_PERM_2X2, random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.solve import greedy_solve_batch, summarize_solve_lens


def _physical_v_star_2x2(max_depth: int) -> dict[bytes, int]:
    """BFS from physical-solved (no canonicalization) — physical QTM distance.

    The canonical V* in ``rubik.oracle.v_star_2x2`` is *orbit*-distance
    (modulo the 24 cube rotations). For greedy-policy testing we need
    *physical* distance to the specific ``spec.solved_state``, since
    ``is_solved`` checks strict equality. Otherwise an orbit-V*-oracle can
    converge to a rotated-solved that ``is_solved`` rejects.
    """
    move_perm = MOVE_PERM_2X2.numpy().astype(np.int64)
    solved = CUBE_2X2.solved_state.numpy().astype(np.int8)
    visited: dict[bytes, int] = {solved.tobytes(): 0}
    frontier: list[np.ndarray] = [solved]
    for depth in range(1, max_depth + 1):
        new_frontier: list[np.ndarray] = []
        for s in frontier:
            for m in range(CUBE_2X2.n_moves):
                child = s[move_perm[m]].astype(np.int8)
                key = child.tobytes()
                if key not in visited:
                    visited[key] = depth
                    new_frontier.append(child)
        if not new_frontier:
            break
        frontier = new_frontier
    return visited


class _VStarNet(torch.nn.Module):
    """Stub net returning true *physical* V* per state — the optimal value head.

    With this oracle, greedy ``argmin V_θ(child)`` is BFS-optimal toward
    the canonical physical-solved state: from any state s with V*(s) = k,
    at least one child has V* = k - 1, others have V* in {k - 1, k, k + 1};
    argmin descends and reaches physical-solved in exactly V*(s) moves.
    """

    def __init__(self, v_star: dict[bytes, int], default: float = 1e6) -> None:
        super().__init__()
        self.v_star = v_star
        # Anything outside the BFS horizon gets a large value so it loses
        # in argmin to in-horizon children. Lets tests use a small BFS depth
        # without the policy "falling off the edge of the map."
        self.default = default
        # Anchor parameter so .to(device) / next(net.parameters()) work.
        self._dev_anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states_np = states.detach().cpu().numpy().astype(np.int8)
        depths = np.array(
            [self.v_star.get(s.tobytes(), self.default) for s in states_np],
            dtype=np.float32,
        )
        return torch.from_numpy(depths).to(self._dev_anchor.device)


class _ConstantNet(torch.nn.Module):
    """Returns a constant value for any input — a deliberately useless policy."""

    def __init__(self, c: float = 0.0) -> None:
        super().__init__()
        self.c = c
        self._dev_anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (states.shape[0],),
            self.c,
            device=self._dev_anchor.device,
            dtype=torch.float32,
        )


def test_already_solved_returns_zero():
    net = _ConstantNet()
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(5, -1).clone()
    lens = greedy_solve_batch(net, CUBE_2X2, states, max_steps=3)
    assert lens.shape == (5,)
    assert torch.all(lens == 0)


def test_v_star_perfect_net_solves_in_v_star_steps():
    """A physical-V*-oracle net solves every state in exactly V*(state) moves.

    Use a max_depth=3 physical BFS; tests d ∈ {1, 2, 3} scrambles. Each
    scrambled row's true physical-V* is its key in the BFS dict (the
    walk's *physical* distance to physical-solved, ≤ walk length).
    """
    v_phys = _physical_v_star_2x2(max_depth=3)
    net = _VStarNet(v_phys)

    rng = torch.Generator().manual_seed(42)
    for d in (1, 2, 3):
        states, _ = random_scrambles(
            CUBE_2X2, batch_size=15, depth=d, generator=rng, prune_same_face=True
        )
        truth = np.array(
            [v_phys[s.numpy().astype(np.int8).tobytes()] for s in states],
            dtype=np.int64,
        )
        lens = greedy_solve_batch(net, CUBE_2X2, states, max_steps=2 * d)
        assert torch.all(lens >= 0), f"d={d}: some rows failed"
        assert np.array_equal(lens.cpu().numpy(), truth), (
            f"d={d}: greedy lens {lens.cpu().tolist()} != V*(state) {truth.tolist()}"
        )


def test_max_steps_zero_marks_unsolved_failed():
    """max_steps=0: solved rows record 0; unsolved rows stay at -1."""
    net = _ConstantNet()
    rng = torch.Generator().manual_seed(7)
    scrambled, _ = random_scrambles(
        CUBE_2X2, batch_size=4, depth=2, generator=rng, prune_same_face=True
    )
    solved = CUBE_2X2.solved_state.unsqueeze(0).expand(2, -1)
    states = torch.cat([solved, scrambled], dim=0)  # (6, 24): 2 solved + 4 scrambled

    lens = greedy_solve_batch(net, CUBE_2X2, states, max_steps=0)
    assert lens.tolist() == [0, 0, -1, -1, -1, -1]


def test_does_not_mutate_input_states():
    """The caller's tensor is preserved; solving works on an internal copy."""
    v_phys = _physical_v_star_2x2(max_depth=3)
    net = _VStarNet(v_phys)
    rng = torch.Generator().manual_seed(11)
    states, _ = random_scrambles(
        CUBE_2X2, batch_size=6, depth=2, generator=rng, prune_same_face=True
    )
    original = states.clone()
    _ = greedy_solve_batch(net, CUBE_2X2, states, max_steps=4)
    assert torch.equal(states, original)


def test_restores_train_mode_after_call():
    net = _ConstantNet()
    net.train()
    assert net.training is True

    states = CUBE_2X2.solved_state.unsqueeze(0).expand(2, -1).clone()
    _ = greedy_solve_batch(net, CUBE_2X2, states, max_steps=1)
    assert net.training is True

    net.eval()
    assert net.training is False
    _ = greedy_solve_batch(net, CUBE_2X2, states, max_steps=1)
    assert net.training is False


def test_constant_net_at_depth_3_with_tight_budget_fails():
    """A constant value head can't choose a useful child; budget=1 from a
    depth-3 scramble must fail (the policy can reach solved by chance only
    if a child happens to be solved, which is almost surely false at d=3).
    """
    net = _ConstantNet()
    rng = torch.Generator().manual_seed(99)
    states, _ = random_scrambles(
        CUBE_2X2, batch_size=20, depth=3, generator=rng, prune_same_face=True
    )
    lens = greedy_solve_batch(net, CUBE_2X2, states, max_steps=1)
    # All 20 unsolved at depth 3, none can reach solved in 1 move.
    assert torch.all(lens == -1)


def test_rejects_wrong_shape():
    import pytest

    net = _ConstantNet()
    bad = torch.zeros((4, 20), dtype=torch.int8)  # wrong sticker count
    with pytest.raises(ValueError, match="states"):
        greedy_solve_batch(net, CUBE_2X2, bad, max_steps=1)


def test_rejects_negative_max_steps():
    import pytest

    net = _ConstantNet()
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(2, -1).clone()
    with pytest.raises(ValueError, match="max_steps"):
        greedy_solve_batch(net, CUBE_2X2, states, max_steps=-1)


def test_summarize_all_solved():
    lens = torch.tensor([0, 1, 3, 5], dtype=torch.int64)
    s = summarize_solve_lens(lens)
    assert s["n"] == 4
    assert s["n_solved"] == 4
    assert s["solve_rate"] == 1.0
    assert s["avg_solve_len"] == (0 + 1 + 3 + 5) / 4


def test_summarize_mixed():
    lens = torch.tensor([2, -1, 4, -1, -1], dtype=torch.int64)
    s = summarize_solve_lens(lens)
    assert s["n"] == 5
    assert s["n_solved"] == 2
    assert s["solve_rate"] == 2 / 5
    assert s["avg_solve_len"] == (2 + 4) / 2


def test_summarize_all_failed_avg_is_none():
    lens = torch.tensor([-1, -1, -1], dtype=torch.int64)
    s = summarize_solve_lens(lens)
    assert s["n"] == 3
    assert s["n_solved"] == 0
    assert s["solve_rate"] == 0.0
    assert s["avg_solve_len"] is None


def test_summarize_empty():
    lens = torch.tensor([], dtype=torch.int64)
    s = summarize_solve_lens(lens)
    assert s["n"] == 0
    assert s["n_solved"] == 0
    assert s["solve_rate"] == 0.0
    assert s["avg_solve_len"] is None
