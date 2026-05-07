"""Beam search primitives — boundary cases, V*-oracle, width monotonicity."""

from __future__ import annotations

import numpy as np
import torch

from rubik.cube.env import (
    MOVE_PERM_2X2,
    apply_move_sequence,
    is_solved,
    random_scrambles,
)
from rubik.cube.spec import CUBE_2X2
from rubik.search import BeamSearchResult, beam_solve_batch
from rubik.solve import greedy_solve_batch


def _physical_v_star_2x2(max_depth: int) -> dict[bytes, int]:
    """BFS from physical-solved (no canonicalization) — physical QTM distance.

    Mirrors the helper in tests/solve/test_greedy.py: orbit-V* in
    ``rubik.oracle.v_star_2x2`` is rotation-equivalence distance, but
    ``is_solved`` checks strict equality with ``spec.solved_state``, so a
    policy oracle wants physical distance.
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
    """Stub net returning true *physical* V* per state — the optimal value head."""

    def __init__(self, v_star: dict[bytes, int], default: float = 1e6) -> None:
        super().__init__()
        self.v_star = v_star
        self.default = default
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
    result = beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=3)
    assert isinstance(result, BeamSearchResult)
    assert result.solve_lens.shape == (5,)
    assert torch.all(result.solve_lens == 0)
    assert all(p == [] for p in result.solve_paths)
    assert result.n_expansions == 0


def test_beam_width_1_matches_greedy_on_corpus():
    """Width=1 beam must reproduce greedy's solve_lens row-for-row.

    With one beam slot and 12 children that are all physically distinct
    (each move is a non-identity permutation), dedup is a no-op and the
    chosen move is ``argmin V_θ(child)`` — exactly greedy. The actual
    move chosen on ties may differ (greedy uses ``argmin`` with -1e9
    sentinel for solved children; beam exits on first solved by index)
    but the *length* is invariant.
    """
    v_phys = _physical_v_star_2x2(max_depth=4)
    net = _VStarNet(v_phys)

    rng = torch.Generator().manual_seed(123)
    for d in (1, 2, 3, 4):
        states, _ = random_scrambles(
            CUBE_2X2,
            batch_size=25,
            depth=d,
            generator=rng,
            prune_same_face=True,
        )
        greedy_lens = greedy_solve_batch(net, CUBE_2X2, states, max_steps=2 * d)
        beam_result = beam_solve_batch(
            net, CUBE_2X2, states, beam_width=1, max_steps=2 * d
        )
        assert beam_result.solve_lens.tolist() == greedy_lens.tolist(), (
            f"d={d}: beam_lens {beam_result.solve_lens.tolist()} != "
            f"greedy_lens {greedy_lens.tolist()}"
        )


def test_beam_oracle_solves_in_v_star_steps():
    """A V*-oracle net at width=4 solves every row in V*(state) moves."""
    v_phys = _physical_v_star_2x2(max_depth=3)
    net = _VStarNet(v_phys)

    rng = torch.Generator().manual_seed(42)
    for d in (1, 2, 3):
        states, _ = random_scrambles(
            CUBE_2X2,
            batch_size=15,
            depth=d,
            generator=rng,
            prune_same_face=True,
        )
        truth = np.array(
            [v_phys[s.numpy().astype(np.int8).tobytes()] for s in states],
            dtype=np.int64,
        )
        result = beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=2 * d)
        assert torch.all(result.solve_lens >= 0), f"d={d}: some rows failed"
        assert np.array_equal(result.solve_lens.cpu().numpy(), truth), (
            f"d={d}: beam lens {result.solve_lens.cpu().tolist()} != "
            f"V*(state) {truth.tolist()}"
        )


def test_beam_width_k_never_worse_than_width_1():
    """Width=k can only solve at least as fast as width=1 (treating -1 as +∞).

    Width=k's frontier always contains width=1's top-1 survivor, so any
    layer where width=1 finds a solved child, width=k finds it too.
    """
    v_phys = _physical_v_star_2x2(max_depth=4)
    net = _VStarNet(v_phys)

    rng = torch.Generator().manual_seed(77)
    states, _ = random_scrambles(
        CUBE_2X2, batch_size=40, depth=4, generator=rng, prune_same_face=True
    )

    def _effective(lens: torch.Tensor) -> np.ndarray:
        a = lens.cpu().numpy().astype(np.float64)
        a[a < 0] = np.inf
        return a

    width_1 = beam_solve_batch(net, CUBE_2X2, states, beam_width=1, max_steps=8)
    for k in (4, 16):
        width_k = beam_solve_batch(net, CUBE_2X2, states, beam_width=k, max_steps=8)
        eff_k = _effective(width_k.solve_lens)
        eff_1 = _effective(width_1.solve_lens)
        assert np.all(eff_k <= eff_1), (
            f"width={k} produced row(s) worse than width=1: "
            f"{width_k.solve_lens.tolist()} vs {width_1.solve_lens.tolist()}"
        )


def test_max_steps_zero_marks_unsolved_failed():
    """max_steps=0: solved rows record 0; unsolved rows stay at -1."""
    net = _ConstantNet()
    rng = torch.Generator().manual_seed(7)
    scrambled, _ = random_scrambles(
        CUBE_2X2, batch_size=4, depth=2, generator=rng, prune_same_face=True
    )
    solved = CUBE_2X2.solved_state.unsqueeze(0).expand(2, -1)
    states = torch.cat([solved, scrambled], dim=0)

    result = beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=0)
    assert result.solve_lens.tolist() == [0, 0, -1, -1, -1, -1]
    assert result.n_expansions == 0


def test_does_not_mutate_input_states():
    v_phys = _physical_v_star_2x2(max_depth=3)
    net = _VStarNet(v_phys)
    rng = torch.Generator().manual_seed(11)
    states, _ = random_scrambles(
        CUBE_2X2, batch_size=6, depth=2, generator=rng, prune_same_face=True
    )
    original = states.clone()
    _ = beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=4)
    assert torch.equal(states, original)


def test_restores_train_mode_after_call():
    net = _ConstantNet()
    net.train()
    assert net.training is True
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(2, -1).clone()
    _ = beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=1)
    assert net.training is True

    net.eval()
    assert net.training is False
    _ = beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=1)
    assert net.training is False


def test_solve_paths_apply_to_solve_state():
    """Each emitted path, applied to the input state, reaches solved."""
    v_phys = _physical_v_star_2x2(max_depth=3)
    net = _VStarNet(v_phys)
    rng = torch.Generator().manual_seed(31)
    states, _ = random_scrambles(
        CUBE_2X2, batch_size=10, depth=3, generator=rng, prune_same_face=True
    )
    result = beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=6)
    assert torch.all(result.solve_lens >= 0)
    for i in range(states.shape[0]):
        path = result.solve_paths[i]
        assert len(path) == int(result.solve_lens[i].item())
        end = apply_move_sequence(states[i], path, CUBE_2X2)
        assert bool(is_solved(end, CUBE_2X2).item()), (
            f"row {i}: path {path} did not reach solved"
        )


def test_n_expansions_counts_per_layer_expansion():
    """``n_expansions`` accumulates ``beam_size × n_moves`` per inner step.

    Already-solved rows expand zero times. A depth-1 scramble with width=4
    expands once (1 parent × 12 = 12). Two such rows → 12 + 12 = 24, plus
    pre-solved rows contribute 0.
    """
    net = _ConstantNet()  # any net suffices — at depth 1 the solved child
    # is found in the first expansion regardless of scoring.
    rng = torch.Generator().manual_seed(5)
    scrambled, _ = random_scrambles(
        CUBE_2X2, batch_size=2, depth=1, generator=rng, prune_same_face=True
    )
    solved = CUBE_2X2.solved_state.unsqueeze(0).expand(1, -1)
    states = torch.cat([solved, scrambled], dim=0)
    result = beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=3)
    assert result.solve_lens.tolist() == [0, 1, 1]
    assert result.n_expansions == 12 + 12  # the two depth-1 scrambles only


def test_rejects_wrong_shape():
    import pytest

    net = _ConstantNet()
    bad = torch.zeros((4, 20), dtype=torch.int8)
    with pytest.raises(ValueError, match="states"):
        beam_solve_batch(net, CUBE_2X2, bad, beam_width=4, max_steps=1)


def test_rejects_negative_max_steps():
    import pytest

    net = _ConstantNet()
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(2, -1).clone()
    with pytest.raises(ValueError, match="max_steps"):
        beam_solve_batch(net, CUBE_2X2, states, beam_width=4, max_steps=-1)


def test_rejects_zero_beam_width():
    import pytest

    net = _ConstantNet()
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(2, -1).clone()
    with pytest.raises(ValueError, match="beam_width"):
        beam_solve_batch(net, CUBE_2X2, states, beam_width=0, max_steps=1)


def test_beam_solve_batch_n_invariant_to_split():
    """Cross-scramble batching is order/N-invariant.

    The C1 rewrite folds all N input scrambles into a single shared beam
    tensor of shape ``(N, B, S)``. Correctness requires that running the
    full corpus in one call produces the same per-row ``solve_lens`` as
    splitting it into chunks and concatenating — i.e. row i's outcome is
    independent of whether row j is also in the batch. This guards against
    subtle ordering bugs (e.g. a per-step reduction that accidentally
    couples scrambles together).
    """
    v_phys = _physical_v_star_2x2(max_depth=4)
    net = _VStarNet(v_phys)

    rng = torch.Generator().manual_seed(2026)
    # 100 scrambles drawn from a mix of depths to exercise both early-solve
    # and deeper-solve paths within the same call.
    parts = []
    for d in (1, 2, 3, 4):
        s, _ = random_scrambles(
            CUBE_2X2, batch_size=25, depth=d, generator=rng, prune_same_face=True
        )
        parts.append(s)
    # Add a couple of pre-solved rows to verify the already-solved short
    # circuit interacts correctly with the batched layout.
    solved = CUBE_2X2.solved_state.unsqueeze(0).expand(2, -1).clone()
    states = torch.cat([solved] + parts, dim=0)
    assert states.shape[0] == 102

    full = beam_solve_batch(
        net, CUBE_2X2, states, beam_width=4, max_steps=8
    )
    head = beam_solve_batch(
        net, CUBE_2X2, states[:50], beam_width=4, max_steps=8
    )
    tail = beam_solve_batch(
        net, CUBE_2X2, states[50:], beam_width=4, max_steps=8
    )

    split_lens = torch.cat([head.solve_lens, tail.solve_lens])
    assert split_lens.tolist() == full.solve_lens.tolist(), (
        "split solve_lens diverged from full-batch solve_lens — "
        "cross-scramble batching introduced row coupling"
    )

    # Paths must reach solved when applied (the per-row path is allowed to
    # differ between split and full runs only if both still solve in the
    # same number of moves; the length invariance is what we gate on).
    split_paths = head.solve_paths + tail.solve_paths
    for i in range(states.shape[0]):
        assert len(split_paths[i]) == int(split_lens[i].item())
        assert len(full.solve_paths[i]) == int(full.solve_lens[i].item())
