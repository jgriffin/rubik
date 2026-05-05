"""Greedy ``argmin V_θ(child)`` policy: per-attempt solve lengths.

Production path for any "evaluate this value net" workflow. Takes pre-scrambled
states (caller controls the scramble distribution / seeding), runs the greedy
policy until each row is solved or its per-row budget is exhausted, returns
the raw ``(N,)`` solve-length tensor — ``-1`` for rows that didn't reach
solved within budget, otherwise the move count to solved.

Solved children get a ``-inf``-flavoured sentinel score so ``argmin`` always
picks a solved child when one exists. This matches the DAVI target boundary
where ``V*(solved) = 0`` and prevents value-noise from dominating last-step
choice when the trained net's estimates near solved are jittery.

The net is set to ``eval()`` while solving and restored to its prior
train/eval state on exit so BN running stats (not batch stats) drive the
forward pass — the policy that runs at deployment / search time.
"""

from __future__ import annotations

import torch

from rubik.cube.env import apply_all_moves, is_solved
from rubik.cube.spec import CubeSpec


@torch.no_grad()
def greedy_solve_batch(
    net: torch.nn.Module,
    spec: CubeSpec,
    states: torch.Tensor,
    *,
    max_steps: int,
) -> torch.Tensor:
    """Greedy solve from each row in ``states``; returns ``(N,)`` solve lengths.

    Args:
        net: Value net mapping ``(B, n_stickers)`` -> ``(B,)``.
        spec: Cube spec (move set, stickers, solved state).
        states: ``(N, n_stickers)`` int tensor. Not mutated — a copy is made
            on the net's device before solving.
        max_steps: Per-row move budget. Rows still unsolved after this many
            steps are recorded as ``-1``.

    Returns:
        ``(N,)`` int64 tensor on the net's device. ``-1`` = not solved within
        budget; ``0`` = already solved on entry; ``k > 0`` = solved after ``k``
        moves.
    """
    if states.ndim != 2 or states.shape[1] != spec.n_stickers:
        raise ValueError(
            f"expected (N, {spec.n_stickers}) states; got shape {tuple(states.shape)}"
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0; got {max_steps}")

    device = next(net.parameters()).device
    states = states.to(device).clone()

    n = states.shape[0]
    was_training = net.training
    net.eval()

    active = torch.ones(n, dtype=torch.bool, device=device)
    solve_lens = torch.full((n,), -1, dtype=torch.int64, device=device)

    already = is_solved(states, spec)
    if already.any():
        solve_lens[already] = 0
        active = active & ~already

    try:
        for step_idx in range(max_steps):
            if not active.any():
                break
            active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
            active_states = states[active_idx]  # (A, n_stickers)
            children = apply_all_moves(active_states, spec)
            a = active_states.shape[0]
            children_flat = children.reshape(a * spec.n_moves, spec.n_stickers)
            child_v = net(children_flat).reshape(a, spec.n_moves)
            solved_children = is_solved(children_flat, spec).reshape(a, spec.n_moves)
            child_v = torch.where(
                solved_children, torch.full_like(child_v, -1e9), child_v
            )
            best_move = child_v.argmin(dim=1)  # (A,)
            new_states = children[torch.arange(a, device=device), best_move]
            states[active_idx] = new_states

            now_solved = is_solved(new_states, spec)
            if now_solved.any():
                solved_active_idx = active_idx[now_solved]
                solve_lens[solved_active_idx] = step_idx + 1
                active[solved_active_idx] = False
    finally:
        if was_training:
            net.train()

    return solve_lens


def summarize_solve_lens(solve_lens: torch.Tensor) -> dict:
    """Summary stats from per-attempt solve_lens.

    Args:
        solve_lens: ``(N,)`` int tensor; ``-1`` entries are failed attempts.

    Returns:
        ``{n, n_solved, solve_rate, avg_solve_len}``. ``avg_solve_len`` is
        ``None`` iff no attempt solved (otherwise mean over solved-only).
    """
    cpu = solve_lens.detach().cpu()
    n = int(cpu.numel())
    solved_mask = cpu >= 0
    n_solved = int(solved_mask.sum().item())
    solve_rate = n_solved / n if n > 0 else 0.0
    if n_solved > 0:
        avg = float(cpu[solved_mask].to(torch.float32).mean().item())
    else:
        avg = None
    return {
        "n": n,
        "n_solved": n_solved,
        "solve_rate": solve_rate,
        "avg_solve_len": avg,
    }
