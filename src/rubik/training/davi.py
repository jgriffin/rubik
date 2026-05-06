"""DAVI step + Bellman-target computation.

Approximate value iteration for the cube cost-to-go:

    target(s) = min_a (1 + V_target(s_a))   with V_target(solved) := 0

The `(solved → 0)` clamp is the boundary condition — DAVI never sees
solved as a training input (`generate_adi_batch` samples depth ≥ 1)
but it appears as a *child* during target expansion, where this clamp
makes the recursion well-defined.

Per-step:
    1. Expand all ``n_moves`` children of each state via ``apply_all_moves``.
    2. Run ``V_target`` on the flat ``(B·n_moves, n_stickers)`` children;
       reshape back to ``(B, n_moves)``.
    3. Mask solved children to 0.
    4. ``targets = (1 + V_target(children)).min(dim=1)``.
    5. MSE loss against ``V_θ(states)``; backprop; optimizer step.

The training loop (logging, checkpointing, eval against V*) lives in
``experiments/davi-2x2/run.py`` (M5 commit 5). This module exposes only
the primitives: ``compute_targets``, ``davi_step``, ``sync_target``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rubik.cube.env import apply_all_moves, is_solved
from rubik.cube.spec import CubeSpec


def compute_targets(
    states: torch.Tensor,
    target_net: nn.Module,
    spec: CubeSpec,
) -> torch.Tensor:
    """Bellman target ``min_a (1 + V_target(child))`` with solved children clamped to 0.

    Args:
        states: ``(B, n_stickers)`` int8/int — non-solved input states.
        target_net: target network (eval mode recommended; gradients are
            disabled internally regardless).
        spec: cube spec.

    Returns:
        ``(B,)`` float32 Bellman target per state.
    """
    B = states.shape[0]
    children = apply_all_moves(states, spec)  # (B, n_moves, n_stickers)
    children_flat = children.reshape(B * spec.n_moves, spec.n_stickers)

    with torch.no_grad():
        child_v = target_net(children_flat).reshape(B, spec.n_moves)

    solved_mask = is_solved(children_flat, spec).reshape(B, spec.n_moves)
    child_v = torch.where(solved_mask, torch.zeros_like(child_v), child_v)
    return (1.0 + child_v).min(dim=1).values


def davi_step(
    net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    spec: CubeSpec,
) -> float:
    """Run one DAVI training step. Returns scalar loss."""
    targets = compute_targets(states, target_net, spec)
    pred = net(states)
    loss = F.mse_loss(pred, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def sync_target(net: nn.Module, target_net: nn.Module) -> None:
    """Copy ``net`` parameters into ``target_net`` (in-place, bit-exact)."""
    target_net.load_state_dict(net.state_dict())
