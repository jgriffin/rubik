"""Eval primitives for DAVI training: V* MAE + greedy-policy solve rate.

Two functions, intended to be imported by ``run.py``:

- ``eval_against_v_star`` — runs the network forward on the depth-stratified
  V* eval set and returns ``{val_mae, macro_mae, per_depth_mae,
  pred_mean, pred_std}``. macro-MAE is the fixed methodology metric: each
  depth contributes equally regardless of bucket size, so it can't be
  gamed by predicting the modal class.
- ``greedy_solve`` — for each test depth, generate fresh random scrambles
  and run the greedy policy ``argmin V_θ(child)`` from each scrambled
  state until solved or a budget is exhausted. Returns per-depth
  solve-rate + average solve length.

Both call ``net.eval()`` while running and restore ``net.train()`` on
exit so BatchNorm running stats are used for inference (matching the
policy that will run at deployment / search time).

The V* MAE logic is ported (not imported) from
``t1-capacity/supervised.py``'s ``_eval_val`` — same shape, but for the
depth-stratified eval set so per_depth_mae always covers the full
{1, ..., 14} range.
"""

from __future__ import annotations

import torch

from rubik.cube.env import apply_all_moves, is_solved, random_scrambles
from rubik.cube.spec import CubeSpec


@torch.no_grad()
def eval_against_v_star(
    net: torch.nn.Module,
    eval_states_dev: torch.Tensor,
    eval_depths_cpu: torch.Tensor,
    *,
    eval_batch_size: int = 1024,
) -> dict:
    """Eval the network on a depth-stratified V* eval set.

    Args:
        net: ValueNet (or any module mapping ``(B, n_stickers)`` -> ``(B,)``).
        eval_states_dev: ``(N, n_stickers)`` int tensor on the network's
            device (cached at run start).
        eval_depths_cpu: ``(N,)`` int tensor of optimal depths on CPU.
        eval_batch_size: chunk size for forward passes.

    Returns dict with:
        - ``val_mae`` (float): uniform mean over states.
        - ``macro_mae`` (float): uniform mean across per-depth MAEs.
        - ``per_depth_mae`` (dict[int, float]): MAE per depth label.
        - ``pred_mean`` (float), ``pred_std`` (float).
    """
    was_training = net.training
    net.eval()
    pred_chunks: list[torch.Tensor] = []
    n = eval_states_dev.shape[0]
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        preds = net(eval_states_dev[start:end])
        pred_chunks.append(preds.detach().cpu())
    if was_training:
        net.train()

    all_preds = torch.cat(pred_chunks)
    depths_cpu = eval_depths_cpu.detach().cpu().to(torch.float32)
    abs_errors = (all_preds - depths_cpu).abs()

    val_mae = float(abs_errors.mean().item())
    pred_mean = float(all_preds.mean().item())
    pred_std = float(all_preds.std(unbiased=False).item())

    depth_ints = eval_depths_cpu.detach().cpu().to(torch.int64)
    per_depth_mae: dict[int, float] = {}
    for d in torch.unique(depth_ints).tolist():
        mask = depth_ints == d
        per_depth_mae[int(d)] = float(abs_errors[mask].mean().item())
    macro_mae = float(sum(per_depth_mae.values()) / len(per_depth_mae))

    return {
        "val_mae": val_mae,
        "macro_mae": macro_mae,
        "per_depth_mae": per_depth_mae,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    }


@torch.no_grad()
def greedy_solve(
    net: torch.nn.Module,
    spec: CubeSpec,
    *,
    depth_budget_factor: int = 2,
    n_per_depth: int = 50,
    depths: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13),
    generator: torch.Generator | None = None,
) -> dict:
    """Greedy policy solve-rate eval: pick ``argmin V_θ(child)`` per step.

    For each test depth ``d``:
        1. Generate ``n_per_depth`` random scrambles of length ``d`` from
           solved.
        2. From each scrambled state, repeatedly expand all 12 children,
           score them with ``net``, take the move whose child has minimum
           V_θ. Stop when the chosen child is solved (record the move
           count, including this last one) or when ``depth_budget_factor *
           d`` moves have been taken without solving (mark as failure).
        3. Solve length is the number of moves applied to reach solved.

    The inner loop is vectorized across the ``n_per_depth`` scrambles per
    test depth — one forward pass per move-step over the unsolved
    remaining states' children, not per-row.

    Returns a flat dict::

        {
            "solve_rate_d{d}": float (in [0, 1]),
            "avg_solve_len_d{d}": float | None,  # None iff no solves
            ...
        }
    """
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()

    out: dict = {}
    for d in depths:
        budget = depth_budget_factor * d
        states, _ = random_scrambles(
            spec,
            batch_size=n_per_depth,
            depth=d,
            generator=generator,
            prune_same_face=True,
        )
        states = states.to(device)

        n = n_per_depth
        # Active mask: True for rows still solving; False once solved
        # (stops further updates and freezes the recorded length).
        active = torch.ones(n, dtype=torch.bool, device=device)
        solve_lens = torch.full((n,), -1, dtype=torch.int64, device=device)
        # Sanity: a depth-d scramble already at solved would be unusual
        # (would mean prune_same_face produced a no-op-net sequence at
        # this depth). Guard anyway so the recorded length is 0 in that
        # case rather than running the loop on solved states.
        already_solved = is_solved(states, spec)
        if already_solved.any():
            solve_lens[already_solved] = 0
            active = active & ~already_solved

        for step_idx in range(budget):
            if not active.any():
                break
            active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
            active_states = states[active_idx]  # (A, n_stickers)

            children = apply_all_moves(active_states, spec)
            # children: (A, n_moves, n_stickers)
            A = active_states.shape[0]
            children_flat = children.reshape(A * spec.n_moves, spec.n_stickers)
            child_v = net(children_flat).reshape(A, spec.n_moves)
            # Solved children should be picked greedily; clamp to -inf in
            # the wrong direction won't help — instead: prefer-solved by
            # forcing solved children to a very-low score so argmin picks
            # them. This matches the "solved == 0" boundary used in DAVI
            # targets and ensures policy is consistent with the value
            # estimate when V_θ has noise on small values.
            solved_children = is_solved(children_flat, spec).reshape(A, spec.n_moves)
            # Use a value smaller than any plausible V_θ on non-solved
            # so argmin always picks a solved child when one exists.
            child_v = torch.where(
                solved_children, torch.full_like(child_v, -1e9), child_v
            )
            best_move = child_v.argmin(dim=1)  # (A,)
            new_states = children[
                torch.arange(A, device=device), best_move
            ]  # (A, n_stickers)
            states[active_idx] = new_states

            # Detect newly-solved rows.
            now_solved = is_solved(new_states, spec)
            if now_solved.any():
                solved_active_idx = active_idx[now_solved]
                # +1 because we just applied move (step_idx + 1) total moves.
                solve_lens[solved_active_idx] = step_idx + 1
                active[solved_active_idx] = False

        solved_mask_cpu = (solve_lens >= 0).cpu()
        n_solved = int(solved_mask_cpu.sum().item())
        solve_rate = n_solved / n
        if n_solved > 0:
            avg_len = float(solve_lens[solve_lens >= 0].float().mean().item())
        else:
            avg_len = None

        out[f"solve_rate_d{d}"] = solve_rate
        out[f"avg_solve_len_d{d}"] = avg_len

    if was_training:
        net.train()
    return out
