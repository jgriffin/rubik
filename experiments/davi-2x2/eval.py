"""Eval primitives for DAVI training: V* MAE + greedy-policy solve rate.

Two functions, intended to be imported by ``run.py``:

- ``eval_against_v_star`` — runs the network forward on the depth-stratified
  V* eval set and returns ``{val_mae, macro_mae, per_depth_mae,
  pred_mean, pred_std}``. macro-MAE is the fixed methodology metric: each
  depth contributes equally regardless of bucket size, so it can't be
  gamed by predicting the modal class.
- ``greedy_solve`` — per-depth wrapper over ``rubik.solve.greedy_solve_batch``:
  for each test depth, generate fresh random scrambles, solve under a
  ``2 * depth`` move budget, and roll up to per-depth ``solve_rate`` /
  ``avg_solve_len``. The greedy primitive itself lives in ``rubik.solve``
  so any future net can be eval'd through the same code path.

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

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CubeSpec
from rubik.solve import greedy_solve_batch, summarize_solve_lens


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


def greedy_solve(
    net: torch.nn.Module,
    spec: CubeSpec,
    *,
    depth_budget_factor: int = 2,
    n_per_depth: int = 50,
    depths: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13),
    generator: torch.Generator | None = None,
) -> dict:
    """Per-depth wrapper around ``rubik.solve.greedy_solve_batch``.

    For each test depth ``d``:
        1. Generate ``n_per_depth`` random scrambles of length ``d``.
        2. Run greedy ``argmin V_θ(child)`` with ``max_steps =
           depth_budget_factor * d``.
        3. Roll the per-attempt ``solve_lens`` into ``solve_rate`` and
           ``avg_solve_len``.

    Returns a flat dict::

        {
            "solve_rate_d{d}": float (in [0, 1]),
            "avg_solve_len_d{d}": float | None,  # None iff no solves
            ...
        }
    """
    out: dict = {}
    for d in depths:
        states, _ = random_scrambles(
            spec,
            batch_size=n_per_depth,
            depth=d,
            generator=generator,
            prune_same_face=True,
        )
        solve_lens = greedy_solve_batch(
            net, spec, states, max_steps=depth_budget_factor * d
        )
        s = summarize_solve_lens(solve_lens)
        out[f"solve_rate_d{d}"] = s["solve_rate"]
        out[f"avg_solve_len_d{d}"] = s["avg_solve_len"]
    return out
