"""Lean three-function eval API for the 3x3 DAVI experiment.

Replaces the older 2x2-shape ``eval_against_v_star`` + ``greedy_solve``
pair with three roles, designed around the realities of the 3x3 setting:

- The full V* oracle is unreachable (~4.3×10¹⁹ states) — we have a
  bounded oracle to depth K=6 only (see
  ``src/rubik/oracle/v_star_bounded_3x3.py``). MAE-vs-V* is therefore a
  partial (d≤K) signal, not the full eval as on 2x2.
- Beam search is the production search policy; greedy solve is just
  ``beam_solve_batch(beam_width=1)`` and gets no separate code path.
- Beam evaluation is too expensive to run on every training sync but
  cheap as a one-shot post-training step. Forward-pass MAE is cheap
  enough to run live every sync and feeds the early-stop signal.

The three functions:

- ``value_eval(net, spec, oracle_dict, *, ...)`` — LIVE during training,
  one forward pass on a freshly-generated random-walk eval set per call
  (deterministic from ``seed`` — the generator is re-seeded each call so
  back-to-back invocations evaluate the network on identical states).
  Returns per-walk-depth predicted-V* stats (the "forever metric" that
  survives K-bounding) plus per-V* MAE for the d≤K subset where the
  oracle has ground truth, plus a uniform ``macro_v_star_mae`` scalar —
  the planned early-stop driver. No search.

- ``beam_eval_walk(net, spec, *, ...)`` — POST-TRAINING capability eval
  on random-walk states across a depth grid. Forever-eval: scales to
  any walk-depth without an oracle. Greedy is recovered by
  ``beam_width=1``.

- ``beam_eval_v_star(net, spec, oracle_arrays, *, ...)`` — POST-TRAINING
  capability eval on V*-stratified states (sampled per V*-layer from the
  bounded oracle) plus per-V* prediction MAE. Disappears when we go
  past the bounded oracle's K, so it's the temporary ground-truth
  capability lens while it exists.

All three call ``net.eval()`` while running and restore the prior
training mode on exit, matching ``rubik.solve.greedy_solve_batch`` and
``rubik.search.beam_solve_batch``. See plan ``plans/m8-3x3-davi.md``
§P1c for the design rationale.
"""

from __future__ import annotations

import numpy as np
import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CubeSpec
from rubik.oracle.v_star_bounded_3x3 import (
    lookup_v_star_bounded_3x3_batch,
    sample_states_at_v_star_3x3,
)
from rubik.search import beam_solve_batch
from rubik.solve import summarize_solve_lens


@torch.no_grad()
def value_eval(
    net: torch.nn.Module,
    spec: CubeSpec,
    oracle_dict: dict[bytes, int],
    *,
    n_per_walk_depth: int = 100,
    walk_depths: tuple[int, ...] = tuple(range(1, 15)),
    eval_batch_size: int = 1024,
    seed: int = 0,
) -> dict:
    """Live forward-pass eval: per-walk-depth pred stats + per-V* MAE.

    Generates ``n_per_walk_depth`` fresh random-walk scrambles per
    walk-depth in ``walk_depths`` (deterministic given ``seed`` —
    constructed fresh each call so back-to-back invocations evaluate the
    network on **identical** states; this is what makes the eval
    trajectory comparable across training steps), forwards everything
    through ``net`` in a single batched pass on the
    network's device, and reports two views of the same predictions:

    - **per_walk_depth/d{d}/pred_{mean,std}** — predicted-V* statistics
      bucketed by the random-walk length used to generate the scramble.
      The "forever metric": works at any depth, no oracle needed. Walk
      redundancy means this is biased toward smaller true V* than ``d``.
    - **per_walk_depth/{shallow,mid,deep}/pred_{mean,std}** — banded
      aggregates over the per-d keys (Shallow d=1..5 / Mid d=6..10 /
      Deep d=11..14, matching the project's 5/5/4 small-multiples
      convention). At-a-glance trend reads. ``pred_std`` aggregate is
      the mean of per-d stds, NOT pooled std — simpler interpretation.
    - **v_star_mae/d{d}** — true MAE against ``oracle_dict`` for the
      subset of walk endpoints whose true V* is in the bounded oracle.
      Endpoints with V* > K (oracle returns ``-1`` sentinel) are
      excluded from this view. Layers with zero post-mask samples are
      omitted from the dict.
    - **macro_v_star_mae** — uniform mean across the populated
      ``v_star_mae/d*`` keys for ``d ≥ 1``. **V*=0 (solved state) is
      excluded** because its MAE captures terminal-value calibration
      (the network's prediction at solved vs. the constant 0), a
      different quality from "V* prediction error on non-trivial
      scrambled states." Walks rarely land on V*=0 (walk redundancy at
      length ≥ 6+) and including it makes the headline scalar
      inconsistent eval-to-eval. The early-stop driver per the plan.

    Args:
        net: ValueNet (or any module mapping ``(B, n_stickers)`` -> ``(B,)``).
        spec: cube spec — must be 3x3 since the oracle is 3x3-only.
        oracle_dict: bounded V* dict from
            ``load_v_star_bounded_3x3``. Lookup with
            ``lookup_v_star_bounded_3x3_batch(..., missing=-1)``.
        n_per_walk_depth: number of fresh scrambles per walk-depth bin.
        walk_depths: walk-depth grid. Defaults to 1..14, well past
            the K=6 oracle bound — depths > K still produce valid
            ``per_walk_depth`` stats but contribute nothing to
            ``v_star_mae``.
        eval_batch_size: chunk size for forward passes (memory bound).
        seed: anchor for the per-call torch generator. Pass the SAME
            seed at every eval call so the eval set is fixed across
            training steps — what changes between calls is the network
            weights, not the eval distribution. Pass ``config.seed +
            17`` (or similar) from callers; offset from the training-data
            seed to keep eval samples independent of any training batch.

    Returns:
        Flat dict with the keys described above plus overall ``pred_mean``
        and ``pred_std`` aggregates across all walk depths.
    """
    n_stickers = spec.n_stickers
    device = next(net.parameters()).device

    # Construct the eval generator FRESH per call (re-seeded from ``seed``).
    # The point of this function is to evaluate the same network on the same
    # eval set, varying only the network's parameters across training steps —
    # so the generator must be re-seeded each call rather than advanced from
    # a single shared instance. (Using a single shared generator is the H4
    # bug from the smoke run: per-V* MAE bounced eval-to-eval because the
    # eval distribution drifted with the generator state.)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    # Generate all walks first; concat into one (D * n_per_walk_depth, n_stickers)
    # tensor so we get one batched forward pass instead of D separate passes.
    walk_states_list: list[torch.Tensor] = []
    walk_depth_labels: list[int] = []
    for d in walk_depths:
        states, _ = random_scrambles(
            spec,
            batch_size=n_per_walk_depth,
            depth=d,
            generator=generator,
            prune_same_face=True,
        )
        walk_states_list.append(states)
        walk_depth_labels.extend([int(d)] * n_per_walk_depth)
    all_states = torch.cat(walk_states_list, dim=0)  # (D*N, n_stickers) int8
    walk_depth_arr = np.asarray(walk_depth_labels, dtype=np.int64)

    # Single batched forward pass on device (chunked at eval_batch_size).
    was_training = net.training
    net.eval()
    pred_chunks: list[torch.Tensor] = []
    states_dev = all_states.to(device)
    n_total = states_dev.shape[0]
    for start in range(0, n_total, eval_batch_size):
        end = min(start + eval_batch_size, n_total)
        preds = net(states_dev[start:end])
        pred_chunks.append(preds.detach().cpu())
    if was_training:
        net.train()
    all_preds = torch.cat(pred_chunks).to(torch.float32)  # (D*N,)

    out: dict = {}

    # Per-walk-depth predicted-V* stats — forever metric, no oracle needed.
    per_walk_depth_means: dict[int, float] = {}
    per_walk_depth_stds: dict[int, float] = {}
    for d in walk_depths:
        mask = walk_depth_arr == int(d)
        bucket = all_preds[torch.from_numpy(mask)]
        m = float(bucket.mean().item())
        # std with unbiased=False matches eval_against_v_star's 2x2 convention
        # and avoids NaN at n=1.
        s = float(bucket.std(unbiased=False).item())
        per_walk_depth_means[int(d)] = m
        per_walk_depth_stds[int(d)] = s
        out[f"per_walk_depth/d{int(d)}/pred_mean"] = m
        out[f"per_walk_depth/d{int(d)}/pred_std"] = s

    # Banded aggregates over per_walk_depth for at-a-glance trend reads.
    # Decision: emit these from `value_eval` (option a) rather than only in
    # the WandbAdapter, so the JSONL captures them too and downstream
    # analyzers can read them without recomputation. The bands match the
    # project's 5/5/4 small-multiples convention (Shallow d=1..5 / Mid
    # d=6..10 / Deep d=11..14) — see render_error_trajectories.py.
    # Aggregate of std is the mean-of-stds (NOT pooled std) — simpler
    # interpretation, fine for at-a-glance.
    _BANDS: tuple[tuple[str, range], ...] = (
        ("shallow", range(1, 6)),
        ("mid", range(6, 11)),
        ("deep", range(11, 15)),
    )
    for band_name, band_range in _BANDS:
        band_means = [
            per_walk_depth_means[d] for d in band_range if d in per_walk_depth_means
        ]
        band_stds = [
            per_walk_depth_stds[d] for d in band_range if d in per_walk_depth_stds
        ]
        if band_means:
            out[f"per_walk_depth/{band_name}/pred_mean"] = float(
                sum(band_means) / len(band_means)
            )
            out[f"per_walk_depth/{band_name}/pred_std"] = float(
                sum(band_stds) / len(band_stds)
            )

    # Per-V* MAE for the d≤K subset where the oracle has ground truth.
    # lookup_v_star_bounded_3x3_batch consumes (N, 54) int8 numpy; convert
    # the all-states tensor exactly once.
    states_np = all_states.numpy()
    if states_np.shape[1] != n_stickers:
        # Defensive: should never trip given the spec contract.
        raise ValueError(
            f"unexpected n_stickers in eval states: got {states_np.shape[1]}, "
            f"expected {n_stickers}"
        )
    v_star_lookup = lookup_v_star_bounded_3x3_batch(
        states_np.astype(np.int8, copy=False), oracle_dict, missing=-1
    )
    valid_mask = v_star_lookup != -1
    v_star_mae: dict[int, float] = {}
    if valid_mask.any():
        valid_v_star = v_star_lookup[valid_mask].astype(np.int64)
        valid_preds = all_preds.numpy()[valid_mask].astype(np.float32)
        abs_err = np.abs(valid_preds - valid_v_star.astype(np.float32))
        # Discover layers dynamically: don't hardcode K=6 — if the oracle
        # is rebuilt at K=8 later this code keeps working unchanged.
        for v in np.unique(valid_v_star).tolist():
            layer_mask = valid_v_star == v
            if not layer_mask.any():
                continue
            v_star_mae[int(v)] = float(abs_err[layer_mask].mean())

    for v, mae in v_star_mae.items():
        out[f"v_star_mae/d{int(v)}"] = mae
    # Macro excludes V*=0 (the solved state). V*=0's MAE measures terminal-value
    # calibration — a different quality from "V* prediction error on non-trivial
    # scrambled states." Including it would also make the macro inconsistent
    # eval-to-eval, since V*=0 only populates when a walk happens to return to
    # solved (rare, walk-depth-dependent) — its presence/absence would jitter
    # the headline scalar that drives early-stop.
    non_terminal_maes = [mae for v, mae in v_star_mae.items() if v > 0]
    if non_terminal_maes:
        out["macro_v_star_mae"] = float(sum(non_terminal_maes) / len(non_terminal_maes))
    else:
        out["macro_v_star_mae"] = float("nan")

    out["pred_mean"] = float(all_preds.mean().item())
    out["pred_std"] = float(all_preds.std(unbiased=False).item())
    return out


def beam_eval_walk(
    net: torch.nn.Module,
    spec: CubeSpec,
    *,
    n_per_depth: int = 100,
    walk_depths: tuple[int, ...] = tuple(range(1, 15)),
    beam_width: int = 256,
    depth_budget_factor: int = 2,
    generator: torch.Generator | None = None,
) -> dict:
    """Post-training random-walk capability eval via beam search.

    For each walk-depth ``d`` in ``walk_depths``:
        1. Generate ``n_per_depth`` length-``d`` random scrambles.
        2. Solve via ``beam_solve_batch(beam_width=beam_width,
           max_steps=depth_budget_factor * d)``.
        3. Roll up to ``solve_rate`` and ``avg_solve_len``.

    Greedy is ``beam_width=1`` — no separate greedy implementation. The
    forever capability eval: extends to any walk-depth without an oracle.

    Returns flat dict::

        {
            "solve_rate_d{d}": float in [0, 1],
            "avg_solve_len_d{d}": float | None,  # None iff zero solves
            ...
        }
    """
    out: dict = {}
    for d in walk_depths:
        states, _ = random_scrambles(
            spec,
            batch_size=n_per_depth,
            depth=d,
            generator=generator,
            prune_same_face=True,
        )
        result = beam_solve_batch(
            net,
            spec,
            states,
            beam_width=beam_width,
            max_steps=depth_budget_factor * d,
        )
        s = summarize_solve_lens(result.solve_lens)
        out[f"solve_rate_d{int(d)}"] = s["solve_rate"]
        out[f"avg_solve_len_d{int(d)}"] = s["avg_solve_len"]
    return out


def beam_eval_v_star(
    net: torch.nn.Module,
    spec: CubeSpec,
    oracle_arrays: tuple[np.ndarray, np.ndarray],
    *,
    n_per_layer: int = 200,
    beam_width: int = 256,
    depth_budget_factor: int = 2,
    max_v_star: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Post-training V*-stratified capability eval + per-V* prediction MAE.

    For each V* layer present in the bounded oracle (``v >= 1``; V*=0 is
    the lone solved state and skipped by design — beam-solving "solved"
    is a no-op), sample ``n_per_layer`` states from that layer, run beam
    search with ``max_steps = depth_budget_factor * v``, and report
    solve rate, mean solve length, and per-V* prediction MAE.

    The ``max_v_star`` arg clips the upper layer — useful for tests
    where running 6 separate beam searches at K=6 oracle is too slow.

    Args:
        net: value net.
        spec: cube spec (3x3).
        oracle_arrays: ``(states, depths)`` from
            ``load_v_star_bounded_3x3_arrays``.
        n_per_layer: number of states sampled per V* layer.
        beam_width: beam width for ``beam_solve_batch``.
        depth_budget_factor: search budget multiplier — ``max_steps =
            depth_budget_factor * v``.
        max_v_star: if not None, only evaluate layers with ``v <= max_v_star``.
        rng: numpy random generator. If None, ``np.random.default_rng(0)``
            is used for determinism.

    Returns flat dict::

        {
            "solve_rate_v{v}": float,
            "avg_solve_len_v{v}": float | None,
            "mae_v{v}": float,
            ...
        }

    For each V* layer ``v`` with at least one sample.
    """
    states_arr, depths_arr = oracle_arrays
    if rng is None:
        rng = np.random.default_rng(0)

    layers = sorted(int(v) for v in np.unique(depths_arr).tolist() if v >= 1)
    if max_v_star is not None:
        layers = [v for v in layers if v <= max_v_star]

    device = next(net.parameters()).device
    out: dict = {}

    for v in layers:
        sampled = sample_states_at_v_star_3x3(
            states_arr,
            depths_arr,
            target_v_star=v,
            n=n_per_layer,
            rng=rng,
        )
        # int8 numpy -> torch; beam_solve_batch handles device migration.
        states_t = torch.from_numpy(sampled.astype(np.int8, copy=False))

        # Per-V* prediction MAE — separate forward pass against the same
        # sampled states so we don't have to fish predictions out of the
        # search internals. Cost is one forward of n_per_layer rows.
        was_training = net.training
        net.eval()
        with torch.no_grad():
            preds = net(states_t.to(device)).detach().cpu().to(torch.float32)
        if was_training:
            net.train()
        mae = float(torch.abs(preds - float(v)).mean().item())

        result = beam_solve_batch(
            net,
            spec,
            states_t,
            beam_width=beam_width,
            max_steps=depth_budget_factor * v,
        )
        s = summarize_solve_lens(result.solve_lens)
        out[f"solve_rate_v{int(v)}"] = s["solve_rate"]
        out[f"avg_solve_len_v{int(v)}"] = s["avg_solve_len"]
        out[f"mae_v{int(v)}"] = mae

    return out
