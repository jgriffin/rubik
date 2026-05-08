"""Beam search policy: V_θ-scored batched search with within-beam dedup.

Per-state batched beam search using V_θ as the scoring function. The beam
holds the K best-scoring frontier states; at each step every beam slot is
expanded over all moves, scored, deduplicated by raw byte equality, and the
top-K survivors form the next frontier. The loop runs to completion for
``max(max_steps_per_state)`` iterations regardless of who solved when —
solved rows have their first-solve step recorded and continue expanding
alongside the unsolved rows so the tensor shapes stay rectangular.

Within-beam dedup uses raw-bytes equality (``state.tobytes()``), not the
24-rotation orbit canonicalization in ``rubik.oracle.v_star_2x2``. The orbit
canonicalizer is correct for V* lookup but wrong for path-tracked search:
collapsing rotation-equivalent states would emit moves that don't reach the
canonical representative when applied to the actual input cube. Raw bytes is
exact equality and faster.

**Cross-scramble batching (M8 C1).** All ``N`` input scrambles share one
beam tensor of shape ``(N, beam_width, n_stickers)``; per-step expansion
becomes a single ``apply_all_moves`` over ``N * beam_width`` parents and a
single ``net(...)`` forward of batch size ``N * beam_width * n_moves``.

**On-device-hash + CPU dedup (M8 C3).** The pre-C3 dedup forced a
per-step ``.cpu().numpy()`` of ``(N, B*n_moves, n_stickers)`` int8
children (~8MB at width=128) plus ``(N, B*n_moves)`` float scores, then
ran a Python ``dict[bytes(...)]`` loop hashing 54-byte keys. C3 keeps
the dedup *logic* on the host but replaces the input: ``state_hash``
runs on-device and produces an int64 hash per child, so the per-step
host transfer shrinks to ``(N, B*n_moves)`` int64 + ``(N, B*n_moves)``
float32 (~1.2MB at width=128, ~7× less data). The host-side dedup uses
``numpy.unique(..., return_inverse=True)`` per row + ``np.minimum.at``
for the min-V scatter, replacing the Python dict-of-bytes with C-level
NumPy ops.

**On-device top-k (M8 C4).** The C3 step's per-row CPU
``sorted(zip(min_v, rep_idx))[:beam_width]`` is replaced by a single
``torch.topk(dense_v, k=beam_width, largest=False, dim=1)`` on-device.
The CPU dedup pass now produces only an ``is_winner`` boolean mask of
shape ``(N, B*n_moves)`` (True at the smallest-idx representative of
each hash class, False elsewhere), pushed to GPU. ``dense_v`` is
``child_v`` with non-winners replaced by ``+inf``; topk over it returns
``(N, beam_width)`` indices directly into the ``B*n_moves`` flat
layout — exactly what the gather needs. Eliminates the per-step
``child_v.cpu()`` transfer (no longer read on host) and the
``np.empty + per-row fill + from_numpy + .to(device)`` round-trip for
``next_flat_idxs``.

**Short-beam handling.** When fewer than ``beam_width`` unique classes
exist (e.g. step 0 has 1 parent, 12 children → with beam_width=16
the topk fills 12 finite-V slots and 4 ``+inf`` slots), the ``+inf``
slots fall on whichever non-winner indices torch.topk picks among the
ties at ``+inf``. Those slots point to **real** cube states (children
of a real parent), just duplicates of states that already won earlier
in the topk output. The next-step dedup collapses them naturally —
no sentinel state, no mask needed. This matches the pre-C4 "pad by
repeating the first survivor" behavior in spirit (filling the beam
with redundant entries that get deduped next step).

**Tied values.** ``torch.topk`` does not guarantee a stable tiebreak
across runs. Within a hash class C3 picks the smallest within-row idx
as the representative (deterministic). *Across* classes, when two
representatives have identical ``min_v``, ``torch.topk`` may pick
either order. In practice tied-V across distinct states is rare with
trained float32 nets, but the choice is benign for aggregate solve
metrics (the C5 equivalence gate is solve_rate within ±10pp, well
above any tiebreak shuffle effect).

**Why not on-device dedup?** ``torch.unique`` and ``torch.sort`` on
MPS are broken for full-range int64 — they only consider the low 32
bits, collapsing distinct hashes whose high bits differ. Verified at
C3 implementation time:
``torch.unique(torch.tensor([1, 1<<32], device='mps')) == [1]``.
Until that lands upstream the dedup logic runs on the host; the hash
itself and the topk over float32 values both run on-device.

**Run-to-completion (M8 C2).** The early-exit-when-all-solved check
(``done_mask.all().item()`` per step) was removed in C2. First-solve
tracking (``solved_at_step``, ``solved_flat_idx``) is kept on-device and
updated each step via ``torch.where`` against an ``is_first_solve_now``
mask; expansions for already-solved or out-of-budget rows are excluded
from the ``n_expansions`` accounting via an on-device active-row count
summed once at loop end. The host-side path-reconstruction read is
deferred to a single ``.cpu()`` call after the loop.

**Smart early-exit (M8 C5a).** C2's run-to-completion choice was
correct *if* the post-loop dedup/topk path was the bottleneck
(it was at C2, when the dedup loop ran on host-side bytes). After
C3 + C4 the dedup is cheap and the ``net(...)`` forward dominates
(~98% of width=128 wall in profile data). Re-introducing early-exit
on the *active-row* condition — break the step loop the moment every
scramble has either solved or exhausted its budget — saves the entire
per-step forward for any cell that converges before its budget. The
``.all().item()`` per-step sync is the same one C2 deleted; it costs
~1-5ms × ~max_budget steps per cell and is dwarfed by the saved
forward (one width=128 forward over 153k rows is ~7-8s on this MPS
device, so a single skipped step pays for hundreds of ``.item()``
syncs). ``solved_at_step`` semantics unchanged: it's still "first
step at which any beam slot of scramble i emitted goal", and the
post-loop ``solved_within_budget`` gate is unchanged. The break is
correctness-preserving because every active row has either solved
already (work won't change) or exceeded its budget (won't be
credited as a solve regardless of further steps).

**Row compaction (M8 C5b).** C5a still ran every per-step op
(forward, dedup, topk) over the full ``(N, B*M, ...)`` shape even
when most rows had finished. C5b carries an explicit
``active_idx`` of shape ``(N_active,)`` and slices the beam tensor
down to ``(N_active, B, S)`` each step. The forward batch is
``N_active * B * M`` rather than ``N * B * M``, so a cell where rows
finish on average at step 3 with beam_width=128 sees the per-step
forward shrink to ~3% of the full shape by step 4, and the cell-wall
collapses from "max_steps × full forward" to "Σ_step (n_active(step) ×
forward)". The ``nonzero()`` to derive next-step active set is one
host sync per step (same magnitude as C5a's ``.all().item()``).
Done-row state is flushed to the global ``solved_at_step`` /
``solved_flat_idx`` tensors before the row drops out; backpointers
are recorded per-step keyed by scramble_idx so ``_walk_back`` looks
up exactly the rows it needs at exactly the steps they were alive.
The full ``(N, B, S)`` beam tensor is never materialized after
step 0 — only ``active_beam`` exists, which is a strict subset of
the rows that were alive at the start of search.

**Per-state budgets (M8 C2).** ``max_steps_per_state`` is an optional
``(N,)`` int tensor letting each scramble set its own step budget. The
loop runs ``int(max_steps_per_state.max())`` iterations; per-row solve
verdicts gate at the row's own budget at the end. Callers passing the
scalar ``max_steps`` get the prior uniform-budget behavior.

The net is set to ``eval()`` while solving and restored to its prior
train/eval state on exit so BN running stats drive the forward pass — the
policy that runs at deployment / search time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rubik.cube.env import apply_all_moves, is_solved
from rubik.cube.spec import CubeSpec
from rubik.search.state_hash import state_hash


@dataclass(frozen=True)
class BeamSearchResult:
    """Outputs of ``beam_solve_batch``.

    Attributes:
        solve_lens: ``(N,)`` int64 on the net's device. ``-1`` = not solved
            within budget; ``0`` = already solved on entry; ``k > 0`` =
            solved after ``k`` moves.
        solve_paths: List of length ``N``. Entry ``i`` is the move-index
            sequence that solves row ``i`` (``len == solve_lens[i]``); empty
            list for failed rows.
        n_expansions: Total node expansions across all parents — diagnostic
            counter for understanding search effort vs. width.
    """

    solve_lens: torch.Tensor
    solve_paths: list[list[int]]
    n_expansions: int


@torch.no_grad()
def beam_solve_batch(
    net: torch.nn.Module,
    spec: CubeSpec,
    states: torch.Tensor,
    *,
    beam_width: int,
    max_steps: int | None = None,
    max_steps_per_state: torch.Tensor | None = None,
) -> BeamSearchResult:
    """Beam-search solve from each row in ``states``.

    All ``N`` input scrambles run on ONE shared beam tensor of shape
    ``(N, beam_width, n_stickers)``; the per-step net forward is a single
    call of batch size ``N * beam_width * n_moves``. Cross-scramble
    batching is the architectural foundation for the rest of the M8 perf
    overhaul (C2-C5 build on this shape).

    Args:
        net: Value net mapping ``(B, n_stickers)`` -> ``(B,)``.
        spec: Cube spec (move set, stickers, solved state).
        states: ``(N, n_stickers)`` int tensor. Not mutated — copied on the
            net's device before solving.
        beam_width: Number of survivors kept per layer per scramble. ``1``
            mirrors greedy (modulo dedup, which is a no-op at width=1
            since the 12 children of any state are all distinct).
        max_steps: Per-row search depth budget applied uniformly to every
            input scramble. Mutually exclusive with ``max_steps_per_state``
            — pass exactly one. Rows still unsolved after ``max_steps``
            beam expansions are recorded as ``-1``.
        max_steps_per_state: Optional ``(N,)`` int tensor on any device
            (will be moved to the net's device). Element ``i`` is the
            step budget for scramble ``i``. The loop runs
            ``int(max_steps_per_state.max())`` iterations; per-row solve
            verdicts gate at the row's own budget. Mutually exclusive with
            ``max_steps``.

    Returns:
        ``BeamSearchResult`` with per-attempt ``solve_lens``, ``solve_paths``,
        and total ``n_expansions`` count.
    """
    if states.ndim != 2 or states.shape[1] != spec.n_stickers:
        raise ValueError(
            f"expected (N, {spec.n_stickers}) states; got shape {tuple(states.shape)}"
        )
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1; got {beam_width}")
    # Mutual-exclusion on the budget kwargs: passing both is ambiguous
    # (whose value wins?), so we raise rather than picking a silent
    # precedence. Passing neither leaves the caller with no budget at all,
    # also an error. Existing scalar-only callers continue to work.
    if max_steps is None and max_steps_per_state is None:
        raise ValueError("must provide either max_steps or max_steps_per_state")
    if max_steps is not None and max_steps_per_state is not None:
        raise ValueError(
            "pass exactly one of max_steps or max_steps_per_state, not both"
        )
    if max_steps is not None and max_steps < 0:
        raise ValueError(f"max_steps must be >= 0; got {max_steps}")

    device = next(net.parameters()).device
    states = states.to(device).clone()

    n = states.shape[0]
    n_stickers = spec.n_stickers
    n_moves = spec.n_moves

    # Resolve per-state budget tensor on-device. Both code paths converge
    # on the same ``(N,)`` int64 tensor so the loop body is uniform.
    if max_steps_per_state is not None:
        if max_steps_per_state.shape != (n,):
            raise ValueError(
                f"max_steps_per_state must have shape ({n},); got "
                f"{tuple(max_steps_per_state.shape)}"
            )
        if (max_steps_per_state < 0).any():
            raise ValueError("max_steps_per_state entries must be >= 0")
        budget = max_steps_per_state.to(device=device, dtype=torch.int64)
    else:
        assert max_steps is not None  # for the type checker
        budget = torch.full((n,), max_steps, dtype=torch.int64, device=device)

    was_training = net.training
    net.eval()

    try:
        return _beam_solve_cross_batched(
            net,
            spec,
            states,
            n=n,
            n_stickers=n_stickers,
            n_moves=n_moves,
            beam_width=beam_width,
            budget=budget,
            device=device,
        )
    finally:
        if was_training:
            net.train()


def _beam_solve_cross_batched(
    net: torch.nn.Module,
    spec: CubeSpec,
    states: torch.Tensor,
    *,
    n: int,
    n_stickers: int,
    n_moves: int,
    beam_width: int,
    budget: torch.Tensor,
    device: torch.device,
) -> BeamSearchResult:
    """All-active-N-at-once beam search. See ``beam_solve_batch`` for semantics."""
    initial_solved = is_solved(states, spec)  # (N,) bool
    solve_paths: list[list[int]] = [[] for _ in range(n)]

    # Total iterations to run: max budget across rows. The per-row gate at
    # the end zeros out solves that landed past their own budget.
    total_steps = int(budget.max().item()) if n > 0 else 0

    if total_steps == 0:
        # No beam steps run. Pre-solved rows record 0; everything else -1.
        solve_lens = torch.where(
            initial_solved,
            torch.zeros(n, dtype=torch.int64, device=device),
            torch.full((n,), -1, dtype=torch.int64, device=device),
        )
        return BeamSearchResult(
            solve_lens=solve_lens,
            solve_paths=solve_paths,
            n_expansions=0,
        )

    # First-solve tracking, fully on-device, full-N shape. Updated each
    # step via scatter from the active subset; read once after the loop
    # for path reconstruction.
    solved_at_step = torch.full((n,), -1, dtype=torch.int64, device=device)
    solved_flat_idx = torch.full((n,), -1, dtype=torch.int64, device=device)

    # Active set: rows that are still being expanded. A row drops out the
    # step it (a) emits goal or (b) exhausts its budget. Initial set
    # excludes pre-solved rows and rows with budget=0.
    initial_active_mask = (~initial_solved) & (budget > 0)  # (N,) bool
    active_idx = initial_active_mask.nonzero(as_tuple=True)[0]  # (N_active,) int64
    active_beam = states[active_idx].unsqueeze(1)  # (N_active, 1, S)
    cur_beam = 1

    # Per-step backpointers, keyed by scramble_idx so ``_walk_back`` can
    # look up exactly the rows it needs at exactly the steps they were
    # alive. Each entry is a dict[int, list[tuple[parent_slot, move_idx]]]
    # of length == n_active_at_that_step, with each list of length cur_beam.
    backpointers: list[dict[int, list[tuple[int, int]]]] = []

    # n_expansions accounting: rows-active * (1 or beam_width) * n_moves
    # per step, accumulated as a Python int. Step 0 has cur_beam=1, later
    # steps have cur_beam=beam_width.
    n_expansions = 0

    step_tensor = torch.zeros((), dtype=torch.int64, device=device)

    for step_idx in range(total_steps):
        n_active = int(active_idx.numel())
        if n_active == 0:
            break

        step_tensor.fill_(step_idx)
        n_expansions += n_active * cur_beam * n_moves

        # Expand: (N_active, B, S) -> (N_active * B, n_moves, S) -> reshape.
        flat_parents = active_beam.reshape(n_active * cur_beam, n_stickers)
        children_full = apply_all_moves(flat_parents, spec)
        n_children_per_row = cur_beam * n_moves
        children = children_full.reshape(n_active, n_children_per_row, n_stickers)

        # Score: net forward over only the active rows.
        flat_children = children.reshape(n_active * n_children_per_row, n_stickers)
        flat_v = net(flat_children).flatten()
        child_v = flat_v.reshape(n_active, n_children_per_row)

        # First-solved tracking on the active subset, then scatter back to
        # the global tensors. ``argmax`` on bool returns the first True
        # (or 0 if all False, gated by ``any_solved``).
        solved_mask = is_solved(flat_children, spec).reshape(
            n_active, n_children_per_row
        )
        any_solved = solved_mask.any(dim=1)  # (N_active,) bool
        first_solved_in_layer = solved_mask.int().argmax(dim=1)  # (N_active,)
        # Newly-solved active rows: never recorded (active rows always have
        # solved_at_step==-1 at this point — once recorded they drop out).
        if bool(any_solved.any().item()):
            newly_solved_active = active_idx[any_solved]
            # ``first_solved_in_layer`` indexes the (n_active, cur_beam*n_moves)
            # layout. We want the same index but only for newly-solved rows.
            solved_at_step[newly_solved_active] = step_tensor
            solved_flat_idx[newly_solved_active] = first_solved_in_layer[any_solved]

        # On-device hash + CPU dedup mask + on-device topk. Same recipe as
        # the pre-compaction loop, just on the (n_active, ...) shape.
        children_for_hash = flat_children.reshape(
            n_active, n_children_per_row, n_stickers
        )
        hashes_cpu = state_hash(children_for_hash).detach().cpu().numpy()

        is_winner_np = np.zeros((n_active, n_children_per_row), dtype=np.bool_)
        for i in range(n_active):
            _unique_h, first_occurrence_idx = np.unique(
                hashes_cpu[i], return_index=True
            )
            is_winner_np[i, first_occurrence_idx] = True

        is_winner = torch.from_numpy(is_winner_np).to(device)
        inf_tensor = torch.full_like(child_v, float("inf"))
        dense_v = torch.where(is_winner, child_v, inf_tensor)

        # Short-beam clamp: torch.topk requires k <= dim_size. Step 0 has
        # cur_beam=1 -> only n_moves children, so beam_width > n_moves
        # needs padding (real-state duplicates that dedup natually next
        # step, matching pre-compaction behavior).
        k_eff = min(beam_width, n_children_per_row)
        _topk_values, topk_idxs = torch.topk(dense_v, k=k_eff, largest=False, dim=1)
        if k_eff < beam_width:
            pad = topk_idxs[:, :1].expand(n_active, beam_width - k_eff)
            next_flat_idxs = torch.cat([topk_idxs, pad], dim=1)
        else:
            next_flat_idxs = topk_idxs

        # Record per-step backpointers keyed by scramble_idx — only the
        # rows that were active at this step have entries. Path
        # reconstruction looks up bp[step][scramble_idx] for each step
        # in [0, solved_at_step[scramble]) and walks the chain.
        next_flat_idxs_cpu = next_flat_idxs.cpu().tolist()
        active_idx_cpu = active_idx.cpu().tolist()
        layer_bp: dict[int, list[tuple[int, int]]] = {
            scr_i: [(j // n_moves, j % n_moves) for j in row]
            for scr_i, row in zip(active_idx_cpu, next_flat_idxs_cpu, strict=True)
        }
        backpointers.append(layer_bp)

        # Gather next beam for the active rows.
        gather_idx = next_flat_idxs.unsqueeze(-1).expand(
            n_active, beam_width, n_stickers
        )
        new_active_beam = torch.gather(children, dim=1, index=gather_idx)
        cur_beam = beam_width

        # Drop done rows from the active set. A row is "done" after this
        # step if it just emitted goal or its next step would exceed
        # budget. ``budget[active_idx]`` aligns the per-row budget vector
        # to the active subset; the next-step gate is ``next_step >= b``.
        next_step = step_idx + 1
        active_budget = budget[active_idx]
        is_still_active = (~any_solved) & (next_step < active_budget)
        # If nothing dropped, skip the gather; if all dropped, the empty
        # tensors propagate fine and the next loop iter sees n_active=0.
        if not bool(is_still_active.all().item()):
            keep = is_still_active.nonzero(as_tuple=True)[0]
            active_idx = active_idx[keep]
            active_beam = new_active_beam[keep]
        else:
            active_beam = new_active_beam

    # Single host transfer for path reconstruction.
    solved_step_per = solved_at_step.cpu().tolist()
    solved_flat_per = solved_flat_idx.cpu().tolist()
    budget_cpu = budget.cpu().tolist()
    initial_solved_cpu = initial_solved.cpu().tolist()

    # Final solve_lens: pre-solved rows -> 0; first-solve within budget ->
    # solved_at_step + 1 moves; otherwise -1.
    solved_within_budget = (solved_at_step != -1) & (solved_at_step < budget)
    solve_lens = torch.where(
        initial_solved,
        torch.zeros(n, dtype=torch.int64, device=device),
        torch.where(
            solved_within_budget,
            solved_at_step + 1,
            torch.full((n,), -1, dtype=torch.int64, device=device),
        ),
    )

    # Reconstruct paths only for rows whose first-solve fits the budget.
    for i in range(n):
        if initial_solved_cpu[i]:
            continue  # solve_paths[i] stays []
        step = solved_step_per[i]
        if step == -1 or step >= budget_cpu[i]:
            continue  # not solved within budget — solve_paths[i] stays []
        flat = solved_flat_per[i]
        parent_slot = flat // n_moves
        move_idx = flat % n_moves
        path = _walk_back(backpointers, parent_slot, scramble_idx=i, up_to_step=step)
        path.append(move_idx)
        solve_paths[i] = path

    return BeamSearchResult(
        solve_lens=solve_lens,
        solve_paths=solve_paths,
        n_expansions=n_expansions,
    )


def _walk_back(
    backpointers: list[dict[int, list[tuple[int, int]]]],
    final_parent_slot: int,
    *,
    scramble_idx: int,
    up_to_step: int,
) -> list[int]:
    """Reconstruct prefix of move sequence for scramble ``scramble_idx``.

    Walks layers ``[0, up_to_step)`` in reverse — the layer ``up_to_step``
    is the one where the goal state was emitted, and that move is appended
    by the caller after ``_walk_back`` returns.

    Each step's backpointer dict only contains entries for scrambles that
    were active at that step. Scramble ``scramble_idx`` is guaranteed to
    be present at every layer in ``[0, up_to_step)`` because rows only
    drop out when they finish; if a scramble solved at step ``up_to_step``
    it was active at all earlier steps.
    """
    path: list[int] = []
    cur = final_parent_slot
    for layer_idx in range(up_to_step - 1, -1, -1):
        parent, move = backpointers[layer_idx][scramble_idx][cur]
        path.insert(0, move)
        cur = parent
    return path
