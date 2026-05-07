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
NumPy ops. The result feeds the existing CPU ``sorted()`` top-k
(replaced by ``torch.topk`` in C4).

**Why not on-device dedup?** ``torch.unique`` and ``torch.sort`` on
MPS are broken for full-range int64 — they only consider the low 32
bits, collapsing distinct hashes whose high bits differ. Verified at
C3 implementation time:
``torch.unique(torch.tensor([1, 1<<32], device='mps')) == [1]``.
Until that lands upstream the dedup runs on the host; the hash itself
stays on-device since it's where the cube state lives.

**Run-to-completion (M8 C2).** The early-exit-when-all-solved check
(``done_mask.all().item()`` per step) is removed. First-solve tracking
(``solved_at_step``, ``solved_flat_idx``) is kept on-device and updated
each step via ``torch.where`` against an ``is_first_solve_now`` mask;
expansions for already-solved or out-of-budget rows are excluded from the
``n_expansions`` accounting via an on-device active-row count summed
once at loop end. The host-side path-reconstruction read is deferred to
a single ``.cpu()`` call after the loop.

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
    """All-N-at-once beam search. See ``beam_solve_batch`` for semantics."""
    initial_solved = is_solved(states, spec)  # (N,) bool
    solve_paths: list[list[int]] = [[] for _ in range(n)]

    # Total iterations to run: max budget across rows, but never more than
    # what any row could possibly need. The per-row gate at the end zeros
    # out solves that landed past their own budget.
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

    # Beam: (N, B, n_stickers). Start at B=1 (the input state per scramble).
    beam_states = states.unsqueeze(1)  # (N, 1, n_stickers)
    cur_beam = 1

    # backpointers[step] is a list of length N; entry i is a list of
    # ``(parent_slot, move_idx)`` tuples of length == cur_beam_at_that_step
    # describing how each survivor at layer ``step`` arose from layer
    # ``step - 1``. Layer 0 (the input root) has no entry. Path
    # reconstruction walks layers in reverse.
    backpointers: list[list[list[tuple[int, int]]]] = []

    # First-solve tracking, fully on-device. ``solved_at_step[i]`` is the
    # step_idx at which scramble i first emitted a goal-state child (-1 if
    # never within ``total_steps``); ``solved_flat_idx[i]`` is the index of
    # that child in the (cur_beam * n_moves) flat layout at that step.
    # These are read once after the loop for path reconstruction — no
    # per-step host transfer.
    solved_at_step = torch.full((n,), -1, dtype=torch.int64, device=device)
    solved_flat_idx = torch.full((n,), -1, dtype=torch.int64, device=device)

    # Per-step active-row count, accumulated as a 0-d tensor and read
    # once at the end. A row is "active" at step k iff: it was not solved
    # at input AND has not yet emitted a goal AND k is within its budget.
    # This matches the pre-C1 semantics where pre-solved and already-solved
    # rows did not contribute to ``n_expansions``.
    active_steps_total = torch.zeros((), dtype=torch.int64, device=device)

    # step_idx (0-indexed) tensor scratch, reused via fill_ to avoid
    # rebuilding per step.
    step_tensor = torch.zeros((), dtype=torch.int64, device=device)

    for step_idx in range(total_steps):
        step_tensor.fill_(step_idx)

        # Expand: (N, B, S) -> apply_all_moves -> (N*B, n_moves, S),
        # then reshape to per-scramble children.
        flat_parents = beam_states.reshape(n * cur_beam, n_stickers)
        children_full = apply_all_moves(flat_parents, spec)  # (N*B, n_moves, S)
        children = children_full.reshape(n, cur_beam * n_moves, n_stickers)

        # Active mask: not pre-solved, not yet emitted goal, within own budget.
        # Accumulate count for n_expansions on-device.
        is_within_budget = step_tensor < budget  # (N,) bool, broadcasts
        is_active = (~initial_solved) & (solved_at_step == -1) & is_within_budget
        active_steps_total = active_steps_total + is_active.sum()

        # Score: one big net call.
        flat_children = children.reshape(n * cur_beam * n_moves, n_stickers)
        flat_v = net(flat_children).flatten()
        child_v = flat_v.reshape(n, cur_beam * n_moves)

        # Per-scramble first-solved tracking. ``solved_mask`` is
        # (N, B*n_moves) bool. ``argmax`` on a bool tensor returns the index
        # of the first True (or 0 if all False — gated below by
        # ``any_solved``).
        solved_mask = is_solved(flat_children, spec).reshape(n, cur_beam * n_moves)
        any_solved = solved_mask.any(dim=1)  # (N,) bool
        first_solved_in_layer = solved_mask.int().argmax(dim=1)  # (N,) int64

        # Only record a first-solve if (1) this scramble has not yet been
        # marked solved, and (2) any beam slot just emitted goal. Already
        # pre-solved rows (initial_solved) are deliberately included here —
        # they may "re-solve" later but their solve_len stays 0 via the
        # final gate against ``initial_solved`` at the bottom.
        is_first_solve_now = (solved_at_step == -1) & any_solved
        solved_at_step = torch.where(
            is_first_solve_now, step_tensor, solved_at_step
        )
        solved_flat_idx = torch.where(
            is_first_solve_now, first_solved_in_layer, solved_flat_idx
        )

        # On-device hash + CPU-side dedup (M8 C3). ``state_hash`` runs on
        # the same device as the children; the per-step host transfer is
        # ``(N, B*n_moves)`` int64 hashes + ``(N, B*n_moves)`` float32
        # values (~1.2MB at width=128) — ~7× less than C2's full state
        # tensor transfer. ``torch.unique`` on MPS is broken for full
        # int64 (only low 32 bits compared), so the dedup runs on the
        # host; ``numpy.unique`` per row + ``numpy.minimum.at`` for the
        # min-V scatter replace the Python ``dict[bytes(...)]`` loop.
        n_children_per_row = cur_beam * n_moves
        children_for_hash = flat_children.reshape(n, n_children_per_row, n_stickers)
        hashes_cpu = state_hash(children_for_hash).detach().cpu().numpy()
        v_cpu = child_v.detach().cpu().numpy()

        # ``next_flat_idxs[i, k]`` is the (B*n_moves) index of the k-th
        # survivor for scramble i, used to gather the next beam.
        next_flat_idxs_np = np.empty((n, beam_width), dtype=np.int64)
        layer_bp_per_scramble: list[list[tuple[int, int]]] = []

        flat_idx_arr_f32 = np.arange(n_children_per_row, dtype=np.float32)
        for i in range(n):
            row_h = hashes_cpu[i]
            row_v = v_cpu[i]
            unique_h, inverse = np.unique(row_h, return_inverse=True)
            n_unique = unique_h.shape[0]

            # min-V per equivalence class. ``np.minimum.at`` is the
            # unbuffered scatter-with-min: equivalent to a sequential
            # ``min_v[inverse[j]] = min(min_v[inverse[j]], row_v[j])``.
            min_v = np.full(n_unique, np.inf, dtype=row_v.dtype)
            np.minimum.at(min_v, inverse, row_v)

            # Tiebreak: among children whose V equals the class min, pick
            # the smallest within-row idx as the representative. Using
            # float32 idxs keeps the same data flow as the on-device path
            # in case we lift this back to GPU once MPS int64 unique is
            # fixed; mantissa precision is ample for n_children_per_row.
            v_at_min = min_v[inverse]
            is_winner = row_v == v_at_min
            idx_winners = np.where(is_winner, flat_idx_arr_f32, np.inf)
            rep_idx_f32 = np.full(n_unique, np.inf, dtype=np.float32)
            np.minimum.at(rep_idx_f32, inverse, idx_winners)

            # KEPT FROM PRE-C3: CPU ``sorted()`` top-k. C4 replaces with
            # ``torch.topk`` — for C3 only the *input* to top-k changed
            # (now compact NumPy dedup output instead of a Python dict
            # keyed on state bytes).
            survivors: list[tuple[float, int]] = sorted(
                zip(min_v.tolist(), rep_idx_f32.astype(np.int64).tolist(), strict=True)
            )[:beam_width]

            # Survivor count can be < beam_width at very early steps if the
            # parent set is small (e.g. cur_beam=1 → only 12 unique children
            # with beam_width=16 → we get 12 survivors). Pad by repeating
            # the first survivor — duplicates dedup on the next step.
            if not survivors:
                # Should not happen unless every child matches a parent
                # exactly (impossible — moves are non-identity). Pad with
                # the first flat child as a safe fallback.
                survivors = [(0.0, 0)] * beam_width
            elif len(survivors) < beam_width:
                pad = [survivors[0]] * (beam_width - len(survivors))
                survivors = survivors + pad

            layer_bp = [
                (j_flat // n_moves, j_flat % n_moves) for _, j_flat in survivors
            ]
            layer_bp_per_scramble.append(layer_bp)
            for k, (_, j_flat) in enumerate(survivors):
                next_flat_idxs_np[i, k] = j_flat

        next_flat_idxs = torch.from_numpy(next_flat_idxs_np).to(device)
        backpointers.append(layer_bp_per_scramble)

        # Gather the next beam: children[i, next_flat_idxs[i]] for each i.
        # Use torch.gather across dim=1 with index broadcast over n_stickers.
        gather_idx = next_flat_idxs.unsqueeze(-1).expand(n, beam_width, n_stickers)
        beam_states = torch.gather(children, dim=1, index=gather_idx)
        cur_beam = beam_width

    # Single host transfer for path reconstruction. The on-device tensors
    # live unchanged for callers reading ``solve_lens`` afterward.
    solved_step_per = solved_at_step.cpu().tolist()
    solved_flat_per = solved_flat_idx.cpu().tolist()
    budget_cpu = budget.cpu().tolist()
    initial_solved_cpu = initial_solved.cpu().tolist()

    # Final solve_lens: pre-solved rows -> 0; first-solve within budget ->
    # solved_at_step + 1 moves; otherwise -1. Built on-device via where.
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

    # ``n_expansions`` semantics: per-row, count children expanded only for
    # steps where the row was actively solving (not pre-solved at input,
    # not yet emitted goal, within own budget). Each row-step expansion
    # produces ``cur_beam_at_that_step * n_moves`` children, where
    # ``cur_beam`` is 1 at step 0 and ``beam_width`` thereafter. We split
    # the accumulated active-row-step count into the two branches:
    n_step0_active = int(((~initial_solved) & (budget > 0)).sum().item())
    total_active_rowsteps = int(active_steps_total.item())
    n_later_rowsteps = total_active_rowsteps - n_step0_active
    n_expansions = (
        n_step0_active * n_moves + n_later_rowsteps * beam_width * n_moves
    )

    return BeamSearchResult(
        solve_lens=solve_lens,
        solve_paths=solve_paths,
        n_expansions=n_expansions,
    )


def _walk_back(
    backpointers: list[list[list[tuple[int, int]]]],
    final_parent_slot: int,
    *,
    scramble_idx: int,
    up_to_step: int,
) -> list[int]:
    """Reconstruct prefix of move sequence for scramble ``scramble_idx``.

    Walks layers ``[0, up_to_step)`` in reverse — the layer ``up_to_step``
    is the one where the goal state was emitted, and that move is appended
    by the caller after ``_walk_back`` returns.
    """
    path: list[int] = []
    cur = final_parent_slot
    for layer_idx in range(up_to_step - 1, -1, -1):
        parent, move = backpointers[layer_idx][scramble_idx][cur]
        path.insert(0, move)
        cur = parent
    return path
