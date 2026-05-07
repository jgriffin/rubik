"""Beam search policy: V_θ-scored batched search with within-beam dedup.

Per-state batched beam search using V_θ as the scoring function. The beam
holds the K best-scoring frontier states; at each step every beam slot is
expanded over all moves, scored, deduplicated by raw byte equality, and the
top-K survivors form the next frontier. Search terminates as soon as every
input scramble has emitted its first goal-state expansion (or all have hit
``max_steps``); the move sequence is recovered by walking back-pointer links.

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
The Python per-scramble dedup loop (``dict[bytes(state)]``) is preserved
in C1 — it is replaced by an on-device hash + ``torch.unique`` in C3.
Likewise the CPU ``sorted()`` top-k is preserved in C1 — replaced by
``torch.topk`` in C4.

The net is set to ``eval()`` while solving and restored to its prior
train/eval state on exit so BN running stats drive the forward pass — the
policy that runs at deployment / search time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rubik.cube.env import apply_all_moves, is_solved
from rubik.cube.spec import CubeSpec


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
    max_steps: int,
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
        max_steps: Per-row search depth budget. Rows still unsolved after
            this many beam expansions are recorded as ``-1``.

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
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0; got {max_steps}")

    device = next(net.parameters()).device
    states = states.to(device).clone()

    n = states.shape[0]
    n_stickers = spec.n_stickers
    n_moves = spec.n_moves

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
            max_steps=max_steps,
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
    max_steps: int,
    device: torch.device,
) -> BeamSearchResult:
    """All-N-at-once beam search. See ``beam_solve_batch`` for semantics."""
    # solve_lens[i] = number of moves applied to reach goal for scramble i.
    # Initialized to 0 for inputs already solved, else -1 (not yet solved).
    initial_solved = is_solved(states, spec)  # (N,) bool
    solve_lens = torch.where(
        initial_solved,
        torch.zeros(n, dtype=torch.int64, device=device),
        torch.full((n,), -1, dtype=torch.int64, device=device),
    )
    solve_paths: list[list[int]] = [[] for _ in range(n)]

    if max_steps == 0:
        # Pre-solved rows kept at 0; everything else stays -1.
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

    # When a scramble first emits a solved child, we capture (step, flat_child_idx)
    # so we can reconstruct its path at the end. ``flat_child_idx`` is into the
    # (cur_beam * n_moves) flat child layout for that scramble.
    solved_step_per: list[int] = [-1] * n  # -1 = not yet solved
    solved_flat_idx: list[int] = [-1] * n

    n_expansions = 0
    # Already-solved rows are recorded as solve_len=0 outside the loop;
    # mark them as "done" so the all-done check trips correctly. They also
    # do not contribute to ``n_expansions`` — the metric counts work spent
    # actively solving rows, matching the pre-C1 behavior where pre-solved
    # rows short-circuited the per-scramble loop.
    done_mask = initial_solved.clone()  # on-device (N,) bool
    n_active = int((~initial_solved).sum().item())

    for step_idx in range(max_steps):
        if bool(done_mask.all().item()):
            break

        # Expand: (N, B, S) -> apply_all_moves -> (N, B, n_moves, S)
        # then collapse the inner two axes to get per-scramble children.
        flat_parents = beam_states.reshape(n * cur_beam, n_stickers)
        children_full = apply_all_moves(flat_parents, spec)  # (N*B, n_moves, S)
        children = children_full.reshape(n, cur_beam * n_moves, n_stickers)
        # Count expansions only for rows still actively solving — matches
        # the pre-C1 semantics where rows that started solved (and rows
        # that solved earlier) didn't run further beam steps.
        n_expansions += n_active * cur_beam * n_moves

        # Score: one big net call.
        flat_children = children.reshape(n * cur_beam * n_moves, n_stickers)
        flat_v = net(flat_children).flatten()
        child_v = flat_v.reshape(n, cur_beam * n_moves)

        # Per-scramble first-solved tracking. solved_mask is (N, B*n_moves) bool.
        solved_mask = is_solved(flat_children, spec).reshape(n, cur_beam * n_moves)
        any_solved_per_scramble = solved_mask.any(dim=1)  # (N,) bool

        # Convert to host once for path-reconstruction bookkeeping. C2 will
        # remove this sync; here we keep parity with the prior implementation.
        # We need: for each scramble that first solved THIS step, the index of
        # its first solved child in the flat (B*n_moves) layout.
        first_solved_flat = torch.where(
            solved_mask.any(dim=1),
            solved_mask.float().argmax(dim=1),
            torch.full((n,), -1, dtype=torch.int64, device=device),
        )
        any_solved_cpu = any_solved_per_scramble.cpu().tolist()
        first_solved_cpu = first_solved_flat.cpu().tolist()
        for i in range(n):
            if solved_step_per[i] != -1 or done_mask[i].item():
                # Already solved earlier (or pre-solved at input): stays.
                continue
            if any_solved_cpu[i]:
                solved_step_per[i] = step_idx
                solved_flat_idx[i] = int(first_solved_cpu[i])
                solve_lens[i] = step_idx + 1

        # ``done_mask`` includes both the pre-solved-at-input rows and rows
        # that have emitted their first goal-state. Once a row is done, we
        # still expand it (its beam continues forward) — its entry in
        # ``solve_lens`` is locked, but expanding keeps the tensor shapes
        # uniform across rows for the rest of the loop. C2 turns this into
        # a clean run-to-completion.
        newly_done = any_solved_per_scramble & (~done_mask)
        done_mask = done_mask | any_solved_per_scramble
        n_active -= int(newly_done.sum().item())

        # Per-scramble dedup + top-k. KEPT FROM PRE-C1: dict-of-bytes Python
        # loop. C3 replaces with on-device hash + torch.unique.
        children_cpu = flat_children.detach().cpu().numpy().reshape(
            n, cur_beam * n_moves, n_stickers
        )
        v_cpu = child_v.detach().cpu().numpy()  # (N, B*n_moves)

        # next_beam_idxs[i] = list of (parent_slot, move_idx) for next layer.
        # Tensor of flat child indices in (B*n_moves) layout, padded to
        # beam_width. Padding repeats the first survivor — duplicates dedup
        # naturally on the next iteration. ``next_actual_size[i]`` records
        # the true unique-survivor count (informational; not used for shape).
        next_flat_idxs = torch.empty(
            (n, beam_width), dtype=torch.int64, device=device
        )
        layer_bp_per_scramble: list[list[tuple[int, int]]] = []

        for i in range(n):
            dedup: dict[bytes, tuple[float, int]] = {}
            row_states = children_cpu[i]  # (B*n_moves, n_stickers)
            row_v = v_cpu[i]  # (B*n_moves,)
            for j in range(row_states.shape[0]):
                key = row_states[j].tobytes()
                v_j = float(row_v[j])
                existing = dedup.get(key)
                if existing is None or v_j < existing[0]:
                    dedup[key] = (v_j, j)
            survivors = sorted(dedup.values())[:beam_width]
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
                next_flat_idxs[i, k] = j_flat

        backpointers.append(layer_bp_per_scramble)

        # Gather the next beam: children[i, next_flat_idxs[i]] for each i.
        # Use torch.gather across dim=1 with index broadcast over n_stickers.
        gather_idx = next_flat_idxs.unsqueeze(-1).expand(n, beam_width, n_stickers)
        beam_states = torch.gather(children, dim=1, index=gather_idx)
        cur_beam = beam_width

    # Reconstruct paths for solved scrambles.
    for i in range(n):
        if solved_step_per[i] == -1:
            continue
        step = solved_step_per[i]
        flat = solved_flat_idx[i]
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
