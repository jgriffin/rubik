"""Beam search policy: V_θ-scored batched search with within-beam dedup.

Per-state batched beam search using V_θ as the scoring function. The beam
holds the K best-scoring frontier states; at each step every beam slot is
expanded over all moves, scored, deduplicated by raw byte equality, and the
top-K survivors form the next frontier. Search terminates as soon as any
expansion produces ``spec.solved_state``; the move sequence is recovered by
walking back-pointer links.

Within-beam dedup uses raw-bytes equality (``state.tobytes()``), not the
24-rotation orbit canonicalization in ``rubik.oracle.v_star_2x2``. The orbit
canonicalizer is correct for V* lookup but wrong for path-tracked search:
collapsing rotation-equivalent states would emit moves that don't reach the
canonical representative when applied to the actual input cube. Raw bytes is
exact equality and faster.

Sequential per-parent processing. Each input state runs its own beam; cross-
parent batching is a perf optimization deferred until the wall-time gate
forces it (M8 3x3). M4 batch sweeps placed the saturation knee around
B=4096; at ``beam_width=256`` the inner expansion is 256×12=3072 net
forwards, sitting just under the knee.

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

    Args:
        net: Value net mapping ``(B, n_stickers)`` -> ``(B,)``.
        spec: Cube spec (move set, stickers, solved state).
        states: ``(N, n_stickers)`` int tensor. Not mutated — copied on the
            net's device before solving.
        beam_width: Number of survivors kept per layer. ``1`` mirrors greedy
            (modulo dedup, which is a no-op at width=1 since the 12 children
            of any state are all distinct).
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
    was_training = net.training
    net.eval()

    solve_lens = torch.full((n,), -1, dtype=torch.int64, device=device)
    solve_paths: list[list[int]] = [[] for _ in range(n)]
    total_expansions = 0

    try:
        for i in range(n):
            length, path, n_exp = _beam_solve_single(
                net,
                spec,
                states[i],
                beam_width=beam_width,
                max_steps=max_steps,
            )
            solve_lens[i] = length
            solve_paths[i] = path
            total_expansions += n_exp
    finally:
        if was_training:
            net.train()

    return BeamSearchResult(
        solve_lens=solve_lens,
        solve_paths=solve_paths,
        n_expansions=total_expansions,
    )


def _beam_solve_single(
    net: torch.nn.Module,
    spec: CubeSpec,
    start_state: torch.Tensor,
    *,
    beam_width: int,
    max_steps: int,
) -> tuple[int, list[int], int]:
    """Beam search for one start state.

    Returns ``(solve_len, path, n_expansions)``.
    """
    n_stickers = spec.n_stickers
    n_moves = spec.n_moves
    device = start_state.device

    if bool(is_solved(start_state, spec).item()):
        return 0, [], 0

    if max_steps == 0:
        return -1, [], 0

    # Beam: (B, n_stickers); start with the single input state.
    beam_states = start_state.unsqueeze(0)
    # backpointers[layer][slot] = (parent_slot_in_prev_layer, move_idx_taken).
    # Layer 0 corresponds to the first set of survivors (after step 0's
    # expansion); the input start state has no entry. Path reconstruction
    # walks layers in reverse to recover the move sequence.
    backpointers: list[list[tuple[int, int]]] = []
    n_expansions = 0

    for step_idx in range(max_steps):
        b = beam_states.shape[0]
        children = apply_all_moves(beam_states, spec).reshape(b * n_moves, n_stickers)
        n_expansions += b * n_moves

        child_v = net(children).flatten()
        solved_mask = is_solved(children, spec)

        if bool(solved_mask.any().item()):
            solved_idxs = solved_mask.nonzero(as_tuple=False).squeeze(-1)
            chosen_flat = int(solved_idxs[0].item())
            parent_slot = chosen_flat // n_moves
            move_idx = chosen_flat % n_moves
            path = _walk_back(backpointers, parent_slot) + [move_idx]
            return step_idx + 1, path, n_expansions

        children_cpu = children.detach().cpu().numpy()
        v_cpu = child_v.detach().cpu().numpy()
        # dict[bytes, (v, parent_slot, move_idx)] keeps best-V representative.
        dedup: dict[bytes, tuple[float, int, int]] = {}
        for j in range(children_cpu.shape[0]):
            key = children_cpu[j].tobytes()
            v_j = float(v_cpu[j])
            existing = dedup.get(key)
            if existing is None or v_j < existing[0]:
                dedup[key] = (v_j, j // n_moves, j % n_moves)

        survivors = sorted(dedup.values())[:beam_width]
        if not survivors:
            return -1, [], n_expansions

        layer_bp = [(parent, move) for _, parent, move in survivors]
        beam_idxs = [parent * n_moves + move for _, parent, move in survivors]
        beam_idxs_t = torch.tensor(beam_idxs, device=device, dtype=torch.int64)
        beam_states = children[beam_idxs_t]
        backpointers.append(layer_bp)

    return -1, [], n_expansions


def _walk_back(
    backpointers: list[list[tuple[int, int]]], final_parent_slot: int
) -> list[int]:
    """Reconstruct move sequence by walking back-pointer chain to the start."""
    path: list[int] = []
    cur = final_parent_slot
    for layer_idx in range(len(backpointers) - 1, -1, -1):
        parent, move = backpointers[layer_idx][cur]
        path.insert(0, move)
        cur = parent
    return path
