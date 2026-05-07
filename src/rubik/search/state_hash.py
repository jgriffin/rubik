"""Stable on-device int64 hash for cube sticker tensors.

Used by ``beam_solve_batch`` to dedup expanded children on-device — replaces
the prior ``dict[bytes(state)] = (v, idx)`` Python loop and its forced
``.cpu().numpy()`` round-trip per beam step.

The hash is a vectorized FNV-1a variant operating on int8 sticker tensors:

1. Sign-extend stickers to int64 then mask to the unsigned byte (``& 0xFF``).
   Cube sticker values are 0..5 (face indices) so negative-on-extension is
   not actually possible, but the mask is cheap and keeps the function
   correct for any int8 input layout.
2. Pad the sticker dimension up to a multiple of 8 with zeros.
3. Reshape to ``(..., n_chunks, 8)`` and pack each 8-byte chunk into one
   int64 via shift-and-sum (byte ``b`` goes to bit-positions ``[8b, 8b+8)``).
4. Fold the chunks together via FNV-1a: ``h = (h ^ chunk) * FNV_PRIME``,
   starting from the canonical FNV offset basis. Multiplication overflows
   the int64 range freely (PyTorch wraps); that's the standard FNV move.

**Collision-rate analysis.** A 64-bit hash on a 432-bit state (3x3, 54
bytes × 8 bits) has per-pair collision probability ≈ 2⁻⁶⁴ ≈ 5×10⁻²⁰
under the random-oracle assumption. The 3x3 has ~3.7M states reachable in
QTM ≤ 14 (the maximum search depth we run). Across an entire eval sweep
the expansion budget is on the order of 30M children. Expected collisions
in a single eval run is ~30M × 30M × 5×10⁻²⁰ ≈ 5×10⁻⁵ — vanishingly
small. Acceptable. For the 2x2 (n_stickers=24) the input is even smaller
and the analysis is even more permissive.

A test in ``tests/search/test_state_hash.py`` exhaustively hashes the
QTM-≤6 BFS oracle from solved (~984k 3x3 states) and asserts no
collisions — a sufficient empirical lower bound for the corpus the search
actually visits at shallow depths.

The ``state_hash`` function is shape-agnostic over the leading dimensions:
``(..., n_stickers)`` int8 in, ``(...,)`` int64 out. Works on CPU and MPS
(int64 multiply, bitwise_xor, bitwise_and, and left-shift are all
MPS-supported).
"""

from __future__ import annotations

import torch

# FNV-1a 64-bit constants (https://en.wikipedia.org/wiki/Fowler-Noll-Vo).
# Stored as Python ints; PyTorch will create an int64 tensor and rely on
# two's-complement wrap for the unsigned-overflow that FNV mixing assumes.
_FNV_PRIME: int = 1099511628211
_FNV_OFFSET_BASIS: int = -3750763034362895579  # 0xCBF29CE484222325 as signed int64


def state_hash(states: torch.Tensor) -> torch.Tensor:
    """Hash int8 cube state(s) to int64.

    Args:
        states: int8 tensor of shape ``(..., n_stickers)``. Last dimension
            is the sticker count; leading dimensions are arbitrary.

    Returns:
        int64 tensor of shape ``states.shape[:-1]``. Same device as input.
    """
    if states.dtype != torch.int8:
        raise ValueError(f"state_hash expects int8 input; got {states.dtype}")
    if states.ndim < 1:
        raise ValueError("state_hash expects at least 1-d input")

    device = states.device
    n_stickers = states.shape[-1]

    # int8 -> int64, then mask to a clean unsigned byte. Multiplication and
    # bitwise ops below operate on int64 and rely on two's-complement wrap
    # for FNV's "implicit modulo 2^64" semantics.
    states_i64 = states.to(torch.int64) & 0xFF  # (..., n_stickers)

    # Pad sticker dim up to multiple of 8 so we can reshape into 8-wide chunks.
    n_chunks = (n_stickers + 7) // 8
    pad_len = n_chunks * 8 - n_stickers
    if pad_len > 0:
        pad_shape = states_i64.shape[:-1] + (pad_len,)
        pad = torch.zeros(pad_shape, dtype=torch.int64, device=device)
        states_i64 = torch.cat([states_i64, pad], dim=-1)

    # (..., n_chunks, 8) — each row of 8 bytes packs into one int64.
    chunks = states_i64.reshape(*states_i64.shape[:-1], n_chunks, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=device) * 8  # (8,)
    # Pack: byte i of each chunk shifted to bit positions [8i, 8i+8).
    # Sum across the byte axis fuses the 8 bytes into one int64 key.
    packed = (chunks << shifts).sum(dim=-1)  # (..., n_chunks)

    # FNV-1a fold. The chunk axis is small (3 for 2x2, 7 for 3x3), so a
    # Python-level loop here adds at most 7 GPU ops per call — kernel-launch
    # overhead is dwarfed by the per-step net forward + scatter_reduce.
    leading_shape = packed.shape[:-1]
    h = torch.full(leading_shape, _FNV_OFFSET_BASIS, dtype=torch.int64, device=device)
    for c in range(n_chunks):
        h = torch.bitwise_xor(h, packed[..., c]) * _FNV_PRIME
    return h
