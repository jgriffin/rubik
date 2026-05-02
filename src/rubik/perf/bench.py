"""Bracket-sync timing primitive + bootstrap CI.

The "primary truth" of the M4 measurement triangulation: warmup → sync →
`perf_counter` → `fn()` → sync → `perf_counter`. Profiler and macmon are
investigation tools layered on top of these numbers, not replacements.

Stateless: no globals, no caching, no knowledge of cubes — pure timing.
"""

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import torch


def time_op(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    trials: int = 30,
    device: str = "mps",
) -> list[float]:
    """Run warmup → sync → timed trials. Returns per-trial seconds.

    Each timed trial brackets `fn()` with `torch.mps.synchronize()` calls so
    wall-time captures GPU work, not just dispatch. The CPU path skips syncs.
    """
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "time_op called with device='mps' but torch.backends.mps.is_available()"
            " is False; refusing to silently fall back to CPU."
        )
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative; got {warmup}")
    if trials < 1:
        raise ValueError(f"trials must be >= 1; got {trials}")

    sync = torch.mps.synchronize if device == "mps" else _noop

    for _ in range(warmup):
        fn()

    timings: list[float] = []
    for _ in range(trials):
        sync()
        start = perf_counter()
        fn()
        sync()
        timings.append(perf_counter() - start)
    return timings


def bootstrap_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Returns (median, lo, hi) for the given confidence level.

    Pure numpy resampling-with-replacement on the median. Deterministic
    under fixed seed.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")

    arr = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    medians = np.median(arr[idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(medians, alpha))
    hi = float(np.quantile(medians, 1.0 - alpha))
    return float(np.median(arr)), lo, hi


def _noop() -> None:
    pass
