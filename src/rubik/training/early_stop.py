"""Early-stop monitor for DAVI training.

Tracks a single scalar metric across evals (sync periods, in practice)
and decides when training should stop based on a plateau criterion:

- ``min_evals`` warmup: never stop while ``n_updates < min_evals``.
- After warmup, compute ``best = min(history)``. If the best value is
  older than ``patience_evals`` evals (i.e. the most recent
  ``patience_evals`` evals haven't improved on ``best`` by at least
  ``min_delta`` in absolute terms), stop.

Spec-agnostic: both 2x2 and 3x3 use the same monitor — no size suffix.

The monitor is pure state + pure logic. It does NOT save checkpoints, log
events, or know about training. The caller does that — see
``experiments/davi-3x3/run.py`` for the end-to-end wiring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopState:
    """Snapshot of the monitor's internal state for logging / debugging."""

    n_updates: int
    history: list[float]
    best_value: float | None
    best_index: int | None  # 0-indexed position in history of the best value
    last_value: float | None


class EarlyStopMonitor:
    """Plateau-based early-stop on a metric where lower is better.

    Args:
        patience_evals: number of evals of no improvement before stopping.
        min_evals: warmup; do not stop until at least this many evals.
        min_delta: absolute improvement threshold. An eval ``x`` "improves"
            on the running best ``b`` iff ``x < b - min_delta``. Tiny
            improvements still count as progress; only true plateaus
            trigger the stop.

    Usage::

        monitor = EarlyStopMonitor(patience_evals=12, min_evals=4, min_delta=0.001)
        for step, metric in eval_stream():
            if monitor.update(metric):
                break
    """

    def __init__(
        self,
        *,
        patience_evals: int,
        min_evals: int,
        min_delta: float,
    ) -> None:
        if patience_evals < 1:
            raise ValueError(f"patience_evals must be >= 1; got {patience_evals}")
        if min_evals < 0:
            raise ValueError(f"min_evals must be >= 0; got {min_evals}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0; got {min_delta}")
        self._patience_evals = patience_evals
        self._min_evals = min_evals
        self._min_delta = min_delta
        self._history: list[float] = []
        self._best_value: float | None = None
        self._best_index: int | None = None  # 0-indexed in self._history

    @property
    def n_updates(self) -> int:
        return len(self._history)

    @property
    def best_value(self) -> float | None:
        return self._best_value

    @property
    def best_index(self) -> int | None:
        """0-indexed position in history of the best value (None if empty)."""
        return self._best_index

    @property
    def history(self) -> list[float]:
        """Defensive copy of the metric history."""
        return list(self._history)

    def state(self) -> EarlyStopState:
        return EarlyStopState(
            n_updates=self.n_updates,
            history=self.history,
            best_value=self._best_value,
            best_index=self._best_index,
            last_value=self._history[-1] if self._history else None,
        )

    def update(self, metric_value: float) -> bool:
        """Append ``metric_value`` and return True iff training should stop.

        The stop rule:
        - never stop while ``n_updates < min_evals`` (warmup);
        - otherwise, stop iff the best value is at least ``patience_evals``
          evals in the past (i.e. ``n_updates - 1 - best_index >=
          patience_evals``).

        ``min_delta`` enters via the "did this update improve?" predicate:
        a new value ``v`` improves on running ``best b`` iff
        ``v < b - min_delta``. Sub-``min_delta`` improvements are not
        considered progress and don't reset the patience window.
        """
        idx = len(self._history)
        self._history.append(float(metric_value))

        if (
            self._best_value is None
            or metric_value < self._best_value - self._min_delta
        ):
            self._best_value = float(metric_value)
            self._best_index = idx

        # Warmup: don't even check until we've seen enough evals.
        if self.n_updates < self._min_evals:
            return False

        # Stop iff the best is at least patience_evals updates in the past.
        # n_updates - 1 - best_index = number of evals since (and not
        # including) the best. When that equals patience_evals, we've
        # observed patience_evals consecutive non-improving evals after
        # the best.
        assert self._best_index is not None
        evals_since_best = (self.n_updates - 1) - self._best_index
        return evals_since_best >= self._patience_evals
