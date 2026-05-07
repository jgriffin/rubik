"""Tests for ``EarlyStopMonitor`` on synthetic histories."""

from __future__ import annotations

import pytest

from rubik.training.early_stop import EarlyStopMonitor


def _run(monitor: EarlyStopMonitor, values: list[float]) -> int | None:
    """Feed values into monitor; return the 0-indexed update at which it
    fired, or None if it never fired across the full sequence."""
    for i, v in enumerate(values):
        if monitor.update(v):
            return i
    return None


def test_constructor_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="patience_evals"):
        EarlyStopMonitor(patience_evals=0, min_evals=1, min_delta=0.0)
    with pytest.raises(ValueError, match="min_evals"):
        EarlyStopMonitor(patience_evals=1, min_evals=-1, min_delta=0.0)
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopMonitor(patience_evals=1, min_evals=0, min_delta=-1e-3)


def test_monotonically_decreasing_never_stops() -> None:
    """Improving metric → never stops within the run."""
    monitor = EarlyStopMonitor(patience_evals=3, min_evals=2, min_delta=0.01)
    values = [10.0 - 0.5 * i for i in range(20)]  # 10.0, 9.5, 9.0, ...
    fired = _run(monitor, values)
    assert fired is None
    assert monitor.best_value == pytest.approx(values[-1])
    assert monitor.best_index == len(values) - 1


def test_flat_from_step_zero_stops_at_min_evals_plus_patience() -> None:
    """Flat 5.0 → stops exactly when patience window after the first eval lapses.

    With min_evals=2, patience=3:
    - update 0 (idx=0): history=[5.0], best_index=0, n_updates=1<min_evals → no stop
    - update 1 (idx=1): n_updates=2, best_index=0, evals_since=1 < 3 → no stop
    - update 2 (idx=2): n_updates=3, evals_since=2 < 3 → no stop
    - update 3 (idx=3): n_updates=4, evals_since=3 == 3 → STOP
    """
    monitor = EarlyStopMonitor(patience_evals=3, min_evals=2, min_delta=0.01)
    fired = _run(monitor, [5.0] * 20)
    assert fired == 3


def test_decreasing_then_flat_stops_patience_after_plateau_begins() -> None:
    """Improves for K updates then plateaus → stops patience evals later."""
    monitor = EarlyStopMonitor(patience_evals=3, min_evals=2, min_delta=0.01)
    # 5 improvements (indices 0..4), then plateau
    values = [10.0, 9.0, 8.0, 7.0, 6.0] + [6.0] * 20
    fired = _run(monitor, values)
    # best_index = 4 (value=6.0 the first time). evals_since_best ≥ 3 first
    # at update index 7 (n_updates=8): evals_since = 7 - 4 = 3.
    assert fired == 7
    assert monitor.best_value == pytest.approx(6.0)
    assert monitor.best_index == 4


def test_sub_min_delta_improvements_behave_like_flat() -> None:
    """Decreasing but each delta < min_delta → behaves like flat."""
    monitor = EarlyStopMonitor(patience_evals=3, min_evals=2, min_delta=0.01)
    # Each step decreases by 0.001, well under min_delta=0.01.
    values = [5.0 - 0.001 * i for i in range(50)]
    fired = _run(monitor, values)
    # best_index stays at 0 (no later value beats 5.0 - 0.01 = 4.99); fires
    # at the same update as the all-flat case.
    assert fired == 3
    assert monitor.best_index == 0


def test_below_min_evals_never_stops_during_warmup() -> None:
    """During warmup (n_updates < min_evals), never stops even if flat."""
    monitor = EarlyStopMonitor(patience_evals=1, min_evals=10, min_delta=0.01)
    # Only 9 updates — strictly below warmup threshold. patience=1 would
    # otherwise fire on update 1.
    fired = _run(monitor, [5.0] * 9)
    assert fired is None
    # Once we cross min_evals, the rule kicks in immediately.
    assert monitor.update(5.0) is True  # this is the 10th update; fires now


def test_state_snapshot_is_consistent() -> None:
    monitor = EarlyStopMonitor(patience_evals=3, min_evals=2, min_delta=0.01)
    monitor.update(5.0)
    monitor.update(4.0)
    snap = monitor.state()
    assert snap.n_updates == 2
    assert snap.best_value == pytest.approx(4.0)
    assert snap.best_index == 1
    assert snap.last_value == pytest.approx(4.0)
    assert snap.history == [5.0, 4.0]
    # history is a defensive copy
    snap.history.append(999.0)
    assert monitor.history == [5.0, 4.0]


def test_min_delta_zero_treats_any_improvement_as_progress() -> None:
    """min_delta=0 means strict-less-than is the improvement test."""
    monitor = EarlyStopMonitor(patience_evals=3, min_evals=2, min_delta=0.0)
    values = [5.0, 5.0, 5.0, 4.999999, 5.0, 5.0, 5.0]
    fired = _run(monitor, values)
    # best_index moves to 3 on the tiny improvement; need 3 more flat
    # evals (indices 4, 5, 6) → fires at update 6.
    assert fired == 6
    assert monitor.best_index == 3
