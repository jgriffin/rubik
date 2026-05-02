"""Tests for `rubik.perf.bench` — timing primitive + bootstrap CI."""

import pytest
import torch

from rubik.perf import bootstrap_ci, time_op


def test_time_op_returns_correct_trial_count():
    timings = time_op(lambda: None, warmup=2, trials=7, device="cpu")
    assert len(timings) == 7
    assert all(t >= 0.0 for t in timings)


def test_time_op_warmup_runs_but_not_timed():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1

    timings = time_op(fn, warmup=5, trials=10, device="cpu")
    assert len(timings) == 10
    assert counter["n"] == 15


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_time_op_mps_sync_called(monkeypatch):
    sync_calls = {"n": 0}
    real_sync = torch.mps.synchronize

    def counting_sync():
        sync_calls["n"] += 1
        real_sync()

    monkeypatch.setattr(torch.mps, "synchronize", counting_sync)
    time_op(lambda: None, warmup=3, trials=8, device="mps")
    assert sync_calls["n"] == 2 * 8


def test_time_op_raises_on_mps_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps"):
        time_op(lambda: None, warmup=1, trials=2, device="mps")


def test_bootstrap_ci_basic_shape():
    samples = list(range(100))
    median, lo, hi = bootstrap_ci(samples, seed=0)
    assert lo <= median <= hi
    assert median == pytest.approx(49.5, abs=0.5)


def test_bootstrap_ci_deterministic_under_seed():
    samples = [float(x) for x in range(50)]
    a = bootstrap_ci(samples, seed=42)
    b = bootstrap_ci(samples, seed=42)
    assert a == b
