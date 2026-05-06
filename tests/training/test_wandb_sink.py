"""Tests for `rubik.training.wandb_sink.WandbAdapter`.

Verifies that the adapter routes fields into the W&B namespace policy
described in the module docstring of ``wandb_sink``: ``train/`` for step
records, ``eval/`` for eval records (with nested dicts flattened and
``_d<N>`` patterns regex-regrouped), ``checkpoint/`` for checkpoints,
``run/`` for run-start/end, and pass-through for records without an
``event`` field.

The wrapped W&B run is a ``MagicMock`` — the duck-typed Protocol shape
lets these tests run without wandb actually installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from rubik.training.wandb_sink import WandbAdapter


def test_step_event_routes_under_train_namespace():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "step",
            "step": 100,
            "loss": 0.123,
            "step_seconds": 0.245,
            "k_max": 28,
        }
    )

    fake_run.log.assert_called_once()
    (sent,), _ = fake_run.log.call_args
    # step + event preserved at top level
    assert sent["step"] == 100
    assert sent["event"] == "step"
    # other fields routed under train/
    assert sent["train/loss"] == 0.123
    assert sent["train/step_seconds"] == 0.245
    assert sent["train/k_max"] == 28
    # raw (unprefixed) field names should NOT remain
    assert "loss" not in sent
    assert "step_seconds" not in sent
    assert "k_max" not in sent


def test_eval_event_routes_scalars_under_eval_namespace():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "eval",
            "step": 1000,
            "val_mae": 3.14,
            "macro_mae": 3.15,
            "pred_mean": 4.6,
            "pred_std": 1.45,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["step"] == 1000
    assert sent["event"] == "eval"
    assert sent["eval/val_mae"] == 3.14
    assert sent["eval/macro_mae"] == 3.15
    assert sent["eval/pred_mean"] == 4.6
    assert sent["eval/pred_std"] == 1.45


def test_eval_event_flattens_per_depth_mae_dict():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "eval",
            "step": 1000,
            "per_depth_mae": {"1": 0.99, "2": 0.67, "14": 8.79},
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["eval/per_depth_mae/1"] == 0.99
    assert sent["eval/per_depth_mae/2"] == 0.67
    assert sent["eval/per_depth_mae/14"] == 8.79
    # Original nested dict should not survive under eval/per_depth_mae.
    assert "eval/per_depth_mae" not in sent
    # Nor at the top level.
    assert "per_depth_mae" not in sent


def test_eval_event_regroups_solve_rate_per_depth():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "eval",
            "step": 1000,
            "solve_rate_d1": 1.0,
            "solve_rate_d2": 0.95,
            "solve_rate_d3": 0.80,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["eval/solve_rate/d1"] == 1.0
    assert sent["eval/solve_rate/d2"] == 0.95
    assert sent["eval/solve_rate/d3"] == 0.80
    # Original keys should not survive.
    assert "solve_rate_d1" not in sent
    assert "eval/solve_rate_d1" not in sent


def test_eval_event_regroups_avg_solve_len_per_depth():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "eval",
            "step": 1000,
            "avg_solve_len_d1": 1.0,
            "avg_solve_len_d14": 9.16,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["eval/avg_solve_len/d1"] == 1.0
    assert sent["eval/avg_solve_len/d14"] == 9.16


def test_eval_event_full_record_combines_all_routings():
    """Smoke test: a realistic eval record gets fully routed in one call."""
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "eval",
            "step": 1000,
            "val_mae": 3.14,
            "macro_mae": 3.15,
            "per_depth_mae": {"1": 0.99, "2": 0.67},
            "solve_rate_d1": 1.0,
            "solve_rate_d2": 0.95,
            "avg_solve_len_d1": 1.0,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["step"] == 1000
    assert sent["event"] == "eval"
    assert sent["eval/val_mae"] == 3.14
    assert sent["eval/macro_mae"] == 3.15
    assert sent["eval/per_depth_mae/1"] == 0.99
    assert sent["eval/per_depth_mae/2"] == 0.67
    assert sent["eval/solve_rate/d1"] == 1.0
    assert sent["eval/solve_rate/d2"] == 0.95
    assert sent["eval/avg_solve_len/d1"] == 1.0


def test_checkpoint_event_routes_path_under_checkpoint_namespace():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log({"event": "checkpoint", "step": 5000, "path": "net_step_5000.pt"})

    (sent,), _ = fake_run.log.call_args
    assert sent["step"] == 5000
    assert sent["event"] == "checkpoint"
    assert sent["checkpoint/path"] == "net_step_5000.pt"
    assert "path" not in sent


def test_run_start_event_routes_under_run_namespace():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "run_start",
            "n_params": 13_213_697,
            "device": "mps",
            "n_steps": 30000,
            "seed": 42,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["event"] == "run_start"
    assert sent["run/n_params"] == 13_213_697
    assert sent["run/device"] == "mps"
    assert sent["run/n_steps"] == 30000
    assert sent["run/seed"] == 42


def test_run_end_event_routes_under_run_namespace():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log({"event": "run_end", "step": 30000, "final_macro_mae": 2.85})

    (sent,), _ = fake_run.log.call_args
    assert sent["step"] == 30000
    assert sent["event"] == "run_end"
    assert sent["run/final_macro_mae"] == 2.85


def test_no_event_field_passes_through_unchanged():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log({"foo": 1, "bar": "baz"})

    fake_run.log.assert_called_once_with({"foo": 1, "bar": "baz"})


def test_log_forwards_to_wrapped_run_log_method():
    """Sanity: the adapter actually delegates to run.log, not e.g. .write."""
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log({"event": "step", "step": 1, "loss": 0.5})

    assert fake_run.log.call_count == 1
    # No other methods on the run should have been touched.
    fake_run.write.assert_not_called()
