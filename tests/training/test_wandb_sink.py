"""Tests for `rubik.training.wandb_sink.WandbAdapter`.

Verifies that the adapter routes fields into the W&B namespace policy
described in the module docstring of ``wandb_sink``: ``train/`` for step
records, ``value/`` for ``event="eval"`` (forward-pass value-net eval),
``beam_walk/`` for ``event="beam_eval_walk"``, ``beam_v_star/`` for
``event="beam_eval_v_star"``, ``checkpoint/`` for checkpoints, ``run/``
for run-start/end, and pass-through for records without an ``event``
field. Also verifies zero-padding of ``d{N}`` keys for natural-sort in
W&B.

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


def test_eval_event_routes_scalars_under_value_namespace():
    """`event="eval"` records (from `value_eval`) land under `value/`."""
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "eval",
            "step": 1000,
            "macro_v_star_mae": 3.14,
            "pred_mean": 4.6,
            "pred_std": 1.45,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["step"] == 1000
    assert sent["event"] == "eval"
    assert sent["value/macro_v_star_mae"] == 3.14
    assert sent["value/pred_mean"] == 4.6
    assert sent["value/pred_std"] == 1.45
    # No `eval/` prefix any more.
    assert not any(k.startswith("eval/") for k in sent)


def test_eval_event_flattens_per_depth_mae_dict():
    """Nested dicts (2x2-shape `per_depth_mae`) flatten under `value/`."""
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
    assert sent["value/per_depth_mae/1"] == 0.99
    assert sent["value/per_depth_mae/2"] == 0.67
    assert sent["value/per_depth_mae/14"] == 8.79
    # Original nested dict should not survive.
    assert "value/per_depth_mae" not in sent
    assert "per_depth_mae" not in sent


def test_eval_event_regroups_solve_rate_per_depth_with_padding():
    """`solve_rate_d1..d3` regroup to `value/solve_rate/d01..d03`."""
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
    # Zero-padded for natural sort in W&B.
    assert sent["value/solve_rate/d01"] == 1.0
    assert sent["value/solve_rate/d02"] == 0.95
    assert sent["value/solve_rate/d03"] == 0.80
    # Original keys should not survive.
    assert "solve_rate_d1" not in sent
    assert "value/solve_rate/d1" not in sent


def test_eval_event_regroups_avg_solve_len_per_depth_with_padding():
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
    assert sent["value/avg_solve_len/d01"] == 1.0
    # d14 already two digits — unchanged.
    assert sent["value/avg_solve_len/d14"] == 9.16


def test_eval_event_pads_d_keys_in_flat_path_segments():
    """3x3 schema's flat slash-segment keys (d1, ..., d14) get padded."""
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "eval",
            "step": 1000,
            "v_star_mae/d1": 0.42,
            "v_star_mae/d6": 1.13,
            "per_walk_depth/d1/pred_mean": 1.05,
            "per_walk_depth/d14/pred_mean": 9.21,
            "per_walk_depth/d1/pred_std": 0.05,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["value/v_star_mae/d01"] == 0.42
    assert sent["value/v_star_mae/d06"] == 1.13
    assert sent["value/per_walk_depth/d01/pred_mean"] == 1.05
    assert sent["value/per_walk_depth/d14/pred_mean"] == 9.21
    assert sent["value/per_walk_depth/d01/pred_std"] == 0.05
    # Unpadded forms must NOT survive.
    assert "value/v_star_mae/d1" not in sent
    assert "value/per_walk_depth/d1/pred_mean" not in sent


def test_eval_event_full_record_combines_all_routings():
    """Smoke test: a realistic eval record gets fully routed in one call."""
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "eval",
            "step": 1000,
            "macro_v_star_mae": 3.14,
            "v_star_mae/d1": 0.42,
            "v_star_mae/d6": 1.13,
            "per_walk_depth/d1/pred_mean": 1.0,
            "per_walk_depth/d14/pred_mean": 9.16,
            "per_walk_depth/shallow/pred_mean": 2.5,
            "per_walk_depth/deep/pred_mean": 8.7,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["step"] == 1000
    assert sent["event"] == "eval"
    assert sent["value/macro_v_star_mae"] == 3.14
    assert sent["value/v_star_mae/d01"] == 0.42
    assert sent["value/v_star_mae/d06"] == 1.13
    assert sent["value/per_walk_depth/d01/pred_mean"] == 1.0
    assert sent["value/per_walk_depth/d14/pred_mean"] == 9.16
    # Banded aggregates pass through under value/ unchanged (no d-pad
    # since shallow/deep/mid aren't d-keys).
    assert sent["value/per_walk_depth/shallow/pred_mean"] == 2.5
    assert sent["value/per_walk_depth/deep/pred_mean"] == 8.7


def test_beam_eval_walk_event_routes_under_beam_walk_namespace():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "beam_eval_walk",
            "step": 16000,
            "solve_rate_d10": 1.0,
            "solve_rate_d14": 0.78,
            "avg_solve_len_d14": 13.2,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["step"] == 16000
    assert sent["event"] == "beam_eval_walk"
    assert sent["beam_walk/solve_rate/d10"] == 1.0
    assert sent["beam_walk/solve_rate/d14"] == 0.78
    assert sent["beam_walk/avg_solve_len/d14"] == 13.2


def test_beam_eval_v_star_event_routes_under_beam_v_star_namespace():
    fake_run = MagicMock()
    adapter = WandbAdapter(fake_run)

    adapter.log(
        {
            "event": "beam_eval_v_star",
            "step": 16000,
            "solve_rate_v1": 1.0,
            "solve_rate_v6": 1.0,
            "mae_v3": 0.21,
        }
    )

    (sent,), _ = fake_run.log.call_args
    assert sent["step"] == 16000
    assert sent["event"] == "beam_eval_v_star"
    # `_v{N}` is not a `_d{N}` pattern — passes through under the prefix
    # without regrouping. Padding regex only matches `/d{N}` segments.
    assert sent["beam_v_star/solve_rate_v1"] == 1.0
    assert sent["beam_v_star/solve_rate_v6"] == 1.0
    assert sent["beam_v_star/mae_v3"] == 0.21


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
