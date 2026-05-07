"""Tests for ``DAVIConfig`` schema + curriculum schedule."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rubik.training.config import DAVIConfig


def _base_kwargs(**overrides) -> dict:
    """Minimal valid DAVIConfig kwargs; tests override individual fields."""
    base = {
        "max_scramble_depth": 14,
        "max_scramble_depth_initial": 0,
        "max_scramble_depth_ramp_steps": 0,
        "batch_size": 32,
        "n_steps": 100,
        "learning_rate": 1e-3,
        "target_sync_interval": 500,
        "body_widths": (64, 32),
        "n_residual_blocks": 2,
        "normalization": "bn",
        "log_every": 10,
        "eval_every": 500,
        "checkpoint_every": 100,
        "seed": 0,
        "device": "cpu",
        "early_stop_enabled": False,
        "early_stop_metric": "macro_v_star_mae",
        "early_stop_patience_evals": 0,
        "early_stop_min_evals": 0,
        "early_stop_min_delta": 0.0,
    }
    base.update(overrides)
    return base


def test_flat_schedule_returns_max_at_every_step() -> None:
    cfg = DAVIConfig(**_base_kwargs(max_scramble_depth=20))
    for step in (1, 100, 1_000, 30_000):
        assert cfg.current_k_max(step) == 20


def test_ramp_schedule_starts_at_initial_and_reaches_final() -> None:
    cfg = DAVIConfig(
        **_base_kwargs(
            max_scramble_depth=28,
            max_scramble_depth_initial=14,
            max_scramble_depth_ramp_steps=15_000,
        )
    )
    assert cfg.current_k_max(1) == 14  # start of ramp
    assert cfg.current_k_max(15_000) == 28  # end of ramp
    assert cfg.current_k_max(20_000) == 28  # past ramp clamps to final
    assert cfg.current_k_max(30_000) == 28


def test_ramp_schedule_is_monotonic_non_decreasing() -> None:
    cfg = DAVIConfig(
        **_base_kwargs(
            max_scramble_depth=28,
            max_scramble_depth_initial=14,
            max_scramble_depth_ramp_steps=15_000,
        )
    )
    prev = cfg.current_k_max(1)
    for step in range(1, 30_001, 100):
        cur = cfg.current_k_max(step)
        assert cur >= prev
        assert 14 <= cur <= 28
        prev = cur


def test_ramp_midpoint_is_between_endpoints() -> None:
    cfg = DAVIConfig(
        **_base_kwargs(
            max_scramble_depth=28,
            max_scramble_depth_initial=14,
            max_scramble_depth_ramp_steps=15_000,
        )
    )
    mid = cfg.current_k_max(7_500)
    assert 14 < mid < 28


def test_step_zero_or_negative_raises() -> None:
    cfg = DAVIConfig(**_base_kwargs())
    with pytest.raises(ValueError):
        cfg.current_k_max(0)
    with pytest.raises(ValueError):
        cfg.current_k_max(-1)


def test_negative_ramp_steps_raises() -> None:
    with pytest.raises(ValueError, match="ramp_steps must be >= 0"):
        DAVIConfig(**_base_kwargs(max_scramble_depth_ramp_steps=-5))


def test_initial_below_one_raises_when_ramping() -> None:
    with pytest.raises(ValueError, match="initial must be >= 1 when ramp_steps > 0"):
        DAVIConfig(
            **_base_kwargs(
                max_scramble_depth=28,
                max_scramble_depth_initial=0,
                max_scramble_depth_ramp_steps=10,
            )
        )


def test_initial_above_max_raises() -> None:
    with pytest.raises(ValueError, match="initial must be <= max_scramble_depth"):
        DAVIConfig(
            **_base_kwargs(
                max_scramble_depth=14,
                max_scramble_depth_initial=20,
                max_scramble_depth_ramp_steps=10,
            )
        )


def test_initial_equal_to_max_with_ramp_is_flat() -> None:
    cfg = DAVIConfig(
        **_base_kwargs(
            max_scramble_depth=20,
            max_scramble_depth_initial=20,
            max_scramble_depth_ramp_steps=1_000,
        )
    )
    for step in (1, 500, 1_000, 2_000):
        assert cfg.current_k_max(step) == 20


def test_yaml_round_trip_preserves_schedule(tmp_path: Path) -> None:
    cfg = DAVIConfig(
        **_base_kwargs(
            max_scramble_depth=28,
            max_scramble_depth_initial=14,
            max_scramble_depth_ramp_steps=15_000,
        )
    )
    yaml_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(yaml_path)
    loaded = DAVIConfig.from_yaml(yaml_path)
    assert loaded == cfg


def test_yaml_missing_curriculum_field_raises(tmp_path: Path) -> None:
    """Old-schema yaml without the new curriculum fields surfaces loudly."""
    yaml_path = tmp_path / "stale.yaml"
    minimal = {
        "max_scramble_depth": 14,
        "batch_size": 32,
        "n_steps": 100,
        "learning_rate": 1e-3,
        "target_sync_interval": 500,
        "body_widths": [64, 32],
        "n_residual_blocks": 2,
        "normalization": "bn",
        "log_every": 10,
        "eval_every": 50,
        "checkpoint_every": 100,
        "seed": 0,
        "device": "cpu",
    }
    yaml_path.write_text(yaml.safe_dump(minimal))
    with pytest.raises(TypeError):
        DAVIConfig.from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# Early-stop fields (P2a)
# ---------------------------------------------------------------------------


def test_early_stop_fields_round_trip(tmp_path: Path) -> None:
    """All 5 early-stop fields survive yaml round-trip."""
    cfg = DAVIConfig(
        **_base_kwargs(
            target_sync_interval=500,
            eval_every=500,
            early_stop_enabled=True,
            early_stop_metric="macro_v_star_mae",
            early_stop_patience_evals=12,
            early_stop_min_evals=4,
            early_stop_min_delta=0.001,
        )
    )
    yaml_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(yaml_path)
    loaded = DAVIConfig.from_yaml(yaml_path)
    assert loaded == cfg
    assert loaded.early_stop_enabled is True
    assert loaded.early_stop_metric == "macro_v_star_mae"
    assert loaded.early_stop_patience_evals == 12
    assert loaded.early_stop_min_evals == 4
    assert loaded.early_stop_min_delta == 0.001


@pytest.mark.parametrize(
    "missing_field",
    [
        "early_stop_enabled",
        "early_stop_metric",
        "early_stop_patience_evals",
        "early_stop_min_evals",
        "early_stop_min_delta",
    ],
)
def test_yaml_missing_early_stop_field_raises(
    tmp_path: Path, missing_field: str
) -> None:
    """Each new early-stop field must be present in yaml (no defaults)."""
    kwargs = _base_kwargs(
        target_sync_interval=500,
        eval_every=500,
        early_stop_enabled=True,
        early_stop_patience_evals=12,
        early_stop_min_evals=4,
        early_stop_min_delta=0.001,
    )
    # Convert tuple → list for yaml safe_dump.
    kwargs["body_widths"] = list(kwargs["body_widths"])
    del kwargs[missing_field]
    yaml_path = tmp_path / "stale.yaml"
    yaml_path.write_text(yaml.safe_dump(kwargs))
    with pytest.raises(TypeError):
        DAVIConfig.from_yaml(yaml_path)


def test_early_stop_alignment_passes_when_equal() -> None:
    """eval_every == target_sync_interval is the canonical aligned case."""
    cfg = DAVIConfig(
        **_base_kwargs(
            target_sync_interval=500,
            eval_every=500,
            early_stop_enabled=True,
        )
    )
    assert cfg.eval_every == cfg.target_sync_interval


def test_early_stop_alignment_passes_when_clean_multiple() -> None:
    """eval_every a clean multiple of target_sync_interval is also allowed."""
    cfg = DAVIConfig(
        **_base_kwargs(
            target_sync_interval=500,
            eval_every=1000,
            early_stop_enabled=True,
        )
    )
    assert cfg.eval_every % cfg.target_sync_interval == 0


def test_early_stop_alignment_fails_when_misaligned() -> None:
    """Misaligned eval_every / target_sync_interval errors at load time."""
    with pytest.raises(ValueError, match="eval_every"):
        DAVIConfig(
            **_base_kwargs(
                target_sync_interval=500,
                eval_every=750,  # not a multiple of 500
                early_stop_enabled=True,
            )
        )


def test_early_stop_alignment_skipped_when_disabled() -> None:
    """Misalignment is fine when early-stop is disabled."""
    cfg = DAVIConfig(
        **_base_kwargs(
            target_sync_interval=500,
            eval_every=750,
            early_stop_enabled=False,
        )
    )
    assert cfg.early_stop_enabled is False
