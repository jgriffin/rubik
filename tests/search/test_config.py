"""BeamEvalConfig — yaml round-trip + validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rubik.search import BeamEvalConfig


def _make_config(**overrides) -> BeamEvalConfig:
    base = dict(
        checkpoint_path="experiments/davi-2x2/davi-baseline/runs/sync500_kmax20-30k/net_final.pt",
        body_widths=(4096, 1024),
        n_residual_blocks=4,
        normalization="bn",
        depths=(1, 2, 3),
        beam_widths=(1, 4, 16),
        n_per_depth=200,
        max_steps=20,
        seed=0,
        device="mps",
    )
    base.update(overrides)
    return BeamEvalConfig(**base)


def test_yaml_round_trip_preserves_tuple_fields(tmp_path: Path):
    """body_widths / depths / beam_widths must round-trip as tuples."""
    cfg = _make_config()
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = BeamEvalConfig.from_yaml(path)
    assert loaded == cfg
    assert isinstance(loaded.body_widths, tuple)
    assert isinstance(loaded.depths, tuple)
    assert isinstance(loaded.beam_widths, tuple)


def test_missing_field_raises_type_error(tmp_path: Path):
    """A YAML missing any required field surfaces as TypeError on construction.

    No silent defaults — the contract is that every YAML on disk carries
    the exact hyperparameters used.
    """
    cfg = _make_config()
    full = cfg.to_dict()
    del full["max_steps"]  # any required field works
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(full, sort_keys=False))
    with pytest.raises(TypeError, match="max_steps"):
        BeamEvalConfig.from_yaml(path)


def test_invalid_normalization_raises():
    with pytest.raises(ValueError, match="normalization"):
        _make_config(normalization="batchnorm")
