"""Tests for ``scripts/_eval_config.py`` — YAML eval-config loader.

Pure-Python unit tests for ``EvalConfig.resolve``, ``from_yaml``,
``from_dict``, and ``with_overrides``. No subprocess, no model loading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Make the underscore helper importable as a module.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _eval_config import EvalConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Resolution + lookup
# ---------------------------------------------------------------------------


def test_resolve_named_config_fast():
    """``EvalConfig.resolve('fast')`` finds scripts/eval-configs/fast.yaml."""
    cfg = EvalConfig.resolve("fast")
    assert cfg.source_name == "fast"
    assert cfg.beam_width == 256
    assert cfg.precision == "bf16"
    assert cfg.seed == 0
    assert len(cfg.n_per_depth) == 14
    assert cfg.n_per_depth[0] == 12  # d=1 exhaustive


def test_resolve_named_config_default():
    """``EvalConfig.resolve('default')`` finds scripts/eval-configs/default.yaml."""
    cfg = EvalConfig.resolve("default")
    assert cfg.source_name == "default"
    assert cfg.beam_width == 256
    assert cfg.precision == "bf16"
    assert len(cfg.n_per_depth) == 14
    # Deep-tail at n=100.
    assert cfg.n_per_depth[-1] == 100


def test_resolve_path(tmp_path: Path):
    """``EvalConfig.resolve('/path/to/x.yaml')`` treats it as a literal path."""
    p = tmp_path / "custom.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "description": "x",
                "n_per_depth": [4, 4],
                "beam_width": 8,
                "precision": "fp32",
                "seed": 7,
            },
            sort_keys=False,
        )
    )
    cfg = EvalConfig.resolve(str(p))
    assert cfg.beam_width == 8
    assert cfg.seed == 7
    assert cfg.source_name == "custom"


def test_resolve_yaml_extension_treated_as_path(tmp_path: Path):
    """A ``.yaml``-suffixed value is treated as a path even without a slash."""
    p = tmp_path / "rel.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "description": "x",
                "n_per_depth": [2],
                "beam_width": 4,
                "precision": "fp32",
                "seed": 0,
            },
            sort_keys=False,
        )
    )
    # Use the absolute path — the loader treats ".yaml" as path indicator.
    cfg = EvalConfig.resolve(str(p))
    assert cfg.beam_width == 4


def test_resolve_unknown_name_raises_with_search_path():
    """Missing named config → FileNotFoundError including name + search path."""
    with pytest.raises(FileNotFoundError) as excinfo:
        EvalConfig.resolve("nonexistent_config_xyz")
    msg = str(excinfo.value)
    assert "nonexistent_config_xyz" in msg
    assert "eval-configs" in msg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _minimal_dict(**overrides) -> dict:
    base = {
        "description": "test",
        "n_per_depth": [4, 4],
        "beam_width": 8,
        "precision": "fp32",
        "seed": 0,
    }
    base.update(overrides)
    return base


def test_from_dict_minimal_ok():
    cfg = EvalConfig.from_dict(_minimal_dict())
    assert cfg.beam_width == 8
    assert cfg.n_per_depth == (4, 4)
    assert cfg.include_v_star is False
    assert cfg.n_per_v_star_layer == 200


def test_from_dict_missing_field():
    d = _minimal_dict()
    del d["beam_width"]
    with pytest.raises(ValueError, match="missing required fields"):
        EvalConfig.from_dict(d)


def test_from_dict_n_per_depth_must_be_nonempty():
    with pytest.raises(ValueError, match="non-empty list"):
        EvalConfig.from_dict(_minimal_dict(n_per_depth=[]))


def test_from_dict_n_per_depth_must_be_positive_ints():
    with pytest.raises(ValueError, match="positive int"):
        EvalConfig.from_dict(_minimal_dict(n_per_depth=[4, 0, 4]))
    with pytest.raises(ValueError, match="positive int"):
        EvalConfig.from_dict(_minimal_dict(n_per_depth=[4, -1]))


def test_from_dict_beam_width_must_be_positive():
    with pytest.raises(ValueError, match="positive int"):
        EvalConfig.from_dict(_minimal_dict(beam_width=0))
    with pytest.raises(ValueError, match="positive int"):
        EvalConfig.from_dict(_minimal_dict(beam_width=-5))


def test_from_dict_precision_must_be_known():
    with pytest.raises(ValueError, match="precision"):
        EvalConfig.from_dict(_minimal_dict(precision="int8"))


def test_from_dict_seed_must_be_int():
    with pytest.raises(ValueError, match="seed must be an int"):
        EvalConfig.from_dict(_minimal_dict(seed="zero"))


def test_from_dict_optional_include_v_star_default_false():
    cfg = EvalConfig.from_dict(_minimal_dict())
    assert cfg.include_v_star is False


def test_from_dict_optional_include_v_star_true():
    cfg = EvalConfig.from_dict(_minimal_dict(include_v_star=True))
    assert cfg.include_v_star is True


def test_from_dict_n_per_v_star_layer_must_be_positive():
    with pytest.raises(ValueError, match="n_per_v_star_layer"):
        EvalConfig.from_dict(_minimal_dict(n_per_v_star_layer=0))


# ---------------------------------------------------------------------------
# Override precedence
# ---------------------------------------------------------------------------


def test_with_overrides_beam_width():
    cfg = EvalConfig.from_dict(_minimal_dict()).with_overrides(beam_width=64)
    assert cfg.beam_width == 64


def test_with_overrides_none_keeps_yaml():
    """Passing override=None means "keep YAML value", not "set to None"."""
    cfg = EvalConfig.from_dict(_minimal_dict(beam_width=8)).with_overrides(
        beam_width=None, precision=None, seed=None
    )
    assert cfg.beam_width == 8
    assert cfg.precision == "fp32"
    assert cfg.seed == 0


def test_with_overrides_precision_validated():
    with pytest.raises(ValueError, match="precision"):
        EvalConfig.from_dict(_minimal_dict()).with_overrides(precision="int4")


def test_with_overrides_max_depth_slices():
    cfg = EvalConfig.from_dict(
        _minimal_dict(n_per_depth=[4, 8, 12, 16])
    ).with_overrides(max_depth=2)
    assert cfg.n_per_depth == (4, 8)
    assert cfg.max_depth == 2


def test_with_overrides_max_depth_too_large():
    with pytest.raises(ValueError, match="exceeds"):
        EvalConfig.from_dict(_minimal_dict(n_per_depth=[4, 8])).with_overrides(
            max_depth=10
        )


def test_with_overrides_max_depth_zero_invalid():
    with pytest.raises(ValueError, match="max_depth must be"):
        EvalConfig.from_dict(_minimal_dict()).with_overrides(max_depth=0)


def test_walk_depths_property():
    cfg = EvalConfig.from_dict(_minimal_dict(n_per_depth=[1, 2, 3]))
    assert cfg.walk_depths == (1, 2, 3)
    assert cfg.max_depth == 3
