"""Tests for ``scripts/beam_eval_model.py`` — single-checkpoint beam-eval CLI.

Mirrors ``tests/experiments/test_eval_3x3.py``'s tiny-net-on-CPU pattern.
We test the *API shape* (CLI exit codes, JSON schema, determinism, config
resolution), not the eval's downstream numerical quality.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from rubik.cube.spec import CUBE_3X3
from rubik.model.network import ValueNet

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "beam_eval_model.py"


# ---------------------------------------------------------------------------
# Tiny-net + config fixture: writes a minimal config.yaml + checkpoint pair
# the script can load. Body width / blocks / norm all match what we
# instantiate the ValueNet with.
# ---------------------------------------------------------------------------


def _tiny_config_kwargs() -> dict:
    """Minimal valid DAVIConfig kwargs — every field is required (no defaults).

    ``max_scramble_depth=14`` matches the legacy default. Under the
    min(yaml, training) auto-derive semantics, eval depth caps at the
    YAML schedule length, so the tiny eval config's length-3 schedule
    produces 3 walk depths. Tests that need to exercise auto-derive at
    a different effective depth override this kwarg.
    """
    return {
        "max_scramble_depth": 14,
        "max_scramble_depth_initial": 0,
        "max_scramble_depth_ramp_steps": 0,
        "batch_size": 32,
        "n_steps": 100,
        "learning_rate": 1e-3,
        "target_sync_interval": 500,
        "body_widths": [64, 32],  # YAML list — DAVIConfig.from_dict tuples it.
        "n_residual_blocks": 1,
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


@pytest.fixture
def tiny_run(tmp_path: Path) -> dict:
    """Build tiny config.yaml + checkpoint .pt pair under ``tmp_path``."""
    run_dir = tmp_path / "tiny-run"
    run_dir.mkdir()

    cfg_kwargs = _tiny_config_kwargs()
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_kwargs, sort_keys=False))

    net = ValueNet(
        CUBE_3X3,
        body_widths=tuple(cfg_kwargs["body_widths"]),
        n_residual_blocks=cfg_kwargs["n_residual_blocks"],
        normalization=cfg_kwargs["normalization"],
    )
    net.eval()
    ckpt_path = run_dir / "net_final.pt"
    torch.save({"net_state": net.state_dict()}, ckpt_path)

    return {
        "run_dir": run_dir,
        "config": cfg_path,
        "checkpoint": ckpt_path,
    }


@pytest.fixture
def tiny_eval_config(tmp_path: Path) -> Path:
    """Cheap eval-config YAML so unit tests don't run the full fast/default schedule."""
    cfg_path = tmp_path / "tiny_eval.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "description": "tiny test config",
                "n_per_depth": [4, 4, 4],
                "beam_width": 4,
                "precision": "fp32",
                "seed": 0,
            },
            sort_keys=False,
        )
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Smoke + schema tests
# ---------------------------------------------------------------------------


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_beam_eval_model_smoke(tiny_run, tiny_eval_config, tmp_path):
    """End-to-end: script exits 0, writes JSON with the right top-level keys."""
    out_path = tmp_path / "out.json"
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_eval_config),
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, f"beam_eval_model failed: stderr={result.stderr!r}"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())

    # Top-level keys (flat schema).
    for key in (
        "model",
        "config_name",
        "config_description",
        "device",
        "precision",
        "max_depth",
        "beam_width",
        "seed",
        "n_per_depth",
        "wall_time_seconds",
        "per_walk_depth",
        "states_scored",
    ):
        assert key in payload, f"missing top-level key: {key}"
    assert payload["max_depth"] == 3
    assert payload["beam_width"] == 4
    assert payload["precision"] == "fp32"
    assert payload["seed"] == 0
    assert payload["device"] == "cpu"
    assert payload["n_per_depth"] == [4, 4, 4]
    # No v_star_results without --include-v-star
    assert "v_star_results" not in payload


def test_beam_eval_model_schema_keys(tiny_run, tiny_eval_config, tmp_path):
    """``per_walk_depth`` is a list of d/n/solve_rate/avg_solve_len dicts."""
    out_path = tmp_path / "out.json"
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_eval_config),
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())

    pwd = payload["per_walk_depth"]
    assert isinstance(pwd, list)
    assert len(pwd) == 3
    for d_idx, item in enumerate(pwd, start=1):
        assert isinstance(item, dict)
        assert item["d"] == d_idx
        assert isinstance(item["solve_rate"], float)
        assert 0.0 <= item["solve_rate"] <= 1.0
        assert item["avg_solve_len"] is None or isinstance(item["avg_solve_len"], float)
        assert item["n"] == 4

    assert isinstance(payload["wall_time_seconds"], float)
    assert payload["wall_time_seconds"] > 0
    assert isinstance(payload["states_scored"], int)
    assert payload["states_scored"] == 4 * 3  # sum of n_per_depth


def test_beam_eval_model_deterministic_seed(tiny_run, tiny_eval_config, tmp_path):
    """Two runs with the same --seed produce bit-identical solve_rate values."""
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    common = [
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_eval_config),
        "--device",
        "cpu",
        "--seed",
        "42",
    ]
    r1 = _run_script(*common, "--output-json", str(out_a))
    assert r1.returncode == 0, r1.stderr
    r2 = _run_script(*common, "--output-json", str(out_b))
    assert r2.returncode == 0, r2.stderr

    pa = json.loads(out_a.read_text())
    pb = json.loads(out_b.read_text())
    a_pwd = pa["per_walk_depth"]
    b_pwd = pb["per_walk_depth"]
    assert len(a_pwd) == len(b_pwd)
    for ai, bi in zip(a_pwd, b_pwd, strict=True):
        assert ai["d"] == bi["d"]
        assert ai["solve_rate"] == bi["solve_rate"], (
            f"d={ai['d']} solve_rate diverged: {ai['solve_rate']} vs {bi['solve_rate']}"
        )


def test_beam_eval_model_include_v_star(tiny_run, tiny_eval_config, tmp_path):
    """With --include-v-star the JSON has v_star_results.per_v_star.

    Skips if the bounded oracle cache isn't on disk (CI without data/).
    """
    oracle_path = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"
    if not oracle_path.exists():
        pytest.skip(f"oracle cache not present: {oracle_path}")

    # Slim the schedule to make this fast.
    short_cfg = tmp_path / "short.yaml"
    short_cfg.write_text(
        yaml.safe_dump(
            {
                "description": "short",
                "n_per_depth": [4, 4],
                "beam_width": 4,
                "precision": "fp32",
                "seed": 0,
                "n_per_v_star_layer": 4,
            },
            sort_keys=False,
        )
    )

    out_path = tmp_path / "out.json"
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        str(short_cfg),
        "--device",
        "cpu",
        "--include-v-star",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())

    assert "v_star_results" in payload
    pvs = payload["v_star_results"]["per_v_star"]
    assert isinstance(pvs, list)
    assert len(pvs) > 0
    for item in pvs:
        assert "v" in item and isinstance(item["v"], int)
        assert item["v"] >= 1  # V*=0 is skipped by design
        assert isinstance(item["solve_rate"], float)
        assert 0.0 <= item["solve_rate"] <= 1.0
        assert item["avg_solve_len"] is None or isinstance(item["avg_solve_len"], float)
        assert isinstance(item["mae"], float)
        assert "n" in item


def test_beam_eval_model_default_output_path(tiny_run, tiny_eval_config):
    """Omitting --output-json: JSON lands at <model-dir>/<stem>_eval_<config>.json."""
    expected = tiny_run["run_dir"] / f"net_final_eval_{tiny_eval_config.stem}.json"
    if expected.exists():
        expected.unlink()
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_eval_config),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    assert expected.exists(), f"expected default output at {expected}"


def test_beam_eval_model_named_config_resolution(tiny_run, tmp_path):
    """``--config fast`` resolves to scripts/eval-configs/fast.yaml."""
    # We don't actually want to run the full fast schedule (too slow for unit
    # test); instead, just override max-depth=2 and verify it resolves.
    out_path = tmp_path / "out.json"
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        "fast",
        "--max-depth",
        "2",
        "--width",
        "4",
        "--precision",
        "fp32",
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, f"resolution failed: stderr={result.stderr!r}"
    payload = json.loads(out_path.read_text())
    assert payload["config_name"] == "fast"
    assert payload["max_depth"] == 2
    assert payload["beam_width"] == 4
    assert payload["precision"] == "fp32"


def test_beam_eval_model_overrides_win_over_yaml(tiny_run, tiny_eval_config, tmp_path):
    """CLI --width / --precision / --seed override YAML fields."""
    out_path = tmp_path / "out.json"
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_eval_config),
        "--width",
        "8",  # YAML has 4
        "--precision",
        "fp32",
        "--seed",
        "99",  # YAML has 0
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())
    assert payload["beam_width"] == 8
    assert payload["seed"] == 99


def test_beam_eval_model_max_depth_slices_schedule(
    tiny_run, tiny_eval_config, tmp_path
):
    """``--max-depth 2`` slices n_per_depth[:2] and runs only depths 1..2."""
    out_path = tmp_path / "out.json"
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_eval_config),
        "--max-depth",
        "2",
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())
    assert payload["max_depth"] == 2
    assert len(payload["per_walk_depth"]) == 2
    assert payload["n_per_depth"] == [4, 4]


def test_beam_eval_model_max_depth_too_large_errors(
    tiny_run, tiny_eval_config, tmp_path
):
    """``--max-depth N`` where N > schedule length → ValueError.

    Under min(yaml, training) auto-derive semantics, callers without
    --max-depth pre-clamp against the YAML length. Explicit
    --max-depth N past the schedule has no sample-count entries; we
    refuse rather than invent values.
    """
    out_path = tmp_path / "out.json"
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_eval_config),
        "--max-depth",
        "5",  # schedule has 3 entries
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode != 0
    assert "exceeds n_per_depth schedule" in (result.stdout + result.stderr)


def test_beam_eval_model_auto_derive_min_training_lt_yaml(tmp_path):
    """Single-checkpoint auto-derive: training=2 < yaml=3 → effective=2."""
    run_dir = tmp_path / "tiny-run"
    run_dir.mkdir()
    cfg = _tiny_config_kwargs()
    cfg["max_scramble_depth"] = 2
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    net = ValueNet(
        CUBE_3X3,
        body_widths=tuple(cfg["body_widths"]),
        n_residual_blocks=cfg["n_residual_blocks"],
        normalization=cfg["normalization"],
    )
    net.eval()
    ckpt_path = run_dir / "net_final.pt"
    torch.save({"net_state": net.state_dict()}, ckpt_path)

    eval_cfg_path = tmp_path / "tiny_eval.yaml"
    eval_cfg_path.write_text(
        yaml.safe_dump(
            {
                "description": "tiny",
                "n_per_depth": [4, 4, 4],
                "beam_width": 4,
                "precision": "fp32",
                "seed": 0,
            },
            sort_keys=False,
        )
    )

    out_path = tmp_path / "out.json"
    result = _run_script(
        str(ckpt_path),
        "--config",
        str(eval_cfg_path),
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    assert "auto-derived max_depth=2" in log
    assert "yaml=3" in log and "training=2" in log
    assert "source=config.yaml" in log
    payload = json.loads(out_path.read_text())
    assert payload["max_depth"] == 2
    assert len(payload["per_walk_depth"]) == 2


def test_beam_eval_model_unknown_config_errors(tiny_run, tmp_path):
    """``--config nonexistent`` → FileNotFoundError surfaces both name + path."""
    out_path = tmp_path / "out.json"
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        "this_config_does_not_exist_anywhere",
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "this_config_does_not_exist_anywhere" in combined
    assert "eval-configs" in combined  # search path


def test_beam_eval_model_carries_bundle_metadata(tiny_run, tiny_eval_config, tmp_path):
    """Bundle with step/wall_time_seconds/wandb_run_id → payload mirrors them.

    The eval consumer surfaces these provenance fields in the JSON
    output so downstream readers (and beam_eval_run.py) can prefer the
    bundle's authoritative step over filename-extracted step.
    """
    # Replace the fixture's bare-bundle checkpoint with a fully-populated
    # one carrying the provenance fields.
    cfg_kwargs = _tiny_config_kwargs()
    net = ValueNet(
        CUBE_3X3,
        body_widths=tuple(cfg_kwargs["body_widths"]),
        n_residual_blocks=cfg_kwargs["n_residual_blocks"],
        normalization=cfg_kwargs["normalization"],
    )
    net.eval()
    target = ValueNet(
        CUBE_3X3,
        body_widths=tuple(cfg_kwargs["body_widths"]),
        n_residual_blocks=cfg_kwargs["n_residual_blocks"],
        normalization=cfg_kwargs["normalization"],
    )
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    ckpt_path: Path = tiny_run["checkpoint"]
    torch.save(
        {
            "net_state": net.state_dict(),
            "target_net_state": target.state_dict(),
            "optimizer_state": optim.state_dict(),
            "step": 7777,
            "wall_time_seconds": 33.25,
            "wandb_run_id": "wandbrun42",
        },
        ckpt_path,
    )

    out_path = tmp_path / "out.json"
    result = _run_script(
        str(ckpt_path),
        "--config",
        str(tiny_eval_config),
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())
    assert payload["checkpoint_step"] == 7777
    assert payload["checkpoint_wall_time_seconds"] == pytest.approx(33.25)
    assert payload["checkpoint_wandb_run_id"] == "wandbrun42"


def test_beam_eval_model_missing_bundle_metadata_yields_none(
    tiny_run, tiny_eval_config, tmp_path
):
    """Bundle with only ``net_state`` + ``step`` → metadata fields = None."""
    cfg_kwargs = _tiny_config_kwargs()
    net = ValueNet(
        CUBE_3X3,
        body_widths=tuple(cfg_kwargs["body_widths"]),
        n_residual_blocks=cfg_kwargs["n_residual_blocks"],
        normalization=cfg_kwargs["normalization"],
    )
    net.eval()
    ckpt_path: Path = tiny_run["checkpoint"]
    # Bundle (has net_state) but lacks the provenance fields and even
    # ``step``. checkpoint_step should come back as None.
    torch.save({"net_state": net.state_dict()}, ckpt_path)

    out_path = tmp_path / "out.json"
    result = _run_script(
        str(ckpt_path),
        "--config",
        str(tiny_eval_config),
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())
    # Keys ARE present (top-level schema is stable) but values are None.
    assert "checkpoint_step" in payload
    assert "checkpoint_wall_time_seconds" in payload
    assert "checkpoint_wandb_run_id" in payload
    assert payload["checkpoint_step"] is None
    assert payload["checkpoint_wall_time_seconds"] is None
    assert payload["checkpoint_wandb_run_id"] is None


def test_beam_eval_model_path_config(tiny_run, tiny_eval_config, tmp_path):
    """``--config /path/to/file.yaml`` accepts a literal filesystem path."""
    out_path = tmp_path / "out.json"
    # Use absolute path — the loader should treat anything containing "/" as
    # a path, not a name lookup.
    result = _run_script(
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_eval_config.resolve()),
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, f"path config failed: stderr={result.stderr!r}"
    payload = json.loads(out_path.read_text())
    assert payload["config_name"] == tiny_eval_config.stem
