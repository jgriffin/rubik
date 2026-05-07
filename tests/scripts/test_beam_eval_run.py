"""Tests for ``scripts/beam_eval_run.py`` — parameterized beam-eval CLI.

Mirrors ``tests/experiments/test_eval_3x3.py``'s tiny-net-on-CPU pattern.
We test the *API shape* (CLI exit codes, JSON schema, determinism), not
the eval's downstream numerical quality.
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
SCRIPT = REPO_ROOT / "scripts" / "beam_eval_run.py"


# ---------------------------------------------------------------------------
# Tiny-net + config fixture: writes a minimal config.yaml + checkpoint pair
# the script can load. Body width / blocks / norm all match what we
# instantiate the ValueNet with.
# ---------------------------------------------------------------------------


def _tiny_config_kwargs() -> dict:
    """Minimal valid DAVIConfig kwargs — every field is required (no defaults)."""
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


def test_beam_eval_run_smoke(tiny_run, tmp_path):
    """End-to-end: script exits 0, writes JSON with the right top-level keys."""
    out_path = tmp_path / "out.json"
    result = _run_script(
        "--checkpoint",
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_run["config"]),
        "--widths",
        "4,8",
        "--max-walk-depth",
        "3",
        "--n-per-depth",
        "4",
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, f"beam_eval_run failed: stderr={result.stderr!r}"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())

    # Top-level keys.
    for key in (
        "checkpoint",
        "config_path",
        "device",
        "max_walk_depth",
        "n_per_depth",
        "seed",
        "results",
    ):
        assert key in payload, f"missing top-level key: {key}"
    assert payload["max_walk_depth"] == 3
    assert payload["n_per_depth"] == 4
    assert payload["seed"] == 0
    assert payload["device"] == "cpu"
    assert set(payload["results"].keys()) == {"4", "8"}
    # No v_star_results without --include-v-star
    assert "v_star_results" not in payload


def test_beam_eval_run_schema_keys(tiny_run, tmp_path):
    """Per-width entry has per_walk_depth (list of d/solve_rate/avg_solve_len/n
    dicts), wall_time_seconds (float > 0), states_scored (int)."""
    out_path = tmp_path / "out.json"
    result = _run_script(
        "--checkpoint",
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_run["config"]),
        "--widths",
        "4",
        "--max-walk-depth",
        "3",
        "--n-per-depth",
        "4",
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())
    entry = payload["results"]["4"]

    assert "per_walk_depth" in entry
    assert "wall_time_seconds" in entry
    assert "states_scored" in entry

    pwd = entry["per_walk_depth"]
    assert isinstance(pwd, list)
    assert len(pwd) == 3  # walk depths 1..3
    for d_idx, item in enumerate(pwd, start=1):
        assert isinstance(item, dict)
        assert item["d"] == d_idx
        assert isinstance(item["solve_rate"], float)
        assert 0.0 <= item["solve_rate"] <= 1.0
        # avg_solve_len is float-or-None
        assert item["avg_solve_len"] is None or isinstance(item["avg_solve_len"], float)
        assert item["n"] == 4

    assert isinstance(entry["wall_time_seconds"], float)
    assert entry["wall_time_seconds"] > 0
    assert isinstance(entry["states_scored"], int)
    assert entry["states_scored"] == 4 * 3  # n_per_depth * len(walk_depths)


def test_beam_eval_run_deterministic_seed(tiny_run, tmp_path):
    """Two runs with the same --seed produce bit-identical solve_rate values."""
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    common = [
        "--checkpoint",
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_run["config"]),
        "--widths",
        "4,8",
        "--max-walk-depth",
        "3",
        "--n-per-depth",
        "8",
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
    for width in ("4", "8"):
        a_pwd = pa["results"][width]["per_walk_depth"]
        b_pwd = pb["results"][width]["per_walk_depth"]
        assert len(a_pwd) == len(b_pwd)
        for ai, bi in zip(a_pwd, b_pwd, strict=True):
            assert ai["d"] == bi["d"]
            assert ai["solve_rate"] == bi["solve_rate"], (
                f"width={width} d={ai['d']} solve_rate diverged: "
                f"{ai['solve_rate']} vs {bi['solve_rate']}"
            )


def test_beam_eval_run_include_v_star(tiny_run, tmp_path):
    """With --include-v-star the JSON has v_star_results with per-width per_v_star.

    Skips if the bounded oracle cache isn't on disk (CI without data/).
    """
    oracle_path = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"
    if not oracle_path.exists():
        pytest.skip(f"oracle cache not present: {oracle_path}")

    out_path = tmp_path / "out.json"
    result = _run_script(
        "--checkpoint",
        str(tiny_run["checkpoint"]),
        "--config",
        str(tiny_run["config"]),
        "--widths",
        "4",
        "--max-walk-depth",
        "2",
        "--n-per-depth",
        "4",
        "--device",
        "cpu",
        "--include-v-star",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())

    assert "v_star_results" in payload
    v_entry = payload["v_star_results"]["4"]
    assert "per_v_star" in v_entry
    assert "wall_time_seconds" in v_entry
    pvs = v_entry["per_v_star"]
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


def test_beam_eval_run_auto_config_discovery(tiny_run, tmp_path):
    """Omitting --config: script picks up the sibling config.yaml."""
    out_path = tmp_path / "out.json"
    result = _run_script(
        "--checkpoint",
        str(tiny_run["checkpoint"]),
        "--widths",
        "4",
        "--max-walk-depth",
        "2",
        "--n-per-depth",
        "4",
        "--device",
        "cpu",
        "--output-json",
        str(out_path),
    )
    assert result.returncode == 0, f"auto-discovery failed: stderr={result.stderr!r}"
    payload = json.loads(out_path.read_text())
    # config_path should resolve to the sibling config.yaml.
    assert Path(payload["config_path"]).resolve() == tiny_run["config"].resolve()
