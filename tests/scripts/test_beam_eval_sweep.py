"""Tests for ``scripts/beam_eval_sweep.py`` — single-checkpoint sweep wrapper.

Smoke-only: the actual eval is exercised in
``tests/scripts/test_beam_eval_model.py``. Here we verify the sweep
machinery — output filenames, output count, and that each output is in
the flat schema with the swept field set per-run.
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
SCRIPT = REPO_ROOT / "scripts" / "beam_eval_sweep.py"


def _tiny_config_kwargs() -> dict:
    return {
        "max_scramble_depth": 14,
        "max_scramble_depth_initial": 0,
        "max_scramble_depth_ramp_steps": 0,
        "batch_size": 32,
        "n_steps": 100,
        "learning_rate": 1e-3,
        "target_sync_interval": 500,
        "body_widths": [64, 32],
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
def tiny_model(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _tiny_config_kwargs()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    net = ValueNet(
        CUBE_3X3,
        body_widths=tuple(cfg["body_widths"]),
        n_residual_blocks=cfg["n_residual_blocks"],
        normalization=cfg["normalization"],
    )
    net.eval()
    ckpt = run_dir / "net_final.pt"
    torch.save({"net_state": net.state_dict()}, ckpt)
    return ckpt


@pytest.fixture
def tiny_eval_config_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "description": "tiny",
                "n_per_depth": [4, 4],
                "beam_width": 4,
                "precision": "fp32",
                "seed": 0,
            },
            sort_keys=False,
        )
    )
    return p


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_sweep_over_width_smoke(tiny_model, tiny_eval_config_yaml):
    """``--over width --values "4,8"`` writes 2 JSONs with width-suffix names."""
    result = _run_script(
        str(tiny_model),
        "--over",
        "width",
        "--values",
        "4,8",
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    parent = tiny_model.parent
    a = parent / "net_final_eval_tiny_width=4.json"
    b = parent / "net_final_eval_tiny_width=8.json"
    missing = [p for p in (a, b) if not p.exists()]
    assert not missing, f"missing: {missing}"

    # Each JSON has the swept value applied.
    pa = json.loads(a.read_text())
    pb = json.loads(b.read_text())
    assert pa["beam_width"] == 4
    assert pb["beam_width"] == 8
    # Both flat-schema.
    assert "per_walk_depth" in pa and "per_walk_depth" in pb


def test_sweep_over_precision_smoke(tiny_model, tiny_eval_config_yaml):
    """``--over precision --values "fp32"`` writes a precision-suffixed JSON."""
    # Single value to keep this fast; two-value version is exercised by width above.
    result = _run_script(
        str(tiny_model),
        "--over",
        "precision",
        "--values",
        "fp32",
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    out = tiny_model.parent / "net_final_eval_tiny_precision=fp32.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["precision"] == "fp32"


def test_sweep_over_seed_smoke(tiny_model, tiny_eval_config_yaml):
    """``--over seed --values "0,1"`` writes 2 JSONs with the swept seed."""
    result = _run_script(
        str(tiny_model),
        "--over",
        "seed",
        "--values",
        "0,1",
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    a = tiny_model.parent / "net_final_eval_tiny_seed=0.json"
    b = tiny_model.parent / "net_final_eval_tiny_seed=1.json"
    assert a.exists() and b.exists()
    pa = json.loads(a.read_text())
    pb = json.loads(b.read_text())
    assert pa["seed"] == 0
    assert pb["seed"] == 1


def test_sweep_over_max_depth_smoke(tiny_model, tiny_eval_config_yaml):
    """``--over max-depth --values "1,2"`` slices n_per_depth per-run."""
    result = _run_script(
        str(tiny_model),
        "--over",
        "max-depth",
        "--values",
        "1,2",
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    a = tiny_model.parent / "net_final_eval_tiny_max-depth=1.json"
    b = tiny_model.parent / "net_final_eval_tiny_max-depth=2.json"
    assert a.exists() and b.exists()
    pa = json.loads(a.read_text())
    pb = json.loads(b.read_text())
    assert pa["max_depth"] == 1
    assert pb["max_depth"] == 2
    assert len(pa["per_walk_depth"]) == 1
    assert len(pb["per_walk_depth"]) == 2


def test_sweep_render_html_chains_renderer(tiny_model, tiny_eval_config_yaml):
    """``--render-html`` produces a single overlay HTML next to the JSONs."""
    result = _run_script(
        str(tiny_model),
        "--over",
        "width",
        "--values",
        "4,8",
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--render-html",
    )
    assert result.returncode == 0, result.stderr
    html_path = tiny_model.parent / "net_final_eval_tiny_width_sweep.html"
    assert html_path.exists()
    html = html_path.read_text()
    # Both width values appear in the overlay legend.
    assert "width=4" in html
    assert "width=8" in html
