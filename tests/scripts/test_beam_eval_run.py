"""Tests for ``scripts/beam_eval_run.py`` — run-dir wrapper.

Renamed in the beam-solve-perf refactor (was ``post_run_beam_eval.py``).
The script walks ``<run-dir>/`` for ``net_step_*.pt`` + ``net_final.pt``,
runs the same eval logic as ``beam_eval_model.py`` per checkpoint, and
writes one flat-schema JSON per checkpoint at
``<run-dir>/results/step_<N>_eval_<config>.json``.

Tests use a tiny CPU ValueNet so they run in seconds.
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


def _make_tiny_net() -> ValueNet:
    cfg = _tiny_config_kwargs()
    net = ValueNet(
        CUBE_3X3,
        body_widths=tuple(cfg["body_widths"]),
        n_residual_blocks=cfg["n_residual_blocks"],
        normalization=cfg["normalization"],
    )
    net.eval()
    return net


@pytest.fixture
def tiny_run_dir(tmp_path: Path) -> Path:
    """Build a tiny run dir with config.yaml + multiple checkpoints."""
    run_dir = tmp_path / "tiny-run"
    run_dir.mkdir()
    cfg = _tiny_config_kwargs()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    net = _make_tiny_net()
    state = {"net_state": net.state_dict()}
    # Three step checkpoints + a final.
    for step in (10, 20, 30):
        torch.save(state, run_dir / f"net_step_{step}.pt")
    torch.save(state, run_dir / "net_final.pt")

    # metrics.jsonl with a final step beyond the last net_step.
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"step": 10, "event": "value_train"}),
                json.dumps({"step": 20, "event": "value_train"}),
                json.dumps({"step": 30, "event": "value_train"}),
                json.dumps({"step": 40, "event": "value_train"}),
            ]
        )
    )
    return run_dir


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


# ---------------------------------------------------------------------------
# Empty-dir behavior — preserved across the rename
# ---------------------------------------------------------------------------


def test_empty_dir_writes_empty_json(tmp_path):
    """Empty run dir → exits 0, writes empty JSON at the legacy path."""
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    result = _run_script(str(run_dir))
    assert result.returncode == 0, result.stderr
    out = run_dir / "results" / "post_run_beam_eval.json"
    assert out.exists()
    assert json.loads(out.read_text()) == {}


# ---------------------------------------------------------------------------
# Per-checkpoint output schema
# ---------------------------------------------------------------------------


def test_per_checkpoint_jsons_written(tiny_run_dir, tiny_eval_config_yaml):
    """Default: every checkpoint gets a step_<N>_eval_<config>.json."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    results_dir = tiny_run_dir / "results"
    written = sorted(results_dir.glob("step_*_eval_tiny.json"))
    # 3 net_step checkpoints + final → 4 outputs.
    assert len(written) == 4
    # Final step is 40 (from metrics.jsonl).
    names = {p.name for p in written}
    assert "step_10_eval_tiny.json" in names
    assert "step_20_eval_tiny.json" in names
    assert "step_30_eval_tiny.json" in names
    assert "step_40_eval_tiny.json" in names


def test_per_checkpoint_json_is_flat_schema(tiny_run_dir, tiny_eval_config_yaml):
    """Each per-checkpoint JSON has the flat schema + a step field."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--steps",
        "20",
    )
    assert result.returncode == 0, result.stderr
    out = tiny_run_dir / "results" / "step_20_eval_tiny.json"
    payload = json.loads(out.read_text())
    # Flat schema fields.
    for key in (
        "model",
        "config_name",
        "per_walk_depth",
        "beam_width",
        "n_per_depth",
        "wall_time_seconds",
        "states_scored",
        "step",
    ):
        assert key in payload, f"missing {key}"
    assert payload["step"] == 20
    assert payload["beam_width"] == 4
    assert payload["max_depth"] == 2
    assert len(payload["per_walk_depth"]) == 2


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_every_steps_filter(tiny_run_dir, tiny_eval_config_yaml):
    """``--every-steps 20`` keeps step_20 + net_final (but skips step_10, step_30)."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--every-steps",
        "20",
    )
    assert result.returncode == 0, result.stderr
    written = sorted((tiny_run_dir / "results").glob("step_*_eval_tiny.json"))
    names = {p.name for p in written}
    # step_20 (multiple of 20) + step_40 (net_final, always kept).
    assert "step_20_eval_tiny.json" in names
    assert "step_40_eval_tiny.json" in names
    assert "step_10_eval_tiny.json" not in names
    assert "step_30_eval_tiny.json" not in names


def test_steps_csv_filter(tiny_run_dir, tiny_eval_config_yaml):
    """``--steps "10,final"`` keeps step_10 + net_final.pt only."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--steps",
        "10,final",
    )
    assert result.returncode == 0, result.stderr
    names = {p.name for p in (tiny_run_dir / "results").glob("step_*_eval_tiny.json")}
    assert names == {"step_10_eval_tiny.json", "step_40_eval_tiny.json"}


def test_every_steps_and_steps_mutually_exclusive(tiny_run_dir, tiny_eval_config_yaml):
    """argparse rejects passing both filters."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--every-steps",
        "10",
        "--steps",
        "10",
    )
    assert result.returncode != 0
    assert "not allowed with" in (result.stderr + result.stdout)


def test_steps_csv_unknown_step_errors(tiny_run_dir, tiny_eval_config_yaml):
    """Requesting a step not present in the run dir → error."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--steps",
        "999999",
    )
    assert result.returncode != 0
    assert "999999" in (result.stderr + result.stdout)


def test_no_post_run_beam_eval_json_when_checkpoints_present(
    tiny_run_dir, tiny_eval_config_yaml
):
    """When checkpoints exist, the legacy post_run_beam_eval.json is NOT written."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--steps",
        "10",
    )
    assert result.returncode == 0, result.stderr
    legacy = tiny_run_dir / "results" / "post_run_beam_eval.json"
    assert not legacy.exists(), "legacy trajectory file should not be written"
