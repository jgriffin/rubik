"""metrics.jsonl provenance fields written by ``experiments/davi-3x3/run.py``.

Two fields covered here:
- ``run_start.wandb_run_id``: always present, ``None`` when ``--no-wandb``.
- ``checkpoint.wall_time_seconds``: float, monotonically reflecting
  wall-clock since run start.

Each test runs ``run.py`` as a subprocess against a tiny CPU config — no
mocks. A 4-step run with checkpoint_every=2 emits two ``checkpoint``
records, which is plenty to verify the field's type + shape.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rubik.training.config import DAVIConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "davi-3x3"


def _write_tiny_config(path: Path, *, n_steps: int, checkpoint_every: int) -> None:
    cfg = DAVIConfig(
        max_scramble_depth=2,
        max_scramble_depth_initial=0,
        max_scramble_depth_ramp_steps=0,
        batch_size=8,
        n_steps=n_steps,
        learning_rate=0.001,
        target_sync_interval=2,
        body_widths=(64, 32),
        n_residual_blocks=1,
        normalization="bn",
        log_every=1,
        eval_every=0,
        checkpoint_every=checkpoint_every,
        seed=0,
        device="cpu",
        early_stop_enabled=False,
        early_stop_metric="macro_v_star_mae",
        early_stop_patience_evals=0,
        early_stop_min_evals=0,
        early_stop_min_delta=0.0,
    )
    cfg.to_yaml(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run(out_dir: Path, config_path: Path) -> subprocess.CompletedProcess:
    script = EXPERIMENT_DIR / "run.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--no-wandb",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_run_start_carries_wandb_run_id_none_when_no_wandb(tmp_path):
    """``--no-wandb`` runs still emit ``wandb_run_id`` on ``run_start``,
    set to ``None`` for schema stability across wandb-on/off runs.
    """
    config_path = tmp_path / "tiny.yaml"
    _write_tiny_config(config_path, n_steps=2, checkpoint_every=0)
    out_dir = tmp_path / "run"

    result = _run(out_dir, config_path)
    assert result.returncode == 0, (
        f"run.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    metrics = _read_jsonl(out_dir / "metrics.jsonl")
    rs = next(r for r in metrics if r.get("event") == "run_start")
    # Key must be present (schema stability) and value must be None
    # (--no-wandb path).
    assert "wandb_run_id" in rs, (
        "run_start should always carry wandb_run_id, even when --no-wandb"
    )
    assert rs["wandb_run_id"] is None


def test_checkpoint_event_carries_wall_time_seconds(tmp_path):
    """Every periodic ``event="checkpoint"`` record carries a positive
    float ``wall_time_seconds``, monotonic across the run.
    """
    config_path = tmp_path / "tiny.yaml"
    # n_steps=4 with checkpoint_every=2 → checkpoints at step 2 and 4.
    _write_tiny_config(config_path, n_steps=4, checkpoint_every=2)
    out_dir = tmp_path / "run"

    result = _run(out_dir, config_path)
    assert result.returncode == 0, (
        f"run.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    metrics = _read_jsonl(out_dir / "metrics.jsonl")
    ckpts = [r for r in metrics if r.get("event") == "checkpoint"]
    assert len(ckpts) == 2, (
        f"expected 2 checkpoint events at step 2 and 4; got {len(ckpts)}"
    )
    assert [r["step"] for r in ckpts] == [2, 4]

    for r in ckpts:
        assert "wall_time_seconds" in r, (
            "checkpoint event should carry wall_time_seconds"
        )
        wall = r["wall_time_seconds"]
        assert isinstance(wall, float), (
            f"wall_time_seconds must be a float; got {type(wall).__name__}"
        )
        assert wall > 0.0, f"wall_time_seconds should be positive; got {wall}"

    # Monotonic: later checkpoint has a strictly larger wall time.
    assert ckpts[1]["wall_time_seconds"] > ckpts[0]["wall_time_seconds"], (
        "wall_time_seconds should increase between successive checkpoints"
    )


def test_run_start_wandb_run_id_present_on_fresh_run(tmp_path):
    """Sanity-check the schema invariant alongside the existing fresh-run
    backwards-compat assertions: ``wandb_run_id`` lands on ``run_start``
    regardless of warm-start.
    """
    config_path = tmp_path / "tiny.yaml"
    _write_tiny_config(config_path, n_steps=2, checkpoint_every=0)
    out_dir = tmp_path / "run"

    result = _run(out_dir, config_path)
    assert result.returncode == 0, (
        f"run.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    metrics = _read_jsonl(out_dir / "metrics.jsonl")
    rs = next(r for r in metrics if r.get("event") == "run_start")
    assert "wandb_run_id" in rs
    # And the existing fresh-run invariant still holds.
    assert "warm_start_from" not in rs
    assert "from_step" not in rs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
