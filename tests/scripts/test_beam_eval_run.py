"""Tests for ``scripts/beam_eval_run.py`` — run-dir wrapper.

The script walks ``<run-dir>/`` for ``net_step_*.pt`` + ``net_final.pt``,
runs the same eval logic as ``beam_eval_model.py`` per checkpoint, and
writes a single consolidated JSONL rollup at
``<run-dir>/results/beam_eval_<config>.jsonl`` (one flat-schema payload
per line, sorted ascending by step). Re-runs merge — new entries replace
existing ones at the same step, prior steps preserved.

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


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file as a list of dicts (one per non-empty line)."""
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


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
# Consolidated JSONL output schema
# ---------------------------------------------------------------------------


def test_jsonl_rollup_written(tiny_run_dir, tiny_eval_config_yaml):
    """Default: all checkpoints land in one beam_eval_<config>.jsonl."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    assert rollup.exists(), f"expected rollup {rollup}"
    records = _read_jsonl(rollup)
    # 3 net_step checkpoints + final → 4 lines.
    assert len(records) == 4
    steps = [r["step"] for r in records]
    # Sorted ascending; final-step-from-metrics is 40.
    assert steps == [10, 20, 30, 40]
    # No per-checkpoint JSONs left behind.
    legacy = sorted((tiny_run_dir / "results").glob("step_*_eval_tiny.json"))
    assert legacy == []


def test_jsonl_records_are_flat_schema(tiny_run_dir, tiny_eval_config_yaml):
    """Each JSONL record has the flat schema + a step field."""
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
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    records = _read_jsonl(rollup)
    assert len(records) == 1
    payload = records[0]
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


def test_rerun_merges_and_dedups_by_step(tiny_run_dir, tiny_eval_config_yaml):
    """Two evals against the same JSONL: union of steps; later run wins overlaps.

    Run 1 with ``--steps "10,20"`` → records at {10, 20}. Run 2 with
    ``--steps "20,30"`` → records at {10, 20, 30}, with the run-2 payload
    at step 20 (the new one replaces the old).
    """
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"

    # Run 1: steps {10, 20}.
    r1 = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--steps",
        "10,20",
    )
    assert r1.returncode == 0, r1.stderr
    records_1 = _read_jsonl(rollup)
    assert sorted(r["step"] for r in records_1) == [10, 20]
    step20_run1 = next(r for r in records_1 if r["step"] == 20)

    # Run 2: steps {20, 30} — should merge with run 1's step-10 record
    # and overwrite step-20.
    r2 = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--steps",
        "20,30",
        # Different seed → different solve_rate values, so we can detect
        # which run's payload survived for step 20.
        "--seed",
        "12345",
    )
    assert r2.returncode == 0, r2.stderr
    records_2 = _read_jsonl(rollup)
    steps = sorted(r["step"] for r in records_2)
    assert steps == [10, 20, 30]
    # Run-2 step-20 payload should replace run-1's. Seed shows up
    # explicitly in the payload as a witness for "which run wrote this".
    step20_run2 = next(r for r in records_2 if r["step"] == 20)
    assert step20_run2.get("seed") == 12345
    assert step20_run1.get("seed") != 12345
    # Run-1's step-10 record persisted unchanged.
    step10 = next(r for r in records_2 if r["step"] == 10)
    assert step10.get("seed") == step20_run1.get("seed")


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
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    records = _read_jsonl(rollup)
    steps = sorted(r["step"] for r in records)
    # step_20 (multiple of 20) + step_40 (net_final, always kept).
    assert steps == [20, 40]


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
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    records = _read_jsonl(rollup)
    steps = sorted(r["step"] for r in records)
    assert steps == [10, 40]


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


def test_dict_step_overrides_filename_step(tmp_path, tiny_eval_config_yaml):
    """A bundle whose dict step disagrees with the filename → dict wins.

    Simulates post-rename drift: filename says ``net_step_5000.pt`` but
    the bundle's ``step`` field says 5050. The JSONL record's ``step``
    field reflects the dict value (the truth), not the filename.
    """
    run_dir = tmp_path / "drift-run"
    run_dir.mkdir()
    cfg = _tiny_config_kwargs()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    net = _make_tiny_net()
    target = _make_tiny_net()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    bundle = {
        "net_state": net.state_dict(),
        "target_net_state": target.state_dict(),
        "optimizer_state": optim.state_dict(),
        "step": 5050,  # mismatches the filename below
    }
    # Filename says 5000, dict says 5050.
    torch.save(bundle, run_dir / "net_step_5000.pt")

    result = _run_script(
        str(run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    rollup = run_dir / "results" / "beam_eval_tiny.jsonl"
    records = _read_jsonl(rollup)
    assert len(records) == 1
    payload = records[0]
    assert payload["step"] == 5050
    assert payload["checkpoint_step"] == 5050


def test_legacy_bare_state_dict_falls_back_to_filename(tmp_path, tiny_eval_config_yaml):
    """Legacy bare state_dict (no ``net_state`` key) → filename step is used.

    The bundle has no provenance metadata, so ``checkpoint_step`` is None
    and the run-wrapper falls back to the filename-extracted step.
    """
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    cfg = _tiny_config_kwargs()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    net = _make_tiny_net()
    # Save a bare state_dict (no top-level dict wrapper) — the legacy format.
    torch.save(net.state_dict(), run_dir / "net_step_1234.pt")

    result = _run_script(
        str(run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    rollup = run_dir / "results" / "beam_eval_tiny.jsonl"
    records = _read_jsonl(rollup)
    assert len(records) == 1
    payload = records[0]
    assert payload["step"] == 1234
    assert payload["checkpoint_step"] is None


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
    # And the new rollup IS written.
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    assert rollup.exists()
