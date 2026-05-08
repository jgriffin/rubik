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
import re
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
    # max_scramble_depth=14 was the legacy default. Under the new
    # min(yaml, training) semantics, auto-derive caps the eval depth at
    # the YAML schedule's length, so the tiny eval config's length-2
    # schedule still produces 2 walk depths regardless of training depth.
    # Tests that need to exercise auto-derive at a different effective
    # depth override this kwarg.
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


def test_rerun_with_force_backs_up_and_starts_fresh(
    tiny_run_dir, tiny_eval_config_yaml
):
    """``--force`` always backs up the existing JSONL and starts from empty.

    Run 1 with ``--steps "10,20"`` → records at {10, 20} in the primary file.
    Run 2 with ``--steps "20,30" --force`` → primary contains ONLY {20, 30}
    (the new run's records). The pre-existing JSONL is preserved untouched
    in a ``.bak.<UTC-timestamp>`` sibling so prior data is never lost.
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
    pre_text = rollup.read_text()

    # Run 2: steps {20, 30} with --force — backs up run 1's file and
    # starts with an empty merge state, so the primary post-run reflects
    # ONLY the run-2 evaluations.
    r2 = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--steps",
        "20,30",
        "--force",
        # Different seed → different solve_rate values, so we can detect
        # which run's payload survived for step 20.
        "--seed",
        "12345",
    )
    assert r2.returncode == 0, r2.stderr
    records_2 = _read_jsonl(rollup)
    steps = sorted(r["step"] for r in records_2)
    assert steps == [20, 30]
    # Run-2 step-20 payload carries the run-2 seed.
    step20_run2 = next(r for r in records_2 if r["step"] == 20)
    assert step20_run2.get("seed") == 12345
    assert step20_run1.get("seed") != 12345

    # Run 1's full JSONL — including the step-10 record — survives in
    # the timestamped backup.
    backups = sorted(rollup.parent.glob("beam_eval_tiny.jsonl.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == pre_text
    backup_records = _read_jsonl(backups[0])
    assert sorted(r["step"] for r in backup_records) == [10, 20]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_stride_filter(tiny_run_dir, tiny_eval_config_yaml):
    """``--stride 20`` keeps step_20 + net_final (but skips step_10, step_30)."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--stride",
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


def test_stride_and_steps_mutually_exclusive(tiny_run_dir, tiny_eval_config_yaml):
    """argparse rejects passing both filters."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--stride",
        "10",
        "--steps",
        "10",
    )
    assert result.returncode != 0
    assert "not allowed with" in (result.stderr + result.stdout)


def test_ensure_skips_existing_step(tiny_run_dir, tiny_eval_config_yaml):
    """Pre-populated JSONL → only missing checkpoints are evaluated."""
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    rollup.parent.mkdir(parents=True, exist_ok=True)
    # Seed step=20 with a sentinel record. Ensure-by-default should
    # preserve it untouched.
    sentinel = {"step": 20, "sentinel": "preexisting"}
    rollup.write_text(json.dumps(sentinel) + "\n")

    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    # The skip line names the existing step.
    assert "skipping step=20" in log

    records = _read_jsonl(rollup)
    steps = sorted(r["step"] for r in records)
    # step_10, step_30, net_final (step=40) get evaluated; step=20 is
    # preserved as the sentinel record.
    assert steps == [10, 20, 30, 40]
    step20 = next(r for r in records if r["step"] == 20)
    assert step20.get("sentinel") == "preexisting"
    # Newly-evaluated records carry the full payload schema.
    step10 = next(r for r in records if r["step"] == 10)
    assert "per_walk_depth" in step10


def test_force_recomputes_existing(tiny_run_dir, tiny_eval_config_yaml):
    """``--force`` recomputes every selected checkpoint, replacing existing rows."""
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    rollup.parent.mkdir(parents=True, exist_ok=True)
    sentinel = {"step": 20, "sentinel": "preexisting"}
    rollup.write_text(json.dumps(sentinel) + "\n")

    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--force",
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    # No skip lines under --force.
    assert "skipping step=" not in log

    records = _read_jsonl(rollup)
    steps = sorted(r["step"] for r in records)
    assert steps == [10, 20, 30, 40]
    step20 = next(r for r in records if r["step"] == 20)
    # The sentinel was overwritten with a real eval payload.
    assert "sentinel" not in step20
    assert "per_walk_depth" in step20


def test_traversal_order_largest_first(tiny_run_dir, tiny_eval_config_yaml):
    """``evaluating ...`` log lines appear in descending step order."""
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr

    # Pull out the order of "evaluating <name>" lines.
    eval_lines = [line for line in log.splitlines() if line.startswith("evaluating ")]
    # Expect 4 evaluations: net_final.pt, then step_30, step_20, step_10.
    assert len(eval_lines) == 4
    # net_final.pt should be first (largest step = 40).
    assert "net_final.pt" in eval_lines[0]
    # The remaining order is descending net_step_*.pt by step.
    assert "net_step_30.pt" in eval_lines[1]
    assert "net_step_20.pt" in eval_lines[2]
    assert "net_step_10.pt" in eval_lines[3]


def test_jsonl_flushed_per_checkpoint(tiny_run_dir, tiny_eval_config_yaml):
    """Each checkpoint completion writes the rollup so partial runs persist.

    Critical for the largest-first workflow: if the user kills mid-run,
    the most-trained checkpoints (evaluated first) must already be on
    disk. Verified by the per-checkpoint ``... record(s) flushed`` log
    line emitted after each successful eval.
    """
    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr

    # Each evaluated checkpoint should print a "N record(s) flushed" line
    # with N strictly increasing (1, 2, 3, 4) — proves per-checkpoint
    # write rather than buffer-and-flush-at-end.
    flush_counts: list[int] = []
    for line in log.splitlines():
        match = re.search(r"(\d+) record\(s\) flushed", line)
        if match:
            flush_counts.append(int(match.group(1)))
    assert flush_counts == [1, 2, 3, 4], (
        f"expected per-checkpoint flushes [1,2,3,4]; got {flush_counts}"
    )


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


# ---------------------------------------------------------------------------
# Auto-derive eval max_depth = min(yaml_max_depth, training_max_depth)
# ---------------------------------------------------------------------------


def _write_eval_yaml(path: Path, *, length: int) -> None:
    """Write a minimal eval YAML with ``n_per_depth`` of the given length."""
    path.write_text(
        yaml.safe_dump(
            {
                "description": "tiny",
                "n_per_depth": [4] * length,
                "beam_width": 4,
                "precision": "fp32",
                "seed": 0,
            },
            sort_keys=False,
        )
    )


def _write_run_dir(
    run_dir: Path,
    *,
    training_max_depth: int,
    metrics_max_depth: int | None = None,
) -> None:
    """Build a tiny run dir with a single net_step_10 checkpoint.

    ``metrics_max_depth`` controls whether (and what) ``run_start`` event
    is written. ``None`` means no metrics.jsonl at all (config.yaml is
    the only training-depth source).
    """
    run_dir.mkdir()
    cfg = _tiny_config_kwargs()
    cfg["max_scramble_depth"] = training_max_depth
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    net = _make_tiny_net()
    torch.save({"net_state": net.state_dict()}, run_dir / "net_step_10.pt")

    if metrics_max_depth is not None:
        run_start = {
            "event": "run_start",
            "max_scramble_depth": int(metrics_max_depth),
        }
        (run_dir / "metrics.jsonl").write_text(json.dumps(run_start) + "\n")


def test_auto_derive_min_yaml_lt_training(tmp_path):
    """yaml_len=2 < training=14 → effective=2 (yaml caps)."""
    run_dir = tmp_path / "yaml-lt-training"
    _write_run_dir(run_dir, training_max_depth=14, metrics_max_depth=14)
    eval_cfg_path = tmp_path / "tiny.yaml"
    _write_eval_yaml(eval_cfg_path, length=2)

    result = _run_script(
        str(run_dir), "--config", str(eval_cfg_path), "--device", "cpu"
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    assert "auto-derived max_depth=2" in log
    assert "yaml=2" in log and "training=14" in log
    assert "source=metrics.jsonl" in log

    rollup = run_dir / "results" / "beam_eval_tiny.jsonl"
    payload = _read_jsonl(rollup)[0]
    assert payload["max_depth"] == 2
    assert len(payload["per_walk_depth"]) == 2


def test_auto_derive_min_training_lt_yaml(tmp_path):
    """training=2 < yaml_len=4 → effective=2 (training caps)."""
    run_dir = tmp_path / "training-lt-yaml"
    _write_run_dir(run_dir, training_max_depth=2, metrics_max_depth=2)
    eval_cfg_path = tmp_path / "tiny.yaml"
    _write_eval_yaml(eval_cfg_path, length=4)

    result = _run_script(
        str(run_dir), "--config", str(eval_cfg_path), "--device", "cpu"
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    assert "auto-derived max_depth=2" in log
    assert "yaml=4" in log and "training=2" in log

    rollup = run_dir / "results" / "beam_eval_tiny.jsonl"
    payload = _read_jsonl(rollup)[0]
    assert payload["max_depth"] == 2
    assert len(payload["per_walk_depth"]) == 2


def test_auto_derive_min_yaml_eq_training(tmp_path):
    """yaml_len == training → effective equals both."""
    run_dir = tmp_path / "yaml-eq-training"
    _write_run_dir(run_dir, training_max_depth=3, metrics_max_depth=3)
    eval_cfg_path = tmp_path / "tiny.yaml"
    _write_eval_yaml(eval_cfg_path, length=3)

    result = _run_script(
        str(run_dir), "--config", str(eval_cfg_path), "--device", "cpu"
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    assert "auto-derived max_depth=3" in log
    rollup = run_dir / "results" / "beam_eval_tiny.jsonl"
    payload = _read_jsonl(rollup)[0]
    assert payload["max_depth"] == 3
    assert len(payload["per_walk_depth"]) == 3


def test_explicit_max_depth_wins_no_min_applied(tmp_path):
    """``--max-depth N`` bypasses the min() — user is asserting they know."""
    run_dir = tmp_path / "explicit-override"
    _write_run_dir(run_dir, training_max_depth=4, metrics_max_depth=4)
    eval_cfg_path = tmp_path / "tiny.yaml"
    _write_eval_yaml(eval_cfg_path, length=4)

    # --max-depth=2 < both yaml(4) and training(4): respected verbatim.
    result = _run_script(
        str(run_dir),
        "--config",
        str(eval_cfg_path),
        "--device",
        "cpu",
        "--max-depth",
        "2",
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    assert "--max-depth=2 override" in log
    rollup = run_dir / "results" / "beam_eval_tiny.jsonl"
    payload = _read_jsonl(rollup)[0]
    assert payload["max_depth"] == 2
    assert len(payload["per_walk_depth"]) == 2


def test_explicit_max_depth_past_yaml_errors(tmp_path):
    """``--max-depth N`` > yaml length errors — no schedule for those depths."""
    run_dir = tmp_path / "explicit-past-yaml"
    _write_run_dir(run_dir, training_max_depth=14, metrics_max_depth=14)
    eval_cfg_path = tmp_path / "tiny.yaml"
    _write_eval_yaml(eval_cfg_path, length=2)

    result = _run_script(
        str(run_dir),
        "--config",
        str(eval_cfg_path),
        "--device",
        "cpu",
        "--max-depth",
        "10",
    )
    assert result.returncode != 0
    assert "exceeds n_per_depth schedule" in (result.stdout + result.stderr)


def test_auto_derive_prefers_metrics_over_config(tmp_path):
    """When metrics.jsonl run_start and config.yaml disagree, metrics wins."""
    run_dir = tmp_path / "source-pref"
    # config.yaml says 4, metrics.jsonl says 2 → effective should reflect 2.
    _write_run_dir(run_dir, training_max_depth=4, metrics_max_depth=2)
    eval_cfg_path = tmp_path / "tiny.yaml"
    _write_eval_yaml(eval_cfg_path, length=10)

    result = _run_script(
        str(run_dir), "--config", str(eval_cfg_path), "--device", "cpu"
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    # metrics.jsonl's value=2 is what's used (not config.yaml's 4).
    assert "auto-derived max_depth=2" in log
    assert "training=2" in log
    assert "source=metrics.jsonl" in log
    rollup = run_dir / "results" / "beam_eval_tiny.jsonl"
    payload = _read_jsonl(rollup)[0]
    assert payload["max_depth"] == 2


def test_auto_derive_falls_back_to_config_yaml_when_metrics_absent(tmp_path):
    """No metrics.jsonl → config.yaml is the training-depth source."""
    run_dir = tmp_path / "no-metrics"
    _write_run_dir(run_dir, training_max_depth=2, metrics_max_depth=None)
    eval_cfg_path = tmp_path / "tiny.yaml"
    _write_eval_yaml(eval_cfg_path, length=10)

    result = _run_script(
        str(run_dir), "--config", str(eval_cfg_path), "--device", "cpu"
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    assert "auto-derived max_depth=2" in log
    assert "source=config.yaml" in log


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


# ---------------------------------------------------------------------------
# Schema / signature check on existing JSONL
# ---------------------------------------------------------------------------


def _synthetic_record(
    *,
    step: int,
    max_depth: int,
    n_per_depth: list[int],
    beam_width: int,
    precision: str,
    seed: int,
) -> dict:
    """Build a JSONL record carrying the signature fields used by the schema check.

    Mirrors the production payload's top-level fields without running the
    real eval — the runner's signature check only inspects the five fields
    enumerated in ``_SIGNATURE_FIELDS``, so other keys are unnecessary.
    """
    return {
        "step": step,
        "max_depth": max_depth,
        "n_per_depth": n_per_depth,
        "beam_width": beam_width,
        "precision": precision,
        "seed": seed,
        "synthetic": True,
    }


def test_schema_mismatch_errors_without_force(tiny_run_dir, tiny_eval_config_yaml):
    """Existing JSONL with a different ``n_per_depth`` than the current invocation
    raises (non-zero exit + actionable error message). The tiny eval config
    produces ``n_per_depth=(4, 4)``; the synthetic uses ``[1, 1]``.
    """
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    rollup.parent.mkdir(parents=True, exist_ok=True)
    pre = _synthetic_record(
        step=20,
        max_depth=2,
        n_per_depth=[1, 1],  # mismatches tiny config's [4, 4]
        beam_width=4,
        precision="fp32",
        seed=0,
    )
    rollup.write_text(json.dumps(pre) + "\n")

    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode != 0
    err = result.stderr + result.stdout
    assert "different config" in err
    # The actionable hints from the error message surface in stderr.
    assert "--force" in err

    # The pre-existing JSONL was NOT modified — error happens before any
    # write or backup.
    surviving = _read_jsonl(rollup)
    assert surviving == [pre]
    backups = list(rollup.parent.glob("beam_eval_tiny.jsonl.bak.*"))
    assert backups == []


def test_schema_mismatch_with_force_backs_up(tiny_run_dir, tiny_eval_config_yaml):
    """``--force`` salvages the mismatched JSONL into a timestamped backup,
    then evaluates fresh. Backup contents == original synthetic content;
    primary file contains records consistent with the current schema.
    """
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    rollup.parent.mkdir(parents=True, exist_ok=True)
    pre = _synthetic_record(
        step=20,
        max_depth=2,
        n_per_depth=[1, 1],
        beam_width=4,
        precision="fp32",
        seed=0,
    )
    pre_text = json.dumps(pre) + "\n"
    rollup.write_text(pre_text)

    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--force",
        "--steps",
        "10",
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    assert "backed up existing JSONL" in log

    # Backup exists and matches the original content byte-for-byte.
    backups = sorted(rollup.parent.glob("beam_eval_tiny.jsonl.bak.*"))
    assert len(backups) == 1, f"expected one backup, got {backups}"
    backup = backups[0]
    # Naming convention: YYYYMMDDTHHMMSSZ suffix.
    assert re.match(r"^beam_eval_tiny\.jsonl\.bak\.\d{8}T\d{6}Z$", backup.name), (
        f"unexpected backup name {backup.name}"
    )
    assert backup.read_text() == pre_text

    # Primary file exists with new (real-schema) content. The pre-existing
    # synthetic step-20 record is gone; only step=10 was evaluated.
    records = _read_jsonl(rollup)
    steps = sorted(r["step"] for r in records)
    assert steps == [10]
    new_record = records[0]
    assert "synthetic" not in new_record
    # Signature fields reflect the current invocation, not the pre-run.
    assert new_record["n_per_depth"] == [4, 4]
    assert new_record["max_depth"] == 2
    assert new_record["precision"] == "fp32"
    assert new_record["seed"] == 0


def test_force_backs_up_even_when_matching(tiny_run_dir, tiny_eval_config_yaml):
    """``--force`` always backs up — even when existing records already match
    the current schema. Verified by running the script once to populate a
    real rollup, then running again with ``--force`` and confirming both
    a backup and a fresh primary file exist.
    """
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"

    # Run 1: produce a real rollup with the current schema.
    r1 = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--steps",
        "10",
    )
    assert r1.returncode == 0, r1.stderr
    pre_records = _read_jsonl(rollup)
    pre_text = rollup.read_text()
    assert len(pre_records) == 1

    # Run 2: --force on a matching-schema file. Backup should still happen.
    r2 = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
        "--force",
        "--steps",
        "20",
    )
    assert r2.returncode == 0, r2.stderr
    log = r2.stdout + r2.stderr
    assert "backed up existing JSONL" in log

    backups = sorted(rollup.parent.glob("beam_eval_tiny.jsonl.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == pre_text

    # Primary file post-run reflects only the run-2 evaluation (step 20).
    # Run 1's step-10 record was moved to the backup, not retained.
    post_records = _read_jsonl(rollup)
    steps = sorted(r["step"] for r in post_records)
    assert steps == [20]
    # And the new record carries the current signature.
    assert post_records[0]["n_per_depth"] == [4, 4]
    assert post_records[0]["beam_width"] == 4


def test_schema_match_no_error(tiny_run_dir, tiny_eval_config_yaml):
    """When existing records' signature matches the current invocation,
    ensure-skip behavior is preserved — existing steps stay, missing
    steps get evaluated, no error or backup.
    """
    rollup = tiny_run_dir / "results" / "beam_eval_tiny.jsonl"
    rollup.parent.mkdir(parents=True, exist_ok=True)
    matching = _synthetic_record(
        step=20,
        max_depth=2,
        n_per_depth=[4, 4],  # matches tiny eval config
        beam_width=4,
        precision="fp32",
        seed=0,
    )
    rollup.write_text(json.dumps(matching) + "\n")

    result = _run_script(
        str(tiny_run_dir),
        "--config",
        str(tiny_eval_config_yaml),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    log = result.stdout + result.stderr
    # Matching schema → no backup, no error.
    assert "backed up existing JSONL" not in log
    assert "different config" not in log
    # Existing step=20 was preserved; the others were evaluated.
    assert "skipping step=20" in log

    records = _read_jsonl(rollup)
    by_step = {r["step"]: r for r in records}
    assert sorted(by_step) == [10, 20, 30, 40]
    # The step-20 record retains the synthetic marker — preserved as-is.
    assert by_step[20].get("synthetic") is True
    # Newly-evaluated steps have the full payload schema.
    assert "per_walk_depth" in by_step[10]
    # No backup file was created.
    backups = list(rollup.parent.glob("beam_eval_tiny.jsonl.bak.*"))
    assert backups == []
