"""Integration tests for the ``--warm-start`` + ``--from-step`` CLI mode
of ``experiments/davi-3x3/run.py``.

Scoped narrow per the warm-start contract:
- ``_load_warm_start`` loads model state_dict, target_net = copy of model,
  optimizer NOT touched.
- ``--warm-start`` requires ``--from-step >= 1`` and is mutually exclusive
  with ``--resume``.
- The training loop's first emitted step is ``from_step + 1``.
- ``event="run_start"`` carries ``warm_start_from`` and ``from_step``
  fields when warm-starting; absent on fresh runs.

The unit-level tests use the ``_load_warm_start`` helper directly with a
tiny (64,32) ValueNet — the heavyweight integration test invokes
``run.py`` end-to-end as a subprocess on a 4-step config (no wandb), with
a checkpoint pre-dropped on disk.
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
from rubik.training.config import DAVIConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "davi-3x3"

# Make experiments/davi-3x3/run.py importable. As in test_eval_3x3.py the
# shadow over the stdlib ``eval`` builtin is intentional and isolated.
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from run import _load_warm_start, _save_checkpoint  # noqa: E402


def _tiny_net() -> ValueNet:
    """Tiny CPU-only ValueNet matching the test-config arch."""
    return ValueNet(
        CUBE_3X3,
        body_widths=(64, 32),
        n_residual_blocks=1,
        normalization="bn",
    )


# ---------------------------------------------------------------------------
# _load_warm_start unit-level
# ---------------------------------------------------------------------------


def test_load_warm_start_legacy_bare_state_dict(tmp_path):
    """Legacy format: bare state_dict at the path root."""
    src = _tiny_net()
    src.eval()
    ckpt_path = tmp_path / "legacy.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = _tiny_net()
    target = _tiny_net()
    _load_warm_start(ckpt_path, dst, target, device=torch.device("cpu"))

    # Model weights match source.
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v), (
            f"warm-start dst state mismatch on {k}"
        )
    # target_net mirrors dst (= source).
    for k, v in dst.state_dict().items():
        assert torch.equal(target.state_dict()[k], v), (
            f"target_net state mismatch on {k}"
        )


def test_load_warm_start_dict_format(tmp_path):
    """New dict format: {net_state, target_net_state, optimizer_state, step}.

    Warm-start uses ONLY ``net_state``; ``target_net_state`` and
    ``optimizer_state`` are ignored. The loaded ``target_net`` mirrors the
    loaded value_net (NOT the saved target_net_state).
    """
    src = _tiny_net()
    src.eval()
    # Different random init — stand-in for a saved target_net_state.
    other = _tiny_net()
    ckpt_path = tmp_path / "dict_fmt.pt"
    torch.save(
        {
            "net_state": src.state_dict(),
            "target_net_state": other.state_dict(),
            "optimizer_state": {"placeholder": "ignored"},
            "step": 16000,
        },
        ckpt_path,
    )

    dst = _tiny_net()
    target = _tiny_net()
    _load_warm_start(ckpt_path, dst, target, device=torch.device("cpu"))

    # dst == src.
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)
    # target == dst (NOT == other). Confirms warm-start ignores
    # target_net_state and copies from the loaded value_net instead.
    for k, v in dst.state_dict().items():
        assert torch.equal(target.state_dict()[k], v)
    # And — sanity-check — target is NOT the saved target_net_state.
    diffs = sum(
        not torch.equal(target.state_dict()[k], v)
        for k, v in other.state_dict().items()
    )
    assert diffs > 0, (
        "target_net unexpectedly identical to the saved target_net_state — "
        "warm-start should ignore that key"
    )


def test_save_checkpoint_bundle_with_provenance(tmp_path):
    """Bundle carries wall_time_seconds + wandb_run_id when supplied."""
    net = _tiny_net()
    target = _tiny_net()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "with_provenance.pt"
    _save_checkpoint(
        ckpt_path,
        net,
        target,
        optim,
        step=4242,
        wall_time_seconds=12.5,
        wandb_run_id="abc123xyz",
    )
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(raw, dict)
    # Core fields.
    assert "net_state" in raw
    assert "target_net_state" in raw
    assert "optimizer_state" in raw
    assert raw["step"] == 4242
    # New provenance fields land with the right types/values.
    assert raw["wall_time_seconds"] == pytest.approx(12.5)
    assert isinstance(raw["wall_time_seconds"], float)
    assert raw["wandb_run_id"] == "abc123xyz"
    assert isinstance(raw["wandb_run_id"], str)


def test_save_checkpoint_omits_provenance_when_unset(tmp_path):
    """When wall_time_seconds/wandb_run_id are None, the keys are absent.

    Backwards-compat invariant: legacy readers must not see ``None``
    placeholders for fields they don't know about. The keys are simply
    omitted from the saved dict.
    """
    net = _tiny_net()
    target = _tiny_net()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "minimal.pt"
    _save_checkpoint(ckpt_path, net, target, optim, step=100)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(raw, dict)
    assert "wall_time_seconds" not in raw
    assert "wandb_run_id" not in raw
    # Core fields still present.
    assert raw["step"] == 100
    assert "net_state" in raw
    assert "target_net_state" in raw
    assert "optimizer_state" in raw


def test_load_warm_start_arch_mismatch_raises(tmp_path):
    """Loading a checkpoint into a dimensionally mismatched net raises."""
    src = ValueNet(
        CUBE_3X3,
        body_widths=(128, 64),
        n_residual_blocks=1,
        normalization="bn",
    )
    ckpt_path = tmp_path / "mismatch.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = _tiny_net()  # body_widths=(64, 32) — different
    target = _tiny_net()
    with pytest.raises(RuntimeError):
        _load_warm_start(ckpt_path, dst, target, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# CLI integration via subprocess
# ---------------------------------------------------------------------------


def _write_tiny_config(path: Path, *, n_steps: int) -> None:
    """Author a tiny DAVIConfig yaml that runs end-to-end fast on CPU.

    n_steps controls how many training iterations the run loop executes.
    All other fields are baseline-tiny.
    """
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
        eval_every=0,  # disable eval to keep this fast (no oracle dep at runtime)
        checkpoint_every=0,
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


def test_warm_start_requires_from_step():
    """``--warm-start`` without ``--from-step >= 1`` errors at argparse."""
    script = EXPERIMENT_DIR / "run.py"
    # Use a config path that doesn't have to exist — argparse cross-checks
    # fire before the config is loaded. A bogus path is fine.
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "nonexistent.yaml",
            "--warm-start",
            "nonexistent.pt",
            "--no-wandb",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--from-step" in result.stderr


def test_warm_start_mutually_exclusive_with_resume():
    script = EXPERIMENT_DIR / "run.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "nonexistent.yaml",
            "--warm-start",
            "x.pt",
            "--from-step",
            "100",
            "--resume",
            "y.pt",
            "--no-wandb",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_from_step_without_warm_start_errors():
    script = EXPERIMENT_DIR / "run.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "nonexistent.yaml",
            "--from-step",
            "100",
            "--no-wandb",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--from-step is only meaningful with --warm-start" in result.stderr


def test_warm_start_end_to_end_step_counter_and_run_start(tmp_path):
    """Run.py with --warm-start <ckpt> --from-step 100 against a tiny
    config (n_steps=104, eval_every=0, no checkpoints): verify the first
    emitted step is 101, run_start carries warm_start_from + from_step,
    and run_end's step is 104.
    """
    # 1. Pre-drop a tiny checkpoint on disk in the legacy bare-state-dict
    # format.
    src = _tiny_net()
    ckpt_path = tmp_path / "warm_src.pt"
    torch.save(src.state_dict(), ckpt_path)

    # 2. Author a tiny config with n_steps=104. Combined with --from-step=100,
    # the loop runs steps 101..104 inclusive (4 iterations).
    config_path = tmp_path / "tiny.yaml"
    _write_tiny_config(config_path, n_steps=104)
    out_dir = tmp_path / "warm-run"

    # 3. Invoke run.py.
    script = EXPERIMENT_DIR / "run.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--warm-start",
            str(ckpt_path),
            "--from-step",
            "100",
            "--no-wandb",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"run.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    # 4. metrics.jsonl assertions.
    metrics = _read_jsonl(out_dir / "metrics.jsonl")
    by_event: dict[str, list[dict]] = {}
    for rec in metrics:
        by_event.setdefault(rec.get("event", ""), []).append(rec)

    # run_start has the warm-start annotation fields.
    assert len(by_event["run_start"]) == 1
    rs = by_event["run_start"][0]
    assert rs["warm_start_from"] == str(ckpt_path)
    assert rs["from_step"] == 100

    # The first event="step" record has step=101 (= from_step + 1).
    step_recs = by_event["step"]
    assert step_recs[0]["step"] == 101, (
        f"first step record should be at step 101 (from_step + 1); "
        f"got step={step_recs[0]['step']}"
    )

    # The training loop ran exactly 4 steps (101..104 inclusive). With
    # log_every=1, that yields 4 step records.
    assert len(step_recs) == 4
    assert [r["step"] for r in step_recs] == [101, 102, 103, 104]

    # run_end carries the final step.
    re = by_event["run_end"][0]
    assert re["step"] == 104
    # `early_stopped` field still emitted (compat) and is False.
    assert re["early_stopped"] is False

    # 5. config.yaml snapshot exists and round-trips.
    snap = yaml.safe_load((out_dir / "config.yaml").read_text())
    assert snap["n_steps"] == 104

    # 6. net_final.pt exists.
    assert (out_dir / "net_final.pt").exists()


def test_fresh_run_has_no_warm_start_fields(tmp_path):
    """Without ``--warm-start``, run_start does not carry warm-start fields.

    Backwards-compat invariant: historical readers parsing run_start must
    not trip on missing keys.
    """
    config_path = tmp_path / "tiny.yaml"
    _write_tiny_config(config_path, n_steps=2)
    out_dir = tmp_path / "fresh-run"

    script = EXPERIMENT_DIR / "run.py"
    result = subprocess.run(
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
    assert result.returncode == 0, (
        f"run.py failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    metrics = _read_jsonl(out_dir / "metrics.jsonl")
    rs = next(r for r in metrics if r.get("event") == "run_start")
    assert "warm_start_from" not in rs
    assert "from_step" not in rs

    # First trained step is step 1 on a fresh run.
    step_recs = [r for r in metrics if r.get("event") == "step"]
    assert step_recs[0]["step"] == 1
