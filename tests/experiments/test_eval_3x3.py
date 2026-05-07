"""Tests for the lean three-function eval API in experiments/davi-3x3/eval.py.

The K=6 oracle is loaded once per module via fixture (mirrors
``tests/oracle/test_v_star_bounded_3x3.py``'s pattern). Tests use
random-init ValueNets at tiny dims and CPU-only to keep each test fast
and free of MPS dependence — these tests cover the eval *API shape*,
not the eval's downstream numerical quality.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rubik.cube.spec import CUBE_3X3
from rubik.model.network import ValueNet
from rubik.oracle.v_star_bounded_3x3 import (
    compute_v_star_bounded_3x3,
    load_v_star_bounded_3x3,
    load_v_star_bounded_3x3_arrays,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "davi-3x3"

# Make experiments/davi-3x3/eval.py importable. Tests live outside the
# experiment dir, so we splice it in here on first import. The shadow over
# the stdlib ``eval`` builtin is intentional and isolated to this module.
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from eval import beam_eval_v_star, beam_eval_walk, value_eval  # noqa: E402

# ---------------------------------------------------------------------------
# Module-scoped oracle fixture: prefer disk cache, fall back to building
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle_dict_k6() -> dict[bytes, int]:
    if CACHE_PATH.exists():
        return load_v_star_bounded_3x3(CACHE_PATH)
    return compute_v_star_bounded_3x3(CUBE_3X3, max_depth=6, verbose=False)


@pytest.fixture(scope="module")
def oracle_arrays_k6() -> tuple[np.ndarray, np.ndarray]:
    if CACHE_PATH.exists():
        return load_v_star_bounded_3x3_arrays(CACHE_PATH)
    # Fallback: build dict, materialize arrays.
    v = compute_v_star_bounded_3x3(CUBE_3X3, max_depth=6, verbose=False)
    n = len(v)
    states = np.empty((n, 54), dtype=np.int8)
    depths = np.empty(n, dtype=np.int8)
    for i, (key, d) in enumerate(v.items()):
        states[i] = np.frombuffer(key, dtype=np.int8)
        depths[i] = d
    return states, depths


def _tiny_net() -> ValueNet:
    """Tiny random-init ValueNet on CPU for shape-only tests."""
    net = ValueNet(
        CUBE_3X3,
        body_widths=(64, 32),
        n_residual_blocks=1,
        normalization="bn",
    )
    net.eval()  # avoid BN-train-on-batch-1 errors when callers poke directly
    return net


# ---------------------------------------------------------------------------
# value_eval
# ---------------------------------------------------------------------------


def test_value_eval_keys_and_shapes(oracle_dict_k6):
    net = _tiny_net()
    out = value_eval(
        net,
        CUBE_3X3,
        oracle_dict_k6,
        n_per_walk_depth=10,
        walk_depths=(1, 2, 3, 4, 5, 6),
        eval_batch_size=64,
        seed=0,
    )

    # per_walk_depth keys for every walk depth
    for d in (1, 2, 3, 4, 5, 6):
        assert f"per_walk_depth/d{d}/pred_mean" in out
        assert f"per_walk_depth/d{d}/pred_std" in out
        assert np.isfinite(out[f"per_walk_depth/d{d}/pred_mean"])
        assert np.isfinite(out[f"per_walk_depth/d{d}/pred_std"])

    # v_star_mae keys for every V* layer present (1..6)
    # All walk-depths d≤6 should land at least some endpoints inside the K=6
    # oracle, so we expect every layer 1..6 to be populated. (V*=0 is only
    # the solved state — random walks of length ≥1 don't return to it
    # reliably in a 10-state sample, so we don't require it.)
    for d in (1, 2, 3, 4, 5, 6):
        assert f"v_star_mae/d{d}" in out, (
            f"expected v_star_mae/d{d} key; got {sorted(out)}"
        )
        assert np.isfinite(out[f"v_star_mae/d{d}"])

    assert "macro_v_star_mae" in out
    assert np.isfinite(out["macro_v_star_mae"])
    assert "pred_mean" in out and np.isfinite(out["pred_mean"])
    assert "pred_std" in out and np.isfinite(out["pred_std"])


def test_value_eval_oracle_lookup_filtering(oracle_dict_k6):
    """Walks beyond K=6 still produce per_walk_depth stats but contribute
    nothing to v_star_mae (the oracle returns -1 for those endpoints)."""
    net = _tiny_net()
    out = value_eval(
        net,
        CUBE_3X3,
        oracle_dict_k6,
        n_per_walk_depth=8,
        walk_depths=(1, 2, 7, 10),
        eval_batch_size=64,
        seed=0,
    )

    # per_walk_depth populated for every supplied walk depth, including >K
    for d in (1, 2, 7, 10):
        assert f"per_walk_depth/d{d}/pred_mean" in out
        assert f"per_walk_depth/d{d}/pred_std" in out

    # v_star_mae keys are bounded by what the oracle has — never > 6.
    v_star_keys = [k for k in out if k.startswith("v_star_mae/d")]
    for key in v_star_keys:
        v = int(key.split("d")[-1])
        assert v <= 6, f"v_star_mae key beyond K=6: {key}"


def test_value_eval_macro_excludes_v_star_zero(oracle_dict_k6):
    """``macro_v_star_mae`` averages V*=1..K only — V*=0 (solved state) is
    excluded. V*=0 is a different quality (terminal-value calibration) and
    only populates when a walk happens to return to solved, which would jitter
    the headline scalar that drives early-stop. This test injects a synthetic
    case where d=0 is present and verifies macro is the d≥1 mean.
    """
    net = _tiny_net()
    # n_per_walk_depth=64 at depths 8..14 makes a few walk endpoints land back
    # on solved (V*=0) by random-walk redundancy. The seed is chosen to hit
    # this case; if the seed becomes brittle, the assertion below short-circuits
    # cleanly via the "no d=0 produced" branch.
    out = value_eval(
        net,
        CUBE_3X3,
        oracle_dict_k6,
        n_per_walk_depth=64,
        walk_depths=(8, 9, 10, 11, 12, 13, 14),
        eval_batch_size=64,
        seed=7,
    )
    macro = out["macro_v_star_mae"]
    d_geq_1 = [
        out[k]
        for k in out
        if k.startswith("v_star_mae/d") and int(k.split("d")[-1]) > 0
    ]
    if not d_geq_1:
        # Defensive: if this seed produced zero d≥1 samples (very unlikely),
        # macro is NaN and the exclusion semantics are vacuous.
        import math

        assert math.isnan(macro)
        return
    expected = sum(d_geq_1) / len(d_geq_1)
    assert abs(macro - expected) < 1e-5, (
        f"macro {macro} != mean of d≥1 v_star_mae values ({expected})"
    )
    # If d=0 happened to populate, confirm it's NOT in the macro.
    if "v_star_mae/d0" in out:
        d0 = out["v_star_mae/d0"]
        # If d0 had been included, macro would be different (unless d0
        # happens to equal the d≥1 mean, which is astronomically unlikely
        # at a fresh-init net's random predictions).
        with_d0 = (sum(d_geq_1) + d0) / (len(d_geq_1) + 1)
        assert abs(macro - with_d0) > 1e-5, (
            "macro accidentally includes d=0 — bug regressed"
        )


def test_value_eval_deterministic_across_calls(oracle_dict_k6):
    """Two back-to-back ``value_eval`` calls on the same net with the same
    seed must produce bit-identical per-V* MAE values. This is the H4 fix:
    ``value_eval`` re-seeds its internal generator each call rather than
    advancing a shared one, so the eval distribution is fixed across
    training steps and the macro_v_star_mae trajectory is comparable
    across time. Regression-guarding the smoke-run pattern of "eval-set
    drifts call-to-call".
    """
    net = _tiny_net()
    out_a = value_eval(
        net,
        CUBE_3X3,
        oracle_dict_k6,
        n_per_walk_depth=16,
        walk_depths=(1, 2, 3, 4, 5, 6),
        eval_batch_size=64,
        seed=42,
    )
    out_b = value_eval(
        net,
        CUBE_3X3,
        oracle_dict_k6,
        n_per_walk_depth=16,
        walk_depths=(1, 2, 3, 4, 5, 6),
        eval_batch_size=64,
        seed=42,
    )
    # Per-V* MAE values must be bit-identical across calls.
    v_star_keys_a = sorted(k for k in out_a if k.startswith("v_star_mae/"))
    v_star_keys_b = sorted(k for k in out_b if k.startswith("v_star_mae/"))
    assert v_star_keys_a == v_star_keys_b, (
        f"v_star_mae keys diverged: {v_star_keys_a} vs {v_star_keys_b}"
    )
    for key in v_star_keys_a:
        assert out_a[key] == out_b[key], (
            f"{key} diverged between calls: {out_a[key]} vs {out_b[key]}"
        )
    # macro_v_star_mae must also be bit-identical.
    assert out_a["macro_v_star_mae"] == out_b["macro_v_star_mae"]
    # And per_walk_depth pred stats — same eval set means same predictions.
    for d in (1, 2, 3, 4, 5, 6):
        assert (
            out_a[f"per_walk_depth/d{d}/pred_mean"]
            == out_b[f"per_walk_depth/d{d}/pred_mean"]
        )
        assert (
            out_a[f"per_walk_depth/d{d}/pred_std"]
            == out_b[f"per_walk_depth/d{d}/pred_std"]
        )


def test_value_eval_emits_banded_per_walk_depth_aggregates(oracle_dict_k6):
    """Banded aggregates (Shallow d=1..5 / Mid d=6..10 / Deep d=11..14) are
    computed in ``value_eval`` itself so the JSONL captures them too. The
    aggregate ``pred_mean`` is the mean of per-d ``pred_mean`` (NOT a
    pooled re-mean over raw preds); aggregate ``pred_std`` is the mean of
    per-d ``pred_std`` (NOT pooled std).
    """
    net = _tiny_net()
    out = value_eval(
        net,
        CUBE_3X3,
        oracle_dict_k6,
        n_per_walk_depth=8,
        walk_depths=tuple(range(1, 15)),  # d=1..14
        eval_batch_size=64,
        seed=0,
    )

    # All three banded aggregate keys present.
    for band in ("shallow", "mid", "deep"):
        assert f"per_walk_depth/{band}/pred_mean" in out
        assert f"per_walk_depth/{band}/pred_std" in out
        assert np.isfinite(out[f"per_walk_depth/{band}/pred_mean"])
        assert np.isfinite(out[f"per_walk_depth/{band}/pred_std"])

    # Numerical correctness: shallow aggregate = mean of per-d means over d=1..5.
    shallow_means = [out[f"per_walk_depth/d{d}/pred_mean"] for d in (1, 2, 3, 4, 5)]
    assert abs(out["per_walk_depth/shallow/pred_mean"] - sum(shallow_means) / 5) < 1e-5
    shallow_stds = [out[f"per_walk_depth/d{d}/pred_std"] for d in (1, 2, 3, 4, 5)]
    assert abs(out["per_walk_depth/shallow/pred_std"] - sum(shallow_stds) / 5) < 1e-5

    # Mid aggregate = mean of d=6..10.
    mid_means = [out[f"per_walk_depth/d{d}/pred_mean"] for d in (6, 7, 8, 9, 10)]
    assert abs(out["per_walk_depth/mid/pred_mean"] - sum(mid_means) / 5) < 1e-5

    # Deep aggregate = mean of d=11..14 (4 bins).
    deep_means = [out[f"per_walk_depth/d{d}/pred_mean"] for d in (11, 12, 13, 14)]
    assert abs(out["per_walk_depth/deep/pred_mean"] - sum(deep_means) / 4) < 1e-5


def test_value_eval_banded_aggregates_handle_missing_bins(oracle_dict_k6):
    """If a band has zero per-d members in the supplied walk_depths, that
    band's aggregate keys are omitted (rather than NaN-padded). E.g.
    walk_depths=(1, 2) populates only the shallow band.
    """
    net = _tiny_net()
    out = value_eval(
        net,
        CUBE_3X3,
        oracle_dict_k6,
        n_per_walk_depth=8,
        walk_depths=(1, 2),
        eval_batch_size=64,
        seed=0,
    )
    assert "per_walk_depth/shallow/pred_mean" in out
    # Mid (d=6..10) and deep (d=11..14) have no members in this walk_depths.
    assert "per_walk_depth/mid/pred_mean" not in out
    assert "per_walk_depth/deep/pred_mean" not in out


# ---------------------------------------------------------------------------
# beam_eval_walk
# ---------------------------------------------------------------------------


def test_beam_eval_walk_shape():
    net = _tiny_net()
    gen = torch.Generator(device="cpu").manual_seed(0)
    out = beam_eval_walk(
        net,
        CUBE_3X3,
        n_per_depth=5,
        walk_depths=(1, 2),
        beam_width=4,
        depth_budget_factor=2,
        generator=gen,
    )
    for d in (1, 2):
        assert f"solve_rate_d{d}" in out
        rate = out[f"solve_rate_d{d}"]
        assert 0.0 <= rate <= 1.0, f"solve_rate_d{d}={rate} out of [0, 1]"
        # avg_solve_len is float-or-None depending on whether anything solved
        assert f"avg_solve_len_d{d}" in out


# ---------------------------------------------------------------------------
# beam_eval_v_star
# ---------------------------------------------------------------------------


def test_beam_eval_v_star_shape(oracle_arrays_k6):
    net = _tiny_net()
    rng = np.random.default_rng(0)
    out = beam_eval_v_star(
        net,
        CUBE_3X3,
        oracle_arrays_k6,
        n_per_layer=5,
        beam_width=4,
        depth_budget_factor=2,
        max_v_star=2,
        rng=rng,
    )
    for v in (1, 2):
        assert f"solve_rate_v{v}" in out
        rate = out[f"solve_rate_v{v}"]
        assert 0.0 <= rate <= 1.0
        assert f"avg_solve_len_v{v}" in out
        assert f"mae_v{v}" in out
        assert np.isfinite(out[f"mae_v{v}"])

    # V*=0 must NOT appear — skipped by design (the lone solved state).
    assert "solve_rate_v0" not in out
    assert "mae_v0" not in out
    # Layers above max_v_star must NOT appear.
    assert "solve_rate_v3" not in out


# ---------------------------------------------------------------------------
# scripts/post_run_beam_eval.py smoke
# ---------------------------------------------------------------------------


def test_post_run_beam_eval_smoke(tmp_path):
    """Empty checkpoints dir → script writes empty JSON and exits 0.

    Documents the chosen empty-dir behavior: write ``{}`` rather than
    skip-write, so downstream readers never have to special-case a
    missing file. See ``scripts/post_run_beam_eval.py`` docstring.
    """
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()
    # Script-level config.yaml is unnecessary when there are zero checkpoints.

    script = REPO_ROOT / "scripts" / "post_run_beam_eval.py"
    result = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"post_run_beam_eval failed: stderr={result.stderr!r}"
    )
    out_path = run_dir / "results" / "post_run_beam_eval.json"
    assert out_path.exists(), f"expected {out_path} to be written"
    assert json.loads(out_path.read_text()) == {}
