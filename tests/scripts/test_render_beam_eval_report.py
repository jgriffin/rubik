"""Tests for ``scripts/render_beam_eval_report.py`` — dedicated HTML renderer.

Visual artifact, so tests are minimal: smoke (HTML written, contains
SVG + the widths/depths from the input) and V*-section presence when
the input carries v_star_results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_beam_eval_report.py"


def _synthetic_payload(
    *,
    widths=("8", "16", "32"),
    walk_depths=(1, 2, 3, 4),
    include_v_star=False,
) -> dict:
    """Build a synthetic beam-eval JSON the way beam_eval_run.py would."""
    results = {}
    for i, w in enumerate(widths):
        per_walk_depth = []
        for d in walk_depths:
            # Plausible monotonic-down solve rate as depth grows.
            rate = max(0.0, 1.0 - 0.15 * d - 0.05 * (len(widths) - 1 - i))
            per_walk_depth.append(
                {
                    "d": int(d),
                    "solve_rate": float(rate),
                    "avg_solve_len": float(d) if rate > 0 else None,
                    "n": 100,
                }
            )
        results[w] = {
            "per_walk_depth": per_walk_depth,
            "wall_time_seconds": 1.0 + float(int(w)) / 16.0,
            "states_scored": 100 * len(walk_depths),
        }
    payload = {
        "checkpoint": "/tmp/synthetic.pt",
        "config_path": "/tmp/synthetic_config.yaml",
        "device": "cpu",
        "max_walk_depth": int(max(walk_depths)),
        "n_per_depth": 100,
        "seed": 0,
        "results": results,
    }
    if include_v_star:
        v_star_results = {}
        for i, w in enumerate(widths):
            per_v_star = []
            for v in (1, 2, 3):
                rate = max(0.0, 1.0 - 0.2 * v - 0.05 * (len(widths) - 1 - i))
                per_v_star.append(
                    {
                        "v": int(v),
                        "solve_rate": float(rate),
                        "avg_solve_len": float(v) if rate > 0 else None,
                        "mae": 0.5,
                        "n": 200,
                    }
                )
            v_star_results[w] = {
                "per_v_star": per_v_star,
                "wall_time_seconds": 0.5 + float(int(w)) / 32.0,
            }
        payload["v_star_results"] = v_star_results
    return payload


def test_render_beam_eval_report_smoke(tmp_path):
    """Script writes HTML containing SVG and the widths + walk-depths."""
    payload = _synthetic_payload(widths=("8", "16"), walk_depths=(1, 2, 3))
    in_path = tmp_path / "beam_eval_run.json"
    in_path.write_text(json.dumps(payload))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(in_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"render failed: stderr={result.stderr!r}"
    out_path = in_path.with_suffix(".html")
    assert out_path.exists(), f"expected default {out_path} to be written"

    html = out_path.read_text()
    assert "<svg" in html
    # Width labels appear in the legend / heatmap headers.
    assert "w=8" in html
    assert "w=16" in html
    # Walk-depth labels appear in heatmap row headers and x ticks.
    assert "walk depth=1" in html
    assert "walk depth=3" in html


def test_render_beam_eval_report_v_star_section(tmp_path):
    """When the input has v_star_results, HTML includes V* labels."""
    payload = _synthetic_payload(
        widths=("4", "8"), walk_depths=(1, 2), include_v_star=True
    )
    in_path = tmp_path / "beam_eval_run.json"
    in_path.write_text(json.dumps(payload))
    out_path = tmp_path / "out.html"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(in_path),
            "--output",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"render failed: stderr={result.stderr!r}"
    html = out_path.read_text()
    # V* sections render the per-V* labels.
    assert "Per-V*" in html or "Per V*" in html or "V*=1" in html
    # Heatmap row headers for V*.
    assert "V*=1" in html
    assert "V*=2" in html


# ---------------------------------------------------------------------------
# Flat-schema (new) inputs
# ---------------------------------------------------------------------------


def _flat_payload(
    *, beam_width=256, precision="bf16", walk_depths=(1, 2, 3), n=8
) -> dict:
    """Synthetic flat-schema payload (as ``beam_eval_model.py`` writes)."""
    pwd = []
    for d in walk_depths:
        rate = max(0.0, 1.0 - 0.2 * d)
        pwd.append(
            {
                "d": int(d),
                "n": int(n),
                "solve_rate": float(rate),
                "avg_solve_len": float(d) if rate > 0 else None,
            }
        )
    return {
        "model": "/tmp/synthetic.pt",
        "config_name": "fast",
        "config_description": "test",
        "device": "cpu",
        "precision": precision,
        "max_depth": int(max(walk_depths)),
        "beam_width": int(beam_width),
        "seed": 0,
        "n_per_depth": [n] * len(walk_depths),
        "wall_time_seconds": 1.5,
        "per_walk_depth": pwd,
        "states_scored": int(n) * len(walk_depths),
    }


def test_render_flat_schema_single_input(tmp_path):
    """Flat-schema input renders a single overlay section with one column."""
    p = _flat_payload()
    in_path = tmp_path / "model_eval_fast.json"
    in_path.write_text(json.dumps(p))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(in_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out_path = in_path.with_suffix(".html")
    html = out_path.read_text()
    assert "<svg" in html
    # Walk-depth labels still appear.
    assert "walk depth=1" in html
    # Stem-based legend (no =sweep suffix).
    assert "model_eval_fast" in html


def test_render_flat_schema_multi_input_overlay(tmp_path):
    """Multiple flat-schema inputs render as one overlay with per-input labels."""
    a = _flat_payload(precision="fp32")
    b = _flat_payload(precision="bf16")
    in_a = tmp_path / "model_eval_fast_precision=fp32.json"
    in_b = tmp_path / "model_eval_fast_precision=bf16.json"
    in_a.write_text(json.dumps(a))
    in_b.write_text(json.dumps(b))
    out_path = tmp_path / "overlay.html"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(in_a),
            str(in_b),
            "--output",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = out_path.read_text()
    # Sweep-suffix legend labels surface, not "w=N".
    assert "precision=fp32" in html
    assert "precision=bf16" in html


# ---------------------------------------------------------------------------
# Trajectory mode
# ---------------------------------------------------------------------------


def _trajectory_payload(
    *, step: int, model_dir: str, walk_depths=tuple(range(1, 15)), wall=10.0
) -> dict:
    """Synthetic trajectory-style flat payload (same beam_width, same model dir)."""
    pwd = []
    for d in walk_depths:
        # Solve rate decays gently with depth + improves slightly with step.
        rate = max(0.0, min(1.0, 1.0 - 0.06 * d + step * 1e-6))
        pwd.append(
            {
                "d": int(d),
                "n": 50,
                "solve_rate": float(rate),
                "avg_solve_len": float(d) if rate > 0 else None,
            }
        )
    return {
        "model": f"{model_dir}/net_step_{step}.pt",
        "config_name": "fast",
        "config_description": "test",
        "device": "cpu",
        "precision": "bf16",
        "max_depth": int(max(walk_depths)),
        "beam_width": 256,
        "step": int(step),
        "seed": 0,
        "n_per_depth": [50] * len(walk_depths),
        "wall_time_seconds": float(wall),
        "per_walk_depth": pwd,
        "states_scored": 50 * len(walk_depths),
    }


def test_trajectory_mode_detected_when_steps_differ(tmp_path):
    """Two flat inputs with different step values + shared model parent → trajectory."""
    model_dir = str(tmp_path / "run_xyz")
    Path(model_dir).mkdir()
    a = _trajectory_payload(step=20000, model_dir=model_dir, wall=39.5)
    b = _trajectory_payload(step=120000, model_dir=model_dir, wall=42.0)
    in_a = tmp_path / "step_20000_eval_fast.json"
    in_b = tmp_path / "step_120000_eval_fast.json"
    in_a.write_text(json.dumps(a))
    in_b.write_text(json.dumps(b))
    out_path = tmp_path / "trajectory.html"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(in_a),
            str(in_b),
            "--output",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = out_path.read_text()

    # Trajectory-mode header references run dir + checkpoint count + step range.
    assert "run dir:" in html
    assert "checkpoints:" in html
    # Step labels appear (compact "20k" / "120k" form).
    assert "20k" in html
    assert "120k" in html
    # The banded layout produces a "depth 14" cell title (Deep band).
    assert "depth 14" in html
    # Banded title-line for the deep band.
    assert "Deep" in html
    # The trajectory-mode banner sentence in the body should be present.
    assert "Beam-eval trajectory" in html


def test_trajectory_heatmap_y_axis_reversed(tmp_path):
    """In trajectory-mode heatmap, walk_depth=14 row appears BEFORE walk_depth=1."""
    model_dir = str(tmp_path / "run_abc")
    Path(model_dir).mkdir()
    a = _trajectory_payload(step=10000, model_dir=model_dir)
    b = _trajectory_payload(step=20000, model_dir=model_dir)
    in_a = tmp_path / "step_10000_eval_fast.json"
    in_b = tmp_path / "step_20000_eval_fast.json"
    in_a.write_text(json.dumps(a))
    in_b.write_text(json.dumps(b))
    out_path = tmp_path / "out.html"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(in_a),
            str(in_b),
            "--output",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = out_path.read_text()

    # Both row-header labels exist.
    assert "walk_depth=14" in html
    assert "walk_depth=1" in html
    # And d=14 comes BEFORE d=1 in the source order (top of the heatmap).
    assert html.index("walk_depth=14") < html.index("walk_depth=1<")


def test_trajectory_step_appears_as_axis_label_not_filename(tmp_path):
    """Trajectory-mode legends/axis use the step value, not the filename."""
    model_dir = str(tmp_path / "run_def")
    Path(model_dir).mkdir()
    a = _trajectory_payload(step=20000, model_dir=model_dir)
    b = _trajectory_payload(step=40000, model_dir=model_dir)
    in_a = tmp_path / "step_20000_eval_fast.json"
    in_b = tmp_path / "step_40000_eval_fast.json"
    in_a.write_text(json.dumps(a))
    in_b.write_text(json.dumps(b))
    out_path = tmp_path / "out.html"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(in_a),
            str(in_b),
            "--output",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = out_path.read_text()

    # Step values surface as compact axis labels.
    assert "step 20k" in html
    assert "step 40k" in html
    # Filenames-as-labels (the old buggy fallback) should NOT be the
    # primary identifier in trajectory mode.
    assert "step_20000_eval_fast" not in html
    assert "step_40000_eval_fast" not in html


def test_sweep_mode_chosen_when_step_shared_but_beam_width_differs(tmp_path):
    """Same step, varying beam_width → sweep mode (not trajectory)."""
    model_dir = str(tmp_path / "run_ghi")
    Path(model_dir).mkdir()
    a = _trajectory_payload(step=20000, model_dir=model_dir)
    b = _trajectory_payload(step=20000, model_dir=model_dir)
    a["beam_width"] = 128
    b["beam_width"] = 512
    in_a = tmp_path / "model_eval_fast_w=128.json"
    in_b = tmp_path / "model_eval_fast_w=512.json"
    in_a.write_text(json.dumps(a))
    in_b.write_text(json.dumps(b))
    out_path = tmp_path / "out.html"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(in_a),
            str(in_b),
            "--output",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = out_path.read_text()
    # Sweep-mode signal: legend labels keyed off the varying field.
    assert "beam_width=128" in html
    assert "beam_width=512" in html
    # Trajectory-mode banner should NOT appear.
    assert "Beam-eval trajectory" not in html


def test_sweep_mode_chosen_when_model_parents_differ(tmp_path):
    """Different model parents → not a single training run → sweep mode."""
    dir_a = tmp_path / "run_one"
    dir_b = tmp_path / "run_two"
    dir_a.mkdir()
    dir_b.mkdir()
    a = _trajectory_payload(step=20000, model_dir=str(dir_a))
    b = _trajectory_payload(step=40000, model_dir=str(dir_b))
    in_a = tmp_path / "a.json"
    in_b = tmp_path / "b.json"
    in_a.write_text(json.dumps(a))
    in_b.write_text(json.dumps(b))
    out_path = tmp_path / "out.html"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(in_a),
            str(in_b),
            "--output",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = out_path.read_text()
    # Trajectory-mode banner should NOT appear.
    assert "Beam-eval trajectory" not in html
    # The varying field is `step` here (model differs but isn't preferred);
    # legend label should reflect step, not run dir.
    assert "step=20000" in html or "step=40000" in html
