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
