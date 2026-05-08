"""Reproducer for the precision-sweep experiment.

This experiment evaluates whether casting the trained ValueNet to lower
inference precision (BF16, FP16) on MPS preserves solve-rate equivalence
while reducing wall time. Inference-only — the training loop is untouched.

TODO(beam-solve-perf): the original sweep was driven by the pre-rename
``scripts/beam_eval_run.py`` (which became ``scripts/beam_eval_model.py``)
with ``--widths`` and ``--precision``. This reproducer now invokes
``scripts/beam_eval_sweep.py --over precision`` and ``--over width``
instead — single-checkpoint sweeps, one JSON per swept value (flat
schema). Existing per-cell JSONs under ``results/`` are still in the
legacy width-keyed schema; the renderer auto-detects and handles both.

All sweeps target the same baseline checkpoint
``experiments/davi-3x3/runs/20260507T043533Z_full_train/net_final.pt``,
``--n-per-depth 100``, ``--max-walk-depth 14``, ``--seed 0`` — so cross-run
deltas isolate the precision change.

Datestamp + run conditions are documented in ``results.md``. Hand-written
intuition is in ``intuition.md`` per project convention.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "davi-3x3"
    / "runs"
    / "20260507T043533Z_full_train"
    / "net_final.pt"
)


def run(precision: str, widths: str, name: str) -> None:
    """Execute a single beam_eval_run sweep."""
    output = RESULTS / f"sweep_{name}.json"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/beam_eval_run.py",
        "--checkpoint",
        str(CHECKPOINT),
        "--widths",
        widths,
        "--n-per-depth",
        "100",
        "--max-walk-depth",
        "14",
        "--precision",
        precision,
        "--output-json",
        str(output),
        "--seed",
        "0",
    ]
    print(f"[{name}] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Phase 1: width=128 only, all three precisions — fast comparison.
    run("fp32", "128", "fp32_w128")
    run("bf16", "128", "bf16_w128")
    run("fp16", "128", "fp16_w128")
    # Phase 2: 7-width sweep, FP32 baseline + BF16 (FP16 skipped — width=128
    # already shows BF16 ≈ FP16 wall and identical solve rates, so the
    # 7-width FP16 cell has no information beyond BF16's).
    run("fp32", "8,16,32,64,128,256,512", "fp32_full")
    run("bf16", "8,16,32,64,128,256,512", "bf16_full")
