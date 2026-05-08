"""Reproducer for the precision-sweep experiment.

This experiment evaluates whether casting the trained ValueNet to lower
inference precision (BF16, FP16) on MPS preserves solve-rate equivalence
while reducing wall time. Inference-only — the training loop is untouched.

The original sweep was driven by the pre-rename ``scripts/beam_eval_run.py``
(now ``scripts/beam_eval_model.py``) with ``--widths`` and ``--precision``
combined into a single invocation. After the beam-solve-perf refactor,
sweeps live in ``scripts/beam_eval_sweep.py`` (one JSON per swept value)
and the per-cell width-keyed schema under ``results/`` is the legacy
shape (still rendered by the schema-aware renderer).

To re-run the experiment from scratch with the new tooling:

    # Phase 1: precision sweep at width=128 (--width override pins the width).
    uv run python scripts/beam_eval_sweep.py \\
        experiments/davi-3x3/runs/20260507T043533Z_full_train/net_final.pt \\
        --over precision --values "fp32,bf16,fp16" \\
        --width 128 --config default

    # Phase 2: width sweeps at fixed precisions.
    uv run python scripts/beam_eval_sweep.py \\
        experiments/davi-3x3/runs/20260507T043533Z_full_train/net_final.pt \\
        --over width --values "8,16,32,64,128,256,512" \\
        --precision fp32 --config default

    uv run python scripts/beam_eval_sweep.py \\
        experiments/davi-3x3/runs/20260507T043533Z_full_train/net_final.pt \\
        --over width --values "8,16,32,64,128,256,512" \\
        --precision bf16 --config default

Outputs land next to the checkpoint under
``experiments/davi-3x3/runs/20260507T043533Z_full_train/`` with names
like ``net_final_eval_default_width=128.json``.

All sweeps target the same baseline checkpoint, ``seed=0``, and the
``default`` schedule (``n_per_depth`` deep-tail at 100) so cross-run
deltas isolate the swept axis.

Datestamp + run conditions are documented in ``results.md``. Hand-written
intuition is in ``intuition.md`` per project convention.

This file is now documentation-only: the existing per-cell JSONs in
``results/sweep_*.json`` are in the legacy width-keyed schema and the
renderer continues to consume them.
"""

if __name__ == "__main__":
    raise SystemExit(
        "This reproducer is now documentation-only — see the docstring "
        "above for the new beam_eval_sweep.py invocations."
    )
