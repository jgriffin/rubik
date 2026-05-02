# Batch-size sensitivity sweep (2x2)

## Purpose

Measure how `apply_moves` and `random_scrambles` throughput scales with
batch size on this M4 Max. The curve shape tells us where dispatch
overhead dies, where memory bandwidth or kernel-launch cost becomes
the wall, and which regimes are actually GPU-saturated. Inputs to M7
(beam search) and M8 (DAVI training) hyperparameter choices.

## What's swept

- **Ops:** `apply_moves`, `random_scrambles` (depth=20).
- **Batch sizes:** powers of 8 from 1 to ~2M (`1, 8, 64, 512, 4096,
  32768, 262144, 2097152`).
- **Per cell:** 5 warmup + 30 timed trials, bracket-synced via
  `rubik.perf.time_op`.
- **CIs:** bootstrap 95% on throughput.

Full grid in `config.yaml`.

## How to run

```bash
uv run python experiments/batch-sensitivity-2x2/run.py
uv run python experiments/batch-sensitivity-2x2/analyze.py
open experiments/batch-sensitivity-2x2/results.md
```

`run.py` writes `runs/<ts>/data.json` (gitignored). `analyze.py` reads
the newest run and rewrites `results.md` with the table + intuition
observations. Both scripts land in commit 6 of the M4 ladder.
