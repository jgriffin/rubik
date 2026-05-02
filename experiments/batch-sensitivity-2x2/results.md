# Batch-size sensitivity sweep (2x2)

## What this dir is

Sweeps `apply_moves` and `random_scrambles` throughput across batch sizes on this M4 Max to find the dispatch-overhead floor, the saturation ceiling, and any allocator/memory cliffs. The curve shapes are inputs to M7 (beam search) and M8 (DAVI training) hyperparameter choices, and a deliberate exercise in building GPU-saturation intuition for this hardware.

Sweep grid lives in `config.yaml`. Driver: `run.py`. Steady-state spot-check: `run_steady_state.py`. Renderer: `analyze.py`.

## Latest run

- **Timestamp:** `2026-05-02T05:55:51.076143Z`
- **Machine:** `macOS-26.3-arm64-arm-64bit`
- **Torch:** `2.11.0`
- **Git SHA:** `63242a219fc4e668c4a5d57e405662b4e898121f`
- **Data:** `experiments/batch-sensitivity-2x2/runs/2026-05-02T05-55-50Z/data.json` (gitignored)
- **Sweep ops:** `['apply_moves']`
- **Batch sizes:** `[64]`
- **Trials per cell:** 3; warmup: 2

## Methodology pointers

- `experiments/mps-methodology/results.md` — full triangulation strategy (bench-bracket → macmon → profiler), gotchas list, tools matrix. Read first if these numbers look surprising.
- `src/rubik/perf/bench.py` — `time_op` (warmup → sync → perf_counter → fn → sync → perf_counter) and `bootstrap_ci` (median + 95% CI via numpy resampling).
- `experiments/batch-sensitivity-2x2/config.yaml` — sweep grid.

## Throughput tables

### `apply_moves` (states/s)

| batch | median (ms) | CI lo (ms) | CI hi (ms) | throughput (median) | throughput CI lo | throughput CI hi |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.749 | 0.980 | 2.143 | 36.596 kstates/s | 29.864 kstates/s | 65.289 kstates/s |

## Observations

Reading these as GPU-saturation intuition for this M4 Max: dispatch overhead dominates at the smallest batches (per-call time is flat regardless of how many states we hand the GPU); the curve lifts when batch is large enough to amortize kernel launch; it plateaus when memory bandwidth or compute throughput is the wall.

### `apply_moves`

- Single batch size in this run — saturation/dispatch curve needs the full sweep (commit 7) to read.

## Bench-bracket vs steady-state regime

Calibration pending — run `run_steady_state.py` and rerun `analyze.py --steady-state-run <path>` to embed the bench-vs-pipelined correction factor here. Until then: the sweep numbers are sync-bracket per-call seconds, which underestimate steady-state pipelined throughput by a regime-dependent factor (see `experiments/mps-methodology/results.md` §5).
