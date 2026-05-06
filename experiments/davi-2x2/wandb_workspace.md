# davi-2x2 — W&B workspace setup

The `WandbAdapter` (see `src/rubik/training/wandb_sink.py`) routes JSONL fields into a namespace policy that mirrors what `analysis/analyze.py` produces locally. Set up the panels below once, then **save the layout as a Workspace template** so future runs auto-inherit it.

## Panel groups

### Train loss (mirrors `analyze.py` "Train loss" section)
- **Line plot.** y = `train/loss`, x = `step`. Use log-y; loss spans ~3 orders of magnitude over a typical 30k run.
- Optional companion: `train/step_seconds` over `step` to spot per-step regressions.
- Optional companion (only useful for cycle-4 curriculum runs): `train/k_max` over `step` — confirms the curriculum schedule executed as configured.

### Macro-MAE trajectory (mirrors "V* macro-MAE trajectory" section)
- **Line plot.** y = `eval/macro_mae`, x = `step`.
- Companion lines on the same plot: `eval/val_mae`, `eval/pred_std` (right axis if scales clash).

### Per-depth MAE (mirrors "Per-depth MAE — start / middle / end" section)
- **Line plot.** y = wildcard `eval/per_depth_mae/*`, x = `step`. W&B auto-renders these as 14 lines (d=1..14). Group by the trailing key segment in the legend.
- Filter by `event = "eval"` to drop the "no record at this step" gaps.

### Solve-rate trajectories per depth (mirrors "Greedy-policy solve rate trajectory" section)
- **Line plot.** y = wildcard `eval/solve_rate/*`, x = `step`. Groups d=1..14 onto one chart.
- Mirror plot: `eval/avg_solve_len/*` (same x, same wildcard pattern).

### Optional: run-level scalars
- Drop `run/n_params`, `run/device`, `run/n_steps` etc. into a "Run config" group as labels — they're emitted once per run on `event = "run_start"` and don't need plots.

## Saving as a template

Once the panels look right: **Workspace menu → Save as template** (or "Save layout"). Future runs in the same project pick up the same panel arrangement automatically.
