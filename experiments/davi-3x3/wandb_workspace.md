# davi-3x3 — W&B workspace setup

The `WandbAdapter` (see `src/rubik/training/wandb_sink.py`) routes JSONL fields into a namespace policy split across **six top-level sections**: `train/`, `value/`, `beam_walk/`, `beam_v_star/`, `checkpoint/`, `run/`. Set up the panels below once, then **save the layout as a Workspace template** so future runs auto-inherit it.

## Section layout

| Namespace | Source records | What lives here |
| --- | --- | --- |
| `train/` | `event="step"` | per-step training scalars: `loss`, `step_seconds`, `k_max` |
| `value/` | `event="eval"` (i.e. `value_eval` records) | forward-pass value-net eval: `macro_v_star_mae`, `pred_mean`, `pred_std`, `v_star_mae/d{NN}`, `per_walk_depth/d{NN}/pred_{mean,std}`, `per_walk_depth/{shallow,mid,deep}/pred_{mean,std}` |
| `beam_walk/` | `event="beam_eval_walk"` | random-walk beam capability: `solve_rate/d{NN}`, `avg_solve_len/d{NN}` |
| `beam_v_star/` | `event="beam_eval_v_star"` | V*-stratified beam capability: `solve_rate_v{N}`, `avg_solve_len_v{N}`, `mae_v{N}` |
| `checkpoint/` | `event="checkpoint"` | `path` per checkpoint emission |
| `run/` | `event="run_start"` / `event="run_end"` | run-config scalars (`n_params`, `device`, `body_widths`, `n_steps`, ...) and run-end finals |

**Two W&B-specific niceties baked in:**

- **Zero-padded `d{NN}` keys.** All `d{N}` segments in W&B keys are zero-padded to two digits (`d01`, ..., `d14`). The on-disk JSONL records keep the unpadded form (`d1`, ..., `d14`) so existing analyzers don't break — padding is purely for W&B's natural-sort behavior in panel legends. Without it, panels go `d1, d10, d11, d12, d13, d14, d2, d3, ..., d9`.
- **`define_metric("*", step_metric="step")`** is called once after `wandb.init()` in `experiments/davi-3x3/run.py`, so all panels default to **training step** on the x-axis instead of W&B's `_step` log-call counter.

## Panel groups

### Train loss (`train/`)
- **Line plot.** y = `train/loss`, x = `step`. Use log-y; loss spans ~3 orders of magnitude over a typical run.
- Optional companion: `train/step_seconds` over `step` to spot per-step regressions.
- Optional companion: `train/k_max` over `step` — confirms curriculum schedule.

### Value-net macro / forever-metric trajectory (`value/`)
- **Line plot.** y = `value/macro_v_star_mae`, x = `step`. Drives the eye to the calibration trajectory.
- Companion lines: `value/pred_mean`, `value/pred_std` (right axis if scales clash).

### Per-V* MAE (`value/`)
- **Line plot.** y = wildcard `value/v_star_mae/*`, x = `step`. With zero-padded keys this groups d=01..06 in natural order.
- Filter by `event = "eval"` to drop the "no record at this step" gaps.

### Per-walk-depth predicted-V* (`value/`)
- **Line plot.** y = wildcard `value/per_walk_depth/d*/pred_mean`, x = `step`. Auto-renders 14 lines (d=01..14) sorted naturally.
- Mirror plot: `value/per_walk_depth/d*/pred_std` on its own panel.
- **Banded aggregates** (`shallow` / `mid` / `deep`) live alongside the per-d keys for at-a-glance reads — promote those to a separate small panel.

### Beam capability (`beam_walk/`)
- **Line plot.** y = wildcard `beam_walk/solve_rate/*`, x = `step`. Renders the per-walk-depth solve-rate trajectory across checkpoints.
- Mirror plot: `beam_walk/avg_solve_len/*`.

### Beam V* capability (`beam_v_star/`)
- **Line plot.** y = wildcard `beam_v_star/solve_rate_v*`, x = `step`.
- Mirror plot: `beam_v_star/avg_solve_len_v*`.
- Note: V*=1..6 is bounded by the K=6 oracle; this section currently saturates at 100% on trained checkpoints — useful as a sanity floor, not a discrimination signal. (Backlog: extend oracle to K=8.)

### Run config (`run/`)
- Drop `run/n_params`, `run/device`, `run/n_steps` etc. into a "Run config" group as labels — emitted once per run on `event = "run_start"`, no plots needed.

## Saving as a template

Once the panels look right: **Workspace menu → Save as template** (or "Save layout"). Future runs in the same project pick up the same panel arrangement automatically.
