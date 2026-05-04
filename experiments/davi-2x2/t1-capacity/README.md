# T1 — capacity floor

**Status:** ⚠️ HALT-AND-DEBUG (2026-05-04). Not closed with a pick.

**Question:** What's the smallest `(h1, h2, n_residual_blocks)` value-network
shape that fits V\* on the 2x2 to val_mae < 0.5 by direct supervised
regression?

**Answer (this cycle):** None of the 5 Phase A cells (widths 207K–4.80M
params, n_residual=0, batch 1024, 7000 steps, lr=1e-3) plus 3 follow-up
diagnostic variants (more steps, more residuals, higher LR) reached the
threshold. All cells plateau at val_mae ≈ 0.85–0.90, very close to the
predict-the-mean baseline (0.9286). Diagnosis: MSE loss × peaked V\*
depth distribution × BatchNorm-MLP traps optimization in a
mean-collapse regime.

Phase B (residual sweep) was not opened, per methodology §T1's
halt-and-debug provision.

See [`results.md`](results.md) for the full Phase A table + diagnostic
runs + diagnosis. See [`intuition.md`](intuition.md) for hypotheses and
the four specific next-cycle debug experiments.

## Files

- `supervised.py` — thin V\*-supervised trainer (no DAVI; MSE loss; Adam;
  reads `data/v_star_2x2.npz`; seed-deterministic 80/20 split).
- `configs/phaseA_*.yaml` — 5 width-sweep configs, all `n_residual=0`.
- `run.sh` — driver: `bash run.sh A` runs all `phaseA_*.yaml` sequentially.
- `analyze.py` — aggregates Phase A + B logs into `results.md` + `_picks.json`.
- `_picks.json` — machine-readable pick (this cycle: `status:
  halt_and_debug`, no pick).
- `intuition.md` — hand-written: why, expected, observations,
  hypotheses, open questions.
- `results.md` — full results writeup (this cycle: halt-and-debug).
- `plots/phaseA_pareto.{svg,html}` — Pareto frontier (essentially
  horizontal — capacity does not explain val_mae here).

## Reproducing

```sh
# Phase A (5 cells, ~3 min wall):
bash experiments/davi-2x2/t1-capacity/run.sh A

# Aggregate + write results.md + _picks.json:
uv run python experiments/davi-2x2/t1-capacity/analyze.py
```

The diagnostic runs (`runs/diag_*`) were one-off — their YAML configs
live in `runs/diag_<name>/config.yaml` (snapshotted by `supervised.py`)
if you need to reproduce.
