> **PARKED 2026-05-04:** pivoted back to V*-supervised with per-depth loss weighting. DAVI infrastructure (eval.py, eval_set.npz, run.py, DAVIConfig with normalization field) retained as method-agnostic; reviving DAVI is a future M7+ phase. See `experiments/davi-2x2/v-star-weighted/`.

# davi-baseline — first DAVI run after the V*-supervised pivot

## Question

Does DAVI training on the pragmatic-pick `(4096, 1024)` BN n=2 network reach M5 acceptance — `macro-MAE < 1.0` AND `greedy-solve > 99% at d ≤ 14` — in 30k steps with placeholder dynamics hyperparameters?

## What's an informed pick vs. a placeholder

- **Architecture** `(4096, 1024)` BN `n_residual_blocks=2` — the comfortable network the supervised T1 ablations landed on (all uniform-sampling cells plateaued, but this size + BN had the best behaviour under depth-balanced sampling). Informed pick.
- **`max_scramble_depth: 7`** — placeholder. Half the QTM diameter, neither trivial nor saturating. The methodology's deferred T3 was meant to sweep this. Not yet earned.
- **`target_sync_interval: 200`** — placeholder. T3 in the methodology was supposed to sweep this jointly with curriculum depth. Not yet earned.
- **`learning_rate: 1e-3`** — supervised T1 ablations showed `1e-3` and `3e-3` within seed noise on the matched architecture under uniform sampling. Best informed value we have for this architecture, but it has not been swept under the DAVI regime; treat as a defensible carry-over.

When this run lands and we see what DAVI's actual training dynamics look like, the placeholders become the natural next sweep.

## Acceptance + plan

- **Pass:** `macro_mae < 1.0` AND every `solve_rate_d{d} > 0.99` at the final eval (greedy depths {1,3,5,7,9,11,13}).
- **Probable-fail informative outcomes worth writing up:** macro-MAE plateau (curriculum or sync-interval is the bottleneck); solve rate stalls at low depths (greedy is fine but value approximation fails on hard states); train loss diverges.
- **Writeup:** intuition + observations land in `intuition.md` (hand-written, per project convention); `analyze.py` regenerates `results.md` with curves + tables.

## Files

- `configs/baseline.yaml` — DAVIConfig YAML with the values above.
- `run.sh` — one-line wrapper.
- `analyze.py` — reads `runs/baseline-30k/metrics.jsonl` and writes a `results.md` skeleton.
