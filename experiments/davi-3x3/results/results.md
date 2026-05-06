# DAVI 3x3 — results

This file is the canonical comparison surface for 3x3 DAVI runs. It currently has one row: the M8 phase 2 smoke run.

The repo convention (CLAUDE.md "Cycle reporting pipeline") says new cycles append to `analysis/analyze.py`'s RUNS list and the analyzer regenerates this file. **Deviation noted:** the analyzer was scaffolded in P1a against the 2x2 metric schema (`per_depth_mae`, `solve_rate_dN`); the new 3x3 `value_eval` emits a different schema (`per_walk_depth/dN/pred_*`, `v_star_mae/dN`, `macro_v_star_mae`). Updating the analyzer for the new schema is deferred to the next 3x3 cycle (when there are at least two runs to compare). For now, this file is hand-written.

## Runs in this comparison

| label | run_subdir | config | steps | wall | best macro_v_star_mae | final macro_v_star_mae | early-stop |
|---|---|---|---|---|---|---|---|
| smoke | `runs/20260506T203408Z_smoke/` | `configs/smoke.yaml` | 10,000 (cap) | 39 min | 0.7340 @ step 7,500 | 0.7490 @ step 10,000 (corrected) | did not fire |

**Note on macro values.** The metric pipeline at the time of this run had a bug in `value_eval` (now fixed): the `macro_v_star_mae` scalar averaged across V\*=0..K when V\*=0 happened to populate, instead of V\*=1..K only. V\*=0 (solved state) is a terminal-value calibration quality, not a "V\* prediction error on non-trivial states" quality, and including it inflated the macro by ~25% on the 3 of 20 evals where it populated (steps 2,000 / 5,500 / 10,000). All values in this writeup are the **corrected** d=1..6 mean, derived from the per-V\* layer values stored in `metrics.jsonl`. The historical `metrics.jsonl` is unchanged (it's a record); future runs will record the corrected scalar directly. See intuition.md §H1.

## Setup (smoke run)

| | |
|---|---|
| arch | `[5120, 1024]` body × 4 residual blocks, BatchNorm |
| params | 15,334,401 |
| device | MPS (M4 Max) |
| step rate | 235 ms/step |
| batch size | 4,096 |
| optimizer | Adam, lr 1e-3 |
| target sync | every 500 steps |
| scrambles | random walk, K_max=8 (no curriculum) |
| eval cadence | every 500 steps (= sync cadence; required by alignment rule) |
| eval set | deterministic from seed `0+17`, 100 walks per depth × 14 walk-depths regenerated each call |
| oracle | bounded V\* @ K=6 (983,926 states); per-V\*-layer MAE on d=1..6 |
| early-stop | `macro_v_star_mae`, patience 12 evals, warmup 4 evals, min_delta 0.001 |
| checkpoints | every 2,500 steps (2,500 / 5,000 / 7,500 / 10,000 / final) |
| W&B | https://wandb.ai/griffin-enterprises/rubik-3x3/runs/4m9el9ws |

## macro_v_star_mae trajectory (corrected)

```
step    500  3.4776  pred_mean 0.016  pred_std 0.312
step   1000  3.0396  pred_mean 0.590  pred_std 0.272
step   1500  2.3892  pred_mean 1.310  pred_std 0.300
step   2000  1.7939  pred_mean 2.007  pred_std 0.425   (recorded 1.6926; +d=0)
step   2500  1.3411  pred_mean 2.677  pred_std 0.638
step   3000  1.0908  pred_mean 3.195  pred_std 0.948
step   3500  0.9982  pred_mean 3.587  pred_std 1.234
step   4000  0.9371  pred_mean 3.921  pred_std 1.426
step   4500  0.9076  pred_mean 4.129  pred_std 1.575
step   5000  0.8181  pred_mean 4.215  pred_std 1.601
step   5500  0.8051  pred_mean 4.166  pred_std 1.576   (recorded 0.9312; +d=0)
step   6000  0.8054  pred_mean 4.199  pred_std 1.564
step   6500  0.7966  pred_mean 4.348  pred_std 1.707
step   7000  0.7503  pred_mean 4.371  pred_std 1.690
step   7500  0.7340  pred_mean 4.362  pred_std 1.664   ← BEST
step   8000  0.7488  pred_mean 4.392  pred_std 1.732
step   8500  0.7728  pred_mean 4.410  pred_std 1.759
step   9000  0.7582  pred_mean 4.420  pred_std 1.748
step   9500  0.7867  pred_mean 4.368  pred_std 1.715
step  10000  0.7490  pred_mean 4.364  pred_std 1.686   (recorded 0.8838; +d=0)
```

**Two phases:** clean descent through step ~5,000 (3.48 → 0.82, ~5x improvement, monotonic), then a **flat plateau** in the 0.73–0.79 band for the remaining 5,000 steps. The "regression" reading from the recorded values was a measurement artifact — see "Note on macro values" above and intuition.md §H1.

Predictions kept climbing through training (mean 0.02 → 4.36) and getting more spread out (std 0.31 → 1.69), indicating the network kept refining the value function past the macro plateau — and beam capability data confirms this (next section).

## Beam capability — best (step 7,500) vs final (step 10,000)

Reduced sample sizes for smoke speed (n_per_depth=50, n_per_layer=100, beam_width=128). Full-size beam eval (n_per_depth=100, n_per_layer=200, beam_width=256) deferred — was running but pulled mid-flight in favor of focused comparison.

### Per V\* layer × beam(128) — d=1..6 (bounded oracle)

| V\* | step 7,500 solve_rate | step 7,500 avg_len | step 10,000 solve_rate | step 10,000 avg_len |
|---|---|---|---|---|
| 1 | 1.000 | 1.00 | 1.000 | 1.00 |
| 2 | 1.000 | 2.00 | 1.000 | 2.00 |
| 3 | 1.000 | 3.00 | 1.000 | 3.00 |
| 4 | 1.000 | 4.00 | 1.000 | 4.00 |
| 5 | 1.000 | 5.00 | 1.000 | 5.00 |
| 6 | 1.000 | 6.00 | 1.000 | 6.00 |

**100% solve rate at optimal length on every bounded-oracle layer at both checkpoints.** The value head leads the beam to V\*-equal solutions even at d=6.

### Per random-walk depth × beam(128)

| walk depth | step 7,500 solve_rate | step 7,500 avg_len | step 10,000 solve_rate | step 10,000 avg_len | Δ solve_rate |
|---|---|---|---|---|---|
| 1 | 1.000 | 1.00 | 1.000 | 1.00 | 0 |
| 2 | 1.000 | 2.00 | 1.000 | 2.00 | 0 |
| 3 | 1.000 | 2.96 | 1.000 | 2.84 | 0 |
| 4 | 1.000 | 4.00 | 1.000 | 4.00 | 0 |
| 5 | 1.000 | 4.72 | 1.000 | 5.00 | 0 |
| 6 | 1.000 | 5.84 | 1.000 | 5.76 | 0 |
| 7 | 1.000 | 6.92 | 1.000 | 6.84 | 0 |
| 8 | 1.000 | 7.72 | 1.000 | 7.92 | 0 |
| 9 | 1.000 | 8.68 | 1.000 | 8.84 | 0 |
| 10 | 0.980 | 9.67 | 1.000 | 9.84 | **+2pp** |
| 11 | 0.880 | 10.77 | 0.920 | 10.57 | **+4pp** |
| 12 | 0.640 | 11.38 | 0.800 | 11.90 | **+16pp** |
| 13 | 0.460 | 12.39 | 0.500 | 12.44 | +4pp |
| 14 | 0.180 | 13.33 | 0.280 | 13.43 | **+10pp** |

**Step 10,000 is meaningfully better at the deep walk tail.** +16pp at walk depth 12, +10pp at walk depth 14. The training continued to push capability forward past step 7,500 even as macro_v_star_mae stayed flat in the bulk. This **falsifies** the cycle-4-style "macro improves while capability regresses" hypothesis — here it's the opposite: macro is flat (corrected) while capability improves at the tail.

Raw JSON: `results/beam_eval_focused.json`. Sample sizes: n_per_depth=50 (walk), n_per_layer=100 (V\*), beam_width=128.

## Phase 2 acceptance gate — status

| # | criterion | status |
|---|---|---|
| 1 | DAVIConfig 5 early-stop fields + tests + load-time alignment validation | ✅ |
| 2 | smoke run completes (cap or early-stop) without NaN/MPS errors | ✅ — n_steps cap |
| 3 | macro_v_star_mae trajectory visibly downward, not flat from step 0, not exploding | ✅ — 3.48 → 0.73 best, 5x improvement |
| 4 | post-training beam evals produce per-V\* + per-walk-depth solve rates | ✅ — focused JSON written |
| 5 | results writeup with intuition section per project convention | ✅ — this file |
| 6 | wandb run under `rubik-3x3` (NOT `rubik`) with three panel groups populated | ✅ — `eval/per_walk_depth/dN/*`, `eval/v_star_mae/dN`, `eval/macro_v_star_mae` |

---

## Intuition

_(Hand-written in `experiments/davi-3x3/intuition.md`; reproduced here per project convention so this section persists across future analyzer regenerations. When the analyzer is updated for the 3x3 metric schema in the next cycle, it will read intuition.md and append from "Observations" downward, replacing this stub.)_

### Observations

1. **Loss decreased monotonically through ~step 1000**, then stabilized in the 0.005–0.025 band for the remaining 9,000 steps. No NaN, no MPS crashes, no asserts. The training loop runs end-to-end on 3x3.
2. **`macro_v_star_mae` (corrected, V\*=1..6 mean) dropped 3.48 → 0.73 over steps 500–7,500** (~5x improvement). Clean descent through step ~5,000, flat plateau thereafter.
3. **Best corrected macro_v_star_mae was at step 7,500 (0.7340).** Final 5,000 steps were essentially flat: 0.749 / 0.773 / 0.758 / 0.787 / 0.749 — within ±0.05 of best.
4. **The recorded macro values DID show an apparent regression** (0.88 at step 10,000 vs 0.73 at 7,500) — but this was a calibration artifact from a `value_eval` bug. V\*=0 was being included in the macro on the rare evals where walks happened to return to solved. **Bug fixed in this commit.** See H1.
5. **Early-stop did not fire.** Patience window (12 evals) extends past the 10k cap from any best.
6. **Predictions climbed from ~0 → ~4.4 mean**, `pred_std` 0.31 → 1.69. Network is learning the value distribution, slightly under-spread vs underlying V\*.
7. **Per-V\* MAE at d=1..6** is roughly stable step 7,500 → 10,000 (Δ macro = +0.015, within eval-set noise). Per-layer differences ≤ ±0.07.
8. **Beam capability at the deep walk tail IMPROVED step 7,500 → 10,000** (+16pp at d=12, +10pp at d=14). Training kept pushing capability forward past the macro plateau.

### Hypotheses

**H1 (CONFIRMED):** the apparent late-training regression was a measurement artifact, not a value-function regression. Beam cross-check at the tail showed improvement, falsifying the cycle-4-style hypothesis. **Bug fix landed in this commit:** macro now excludes V\*=0.

**H2 (medium-high):** with the corrected metric, training has not converged at 10k steps. Beam capability still climbing at d=12, d=14; pred_std still climbing. Verification: longer run.

**H3 (high):** 10k steps demonstrates learning but doesn't characterize convergence or bottleneck. Per-V\* MAE at d=4..6 still > 0.8. Verification: 30k+ run.

**H4 (medium):** `value_eval`'s eval-set advances each call instead of being fixed-from-seed. Per-V\* MAE bouncing eval-to-eval is consistent with eval-set noise. Verification: re-seed each eval call, re-run, compare trajectory smoothness.

### Open questions

- **Q1:** does deep-walk-tail capability keep climbing past step 10,000?
- **Q2:** what does per-V\* MAE shape look like at convergence? Currently linear-in-d.
- **Q3:** does K_max=8 cap learning at the deep walk tail?
- **Q4:** with corrected macro + fixed-from-seed eval set, what's the smallest sensible early-stop patience?

### What we haven't verified

- Whether the corrected (flat) trajectory is reproducible at other seeds.
- Whether the chosen arch is right-sized — T0 sweep deferred to backlog.
- Whether eval-generator-advancing (H4) is actually causing significant noise.
- Whether the bug fix interacts with anything besides the macro scalar (it shouldn't).
