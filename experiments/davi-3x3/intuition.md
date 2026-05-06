# Intuition — 3x3 DAVI smoke training

Hand-written per the project convention (see CLAUDE.md "Experiment results.md format"). Distinguishes observation from inference from hypothesis; hypotheses get verification plans, not authoritative tone.

Run: `runs/20260506T203408Z_smoke/` — single smoke training, 10,000 steps, hand-picked arch `[5120, 1024] × 4 BN`, K_max=8 random scrambles, MPS device. Wall: ~39 min @ 235 ms/step. Goal was infrastructure-shake-out, not champion training.

## Observations (mechanical, from this run's data)

1. **Loss decreased monotonically through ~step 1000**, then stabilized in the 0.005–0.025 band for the remaining 9,000 steps. No NaN, no MPS crashes, no asserts. The training loop runs end-to-end on 3x3.
2. **`macro_v_star_mae` (corrected, V*=1..6 mean) dropped 3.48 → 0.73 over steps 500–7,500** (~5x improvement). The descent was clean through step 5,000 and noisy thereafter. The trajectory was **not flat from step 0** (acceptance gate (3) met) and **not exploding** (acceptance gate also met).
3. **Best corrected macro_v_star_mae was at step 7,500 (0.7340).** The final 2,500 steps were essentially flat: corrected values 0.749 / 0.773 / 0.758 / 0.787 / 0.749 — bouncing within ±0.05 of the best. **There is no real late-training regression.**
4. **The recorded macro_v_star_mae values reported via the metric pipeline DID show an apparent regression at step 10,000 (0.88 vs 0.73 best).** Investigation (see H1 below) found this was a calibration artifact from a bug in `value_eval`: V*=0 (the solved state) was included in the macro mean when walks happened to return to solved. Fixed in the same commit as this writeup.
5. **Early-stop did not fire.** Patience window is 12 evals × 500 steps = 6,000 steps from the best-eval index. Best at step 7,500 → patience would end at step 13,500, past the 10k cap. With the corrected (flat) trajectory, even a tighter patience would have stayed armed past 10k since each eval was within `min_delta=0.001` of best for fewer than 12 consecutive evals.
6. **Predictions climbed from ~0 → ~4.4 mean** over training. `pred_std` climbed 0.31 → 1.69. The eval set is a random-walk distribution over depths 1..14 with mean true-depth biased toward the deeper bins (walk redundancy notwithstanding); a `pred_mean` of ~4.4 with std ~1.7 means the network is predicting in roughly the right ballpark but slightly under-spread vs the underlying V* distribution.
7. **Per-V\*-layer MAE on the bounded oracle (d=1..6)** is roughly stable from step 7,500 to step 10,000:

   | V\* | step 7,500 MAE | step 10,000 MAE | Δ |
   |---|---|---|---|
   | 1 | 0.041 | 0.022 | -0.019 |
   | 2 | 0.182 | 0.217 | +0.035 |
   | 3 | 0.435 | 0.505 | +0.070 |
   | 4 | 0.883 | 0.891 | +0.008 |
   | 5 | 1.193 | 1.225 | +0.033 |
   | 6 | 1.670 | 1.633 | -0.037 |
   | macro | **0.734** | **0.749** | **+0.015** |

   Roughly linear in V*; the deepest oracle layer (d=6) is the noisiest. The per-layer differences between the two checkpoints are within eval-set noise.

8. **Beam capability at step 7,500 vs step 10,000 (beam_width=128, n_per_depth=50, n_per_layer=100):**

   On bounded-oracle V\* layers (d=1..6): **both checkpoints solve 100% with optimal solution lengths.** The value head leads beam to V\*-equal solutions even at d=6.

   On random-walk depths beyond the oracle (d=10..14):

   | walk depth | step 7,500 solve_rate | step 10,000 solve_rate | Δ |
   |---|---|---|---|
   | 10 | 0.98 | 1.00 | +2pp |
   | 11 | 0.88 | 0.92 | +4pp |
   | 12 | 0.64 | 0.80 | **+16pp** |
   | 13 | 0.46 | 0.50 | +4pp |
   | 14 | 0.18 | 0.28 | **+10pp** |

   **Step 10,000 is meaningfully better at the deep walk tail.** The training continued to push capability forward past step 7,500 even as macro_v_star_mae stayed flat in the bulk.

## Hypotheses

### H1: The apparent late-training regression in `macro_v_star_mae` was a measurement artifact, not a real value-function regression. CONFIRMED.

**Confidence:** high (after investigation).

**Evidence:** corrected macro values (V\*=1..6 mean, V\*=0 excluded) at the affected evals show **almost no regression**: step 10,000 corrected = 0.749 vs step 7,500's 0.734 (+0.015, within eval-set noise). The bug: `value_eval` averaged across all populated V\* layers including V\*=0 (the solved state). Walks of length ≥ ~6 occasionally return to solved by random-walk redundancy; when they do, V\*=0 populates with MAE ~1.7 (the network's prediction at solved, vs the true value 0). When V\*=0 wasn't present (most evals), macro was the d=1..6 mean; when V\*=0 was present (3 of 20 evals here, including step 10,000), macro was the d=0..6 mean. **The headline scalar was inconsistent eval-to-eval.**

Beam capability cross-check **falsifies the original H1** (cycle-4-style regression): step 10,000 IMPROVES beam solve rate at deep walk depths (+16pp at d=12, +10pp at d=14). The network kept learning at the tail past step 7,500.

**Fix landed in this commit:** `experiments/davi-3x3/eval.py` now excludes V\*=0 from the macro. V\*=0 is a different quality (terminal-value calibration) from "V\* prediction error on non-trivial scrambled states." Test added.

### H2: With the corrected metric, training has not converged at 10k steps — the trajectory shows continued improvement at the tail.

**Confidence:** medium-high.

**Evidence:** beam capability at d=12, d=14 still climbing at step 10,000. Per-V\* MAE bounded-oracle subset (d=1..6) is stable but the **bulk of the network's predictions live above d=6** (pred_mean=4.4, pred_std=1.7). The eval surface there (per_walk_depth/dN/pred_*) shows non-trivial spread that's still narrowing. Training looks like it could still benefit from more steps.

**Verification plan:** longer run (e.g. 20k–30k steps) at the same config to see whether deep-walk solve rates keep climbing or plateau. Or a tighter early-stop patience to test whether 6-eval patience would have stopped meaningfully earlier (probably yes — corrected macro had ≤2 consecutive within-min_delta evals at any point past step 7,500).

### H3: 10k steps is enough to demonstrate learning on `[5120, 1024] × 4 BN`, but not enough to characterize convergence or bottleneck.

**Confidence:** high.

**Evidence:** the trajectory clearly hadn't converged — beam capability at d=12, d=14 is still climbing; per-V\*-layer MAE at d=4..6 is still > 0.8 (the network is biased low at the deep oracle layers); pred_std is still climbing. Either training isn't done or the chosen K_max=8 is too narrow for the model capacity.

**Verification plan:** longer run (30k+ steps) at the same config to see convergence shape. Or a K_max sweep (e.g. K_max ∈ {6, 8, 12, 16}) at fixed step budget to see how max_scramble_depth changes the trajectory. Either belongs in the next training cycle, not this smoke run.

### H4: `value_eval` should treat eval-set sampling as fixed-from-seed each call, not advanced-by-generator.

**Confidence:** medium.

**Evidence:** the per-V\* MAE values change between step 7,500 and step 10,000 (e.g. d=1: 0.041 → 0.022, d=3: 0.435 → 0.505). The net's outputs change with training, so MAE should change — but the **eval-set states themselves** should ideally be fixed across calls so the changes attribute purely to the net. Currently the run passes `eval_generator = torch.Generator().manual_seed(seed + 17)` which is advanced through `random_scrambles` each call, so the eval set differs each time.

**Verification plan:** modify `value_eval` (or its caller) to re-seed the generator at the start of each eval call. Compare a re-seeded run's macro trajectory vs the current advancing-generator run — if the corrected macro becomes monotonic-flat past step 7,500 (no bouncing), we've confirmed the bouncing was eval-set noise, not net dynamics. Belongs in the next 3x3 cycle's prep, not this smoke.

## Open questions

- **Q1:** does beam capability at the deep walk tail (d=12, d=14) keep improving past step 10,000? Answers H2 and H3.
- **Q2:** what does the per-V\*-layer MAE shape look like at convergence (assuming we get there)? Ideally roughly flat across V\* (the network treats all depths equally well). Currently roughly linear-in-d — meaning deeper layers are systematically harder, not just noisier.
- **Q3:** does the K_max=8 ceiling cap learning at the deep walk tail, or is the tail capability bottlenecked by net capacity / training duration / something else?
- **Q4:** with corrected macro_v_star_mae and a fixed-from-seed eval set, what's the smallest sensible early-stop patience? Currently 12 evals; possibly excessive given a less noisy headline scalar.

## What we haven't verified

- Whether the corrected trajectory shape (flat past step 7,500) is reproducible at other seeds.
- Whether `[5120, 1024] × 4 BN` is the right capacity at all. Per CLAUDE.md "earn every hyperparameter," T0 capacity sweep is deferred to backlog — this run's results don't tell us whether the arch is over- or under-sized.
- Whether the eval-generator-advancing pattern (H4) is producing significant noise, or whether it's actually fine in practice.
- Whether the bug fix (V\*=0 exclusion) interacts with anything in the early-stop monitor besides the macro scalar — it shouldn't, but a fresh run with the fix would confirm.
