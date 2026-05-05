# davi-baseline — 30k DAVI run results

_Run dir: `experiments/davi-2x2/davi-baseline/runs/baseline-30k`. Records: 339 total, 31 eval cycles._

## Train loss

- start: 0.1632
- end:   0.0234
- min:   0.0040 (step 1900)

## V* macro-MAE trajectory

| step | train_loss_recent | macro_mae | val_mae | pred_mean | pred_std |
|-----:|-----:|-----:|-----:|-----:|-----:|
| 1000 | — | 7.1037 | 7.1037 | 0.396 | 0.189 |
| 2000 | — | 5.5657 | 5.5657 | 1.973 | 0.283 |
| 3000 | — | 4.2704 | 4.2704 | 3.410 | 0.823 |
| 4000 | — | 3.6476 | 3.6476 | 4.052 | 1.290 |
| 5000 | — | 3.5256 | 3.5256 | 4.198 | 1.287 |
| 6000 | — | 3.4456 | 3.4456 | 4.326 | 1.325 |
| 7000 | — | 3.2982 | 3.2982 | 4.449 | 1.405 |
| 8000 | — | 3.2600 | 3.2600 | 4.494 | 1.436 |
| 9000 | — | 3.1166 | 3.1166 | 4.655 | 1.507 |
| 10000 | — | 3.1046 | 3.1046 | 4.673 | 1.540 |
| 11000 | — | 3.1913 | 3.1913 | 4.568 | 1.494 |
| 12000 | — | 3.2358 | 3.2358 | 4.561 | 1.422 |
| 13000 | — | 3.2367 | 3.2367 | 4.556 | 1.457 |
| 14000 | — | 3.2514 | 3.2514 | 4.563 | 1.376 |
| 15000 | — | 3.2590 | 3.2590 | 4.599 | 1.357 |
| 16000 | — | 3.1764 | 3.1764 | 4.614 | 1.454 |
| 17000 | — | 3.2712 | 3.2712 | 4.579 | 1.356 |
| 18000 | — | 3.2326 | 3.2326 | 4.520 | 1.429 |
| 19000 | — | 3.2619 | 3.2619 | 4.465 | 1.397 |
| 20000 | — | 3.2778 | 3.2778 | 4.543 | 1.332 |
| 21000 | — | 3.1078 | 3.1078 | 4.758 | 1.396 |
| 22000 | — | 3.1085 | 3.1085 | 4.777 | 1.400 |
| 23000 | — | 3.0704 | 3.0704 | 4.798 | 1.433 |
| 24000 | — | 3.1130 | 3.1130 | 4.757 | 1.401 |
| 25000 | — | 3.0921 | 3.0921 | 4.758 | 1.418 |
| 26000 | — | 3.1682 | 3.1682 | 4.692 | 1.361 |
| 27000 | — | 3.1224 | 3.1224 | 4.752 | 1.368 |
| 28000 | — | 3.1588 | 3.1588 | 4.736 | 1.349 |
| 29000 | — | 3.0900 | 3.0900 | 4.789 | 1.416 |
| 30000 | — | 3.1315 | 3.1315 | 4.731 | 1.440 |
| 30000 | — | 3.1315 | 3.1315 | 4.731 | 1.440 |

## Per-depth MAE — start / middle / end

| depth | start (step 1000) | middle (step 16000) | end (step 30000) |
|------:|------:|------:|------:|
| 1 | 0.866 | 1.035 | 1.162 |
| 2 | 1.712 | 0.620 | 0.872 |
| 3 | 2.664 | 0.428 | 0.648 |
| 4 | 3.571 | 0.672 | 0.586 |
| 5 | 4.559 | 0.889 | 0.684 |
| 6 | 5.573 | 1.243 | 1.118 |
| 7 | 6.541 | 1.875 | 1.766 |
| 8 | 7.554 | 2.595 | 2.440 |
| 9 | 8.557 | 3.454 | 3.337 |
| 10 | 9.558 | 4.273 | 4.170 |
| 11 | 10.574 | 5.153 | 5.063 |
| 12 | 11.566 | 6.223 | 6.080 |
| 13 | 12.581 | 7.066 | 7.024 |
| 14 | 13.576 | 8.944 | 8.890 |

## Greedy-policy solve rate trajectory

| step | d1 rate | d3 rate | d5 rate | d7 rate | d9 rate | d11 rate | d13 rate | d1 avg_len | d3 avg_len | d5 avg_len | d7 avg_len | d9 avg_len | d11 avg_len | d13 avg_len |
|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 1000 | 1.00 | 0.20 | 0.06 | 0.02 | 0.00 | 0.02 | 0.00 | 1.00 | 2.60 | 3.67 | 3.00 | — | 1.00 | — |
| 2000 | 1.00 | 0.98 | 0.84 | 0.50 | 0.36 | 0.18 | 0.02 | 1.00 | 3.24 | 5.14 | 6.36 | 6.22 | 7.67 | 7.00 |
| 3000 | 1.00 | 1.00 | 0.84 | 0.60 | 0.40 | 0.30 | 0.16 | 1.00 | 2.96 | 4.67 | 5.20 | 6.00 | 6.47 | 7.00 |
| 4000 | 1.00 | 1.00 | 1.00 | 0.72 | 0.40 | 0.42 | 0.18 | 1.00 | 3.00 | 4.76 | 5.56 | 6.60 | 6.62 | 6.56 |
| 5000 | 1.00 | 1.00 | 1.00 | 0.80 | 0.48 | 0.46 | 0.28 | 1.00 | 2.96 | 4.64 | 6.05 | 6.83 | 6.65 | 7.86 |
| 6000 | 1.00 | 1.00 | 0.96 | 0.80 | 0.66 | 0.40 | 0.40 | 1.00 | 2.96 | 4.71 | 6.00 | 6.27 | 7.00 | 7.30 |
| 7000 | 1.00 | 1.00 | 1.00 | 0.92 | 0.60 | 0.38 | 0.30 | 1.00 | 3.00 | 4.60 | 5.87 | 6.80 | 7.32 | 7.53 |
| 8000 | 1.00 | 0.98 | 0.92 | 0.80 | 0.60 | 0.38 | 0.32 | 1.00 | 2.96 | 4.74 | 6.00 | 6.87 | 7.63 | 8.12 |
| 9000 | 1.00 | 1.00 | 0.88 | 0.82 | 0.58 | 0.42 | 0.30 | 1.00 | 3.08 | 4.68 | 5.73 | 6.59 | 6.81 | 6.87 |
| 10000 | 1.00 | 1.00 | 0.98 | 0.88 | 0.50 | 0.44 | 0.44 | 1.00 | 2.96 | 4.43 | 5.95 | 6.68 | 7.18 | 7.27 |
| 11000 | 1.00 | 1.00 | 1.00 | 0.90 | 0.58 | 0.28 | 0.22 | 1.00 | 2.92 | 4.60 | 5.67 | 6.31 | 7.00 | 6.82 |
| 12000 | 1.00 | 1.00 | 1.00 | 0.88 | 0.68 | 0.42 | 0.40 | 1.00 | 2.92 | 4.56 | 5.55 | 6.35 | 7.00 | 7.10 |
| 13000 | 1.00 | 1.00 | 1.00 | 0.84 | 0.56 | 0.42 | 0.28 | 1.00 | 3.00 | 4.80 | 5.76 | 6.00 | 6.90 | 8.29 |
| 14000 | 1.00 | 1.00 | 0.96 | 0.80 | 0.52 | 0.26 | 0.26 | 1.00 | 2.88 | 4.38 | 5.80 | 6.69 | 7.15 | 7.31 |
| 15000 | 1.00 | 1.00 | 1.00 | 0.78 | 0.56 | 0.40 | 0.20 | 1.00 | 2.96 | 4.72 | 5.72 | 6.50 | 6.70 | 6.80 |
| 16000 | 1.00 | 1.00 | 1.00 | 0.80 | 0.50 | 0.30 | 0.32 | 1.00 | 2.96 | 4.76 | 5.60 | 5.48 | 6.07 | 6.50 |
| 17000 | 1.00 | 1.00 | 0.94 | 0.74 | 0.60 | 0.46 | 0.20 | 1.00 | 2.96 | 4.70 | 5.65 | 6.33 | 7.35 | 6.00 |
| 18000 | 1.00 | 1.00 | 0.96 | 0.82 | 0.74 | 0.34 | 0.18 | 1.00 | 3.00 | 4.79 | 6.32 | 5.97 | 7.00 | 7.00 |
| 19000 | 1.00 | 1.00 | 0.92 | 0.82 | 0.62 | 0.24 | 0.24 | 1.00 | 3.00 | 4.65 | 5.93 | 6.35 | 6.17 | 7.50 |
| 20000 | 1.00 | 1.00 | 1.00 | 0.82 | 0.46 | 0.34 | 0.20 | 1.00 | 2.96 | 4.44 | 5.78 | 6.04 | 7.00 | 6.80 |
| 21000 | 1.00 | 1.00 | 0.96 | 0.90 | 0.52 | 0.30 | 0.30 | 1.00 | 2.88 | 4.71 | 5.84 | 6.00 | 6.73 | 7.00 |
| 22000 | 1.00 | 1.00 | 1.00 | 0.74 | 0.54 | 0.38 | 0.32 | 1.00 | 2.96 | 4.44 | 5.97 | 6.78 | 6.58 | 6.62 |
| 23000 | 1.00 | 1.00 | 1.00 | 0.90 | 0.66 | 0.32 | 0.20 | 1.00 | 2.92 | 4.52 | 5.80 | 6.58 | 6.50 | 7.40 |
| 24000 | 1.00 | 1.00 | 0.96 | 0.90 | 0.60 | 0.32 | 0.26 | 1.00 | 2.92 | 4.50 | 5.71 | 6.47 | 6.75 | 7.00 |
| 25000 | 1.00 | 1.00 | 1.00 | 0.86 | 0.54 | 0.30 | 0.24 | 1.00 | 3.00 | 4.56 | 5.84 | 7.00 | 6.87 | 7.33 |
| 26000 | 1.00 | 1.00 | 1.00 | 0.70 | 0.52 | 0.40 | 0.26 | 1.00 | 2.92 | 4.72 | 5.51 | 6.38 | 6.80 | 7.92 |
| 27000 | 1.00 | 1.00 | 1.00 | 0.88 | 0.58 | 0.44 | 0.24 | 1.00 | 2.92 | 4.60 | 5.68 | 6.52 | 6.82 | 6.83 |
| 28000 | 1.00 | 1.00 | 0.98 | 0.80 | 0.54 | 0.46 | 0.22 | 1.00 | 2.96 | 4.59 | 5.30 | 6.26 | 6.65 | 7.00 |
| 29000 | 1.00 | 1.00 | 1.00 | 0.88 | 0.50 | 0.38 | 0.26 | 1.00 | 2.96 | 4.84 | 5.86 | 6.60 | 6.79 | 7.31 |
| 30000 | 1.00 | 1.00 | 1.00 | 0.86 | 0.52 | 0.42 | 0.26 | 1.00 | 3.00 | 4.56 | 6.02 | 6.54 | 7.10 | 6.54 |
| 30000 | 1.00 | 1.00 | 1.00 | 0.78 | 0.74 | 0.40 | 0.20 | 1.00 | 2.92 | 4.52 | 6.08 | 6.46 | 7.10 | 7.20 |


---

## Intuition

_Hand-written 2026-05-04 immediately after the run completed. Format:
Observations (mechanical) → Hypotheses (with evidence + verification
plan) → Open questions._

### Observations

1. **Wavefront propagated to ~depth 5, then stalled.** Final per-depth
   MAE (from `## Per-depth MAE` table): 1.16 / 0.87 / 0.65 / 0.59 /
   0.68 / 1.12 / 1.77 / 2.44 / 3.34 / 4.17 / 5.06 / 6.08 / 7.02 / 8.89
   for depths 1..14. Depths 1–5 well-learned (sub-1.2 MAE);
   depths 6–14 degrade linearly with depth at slope ~0.85
   per-depth — consistent with V_θ saturating around a constant value
   in the bulk, so true depth d gives MAE ≈ |d − constant|.

2. **Constant is ~5.** Pred_mean stabilized at 4.73 by step 30000;
   pred_std at 1.44. The model has *not* learned to span the full V*
   range [1, 14]. Most predictions sit in [3.3, 6.2].

3. **Train loss flat-lined early.** Loss start 0.16, end 0.02, min
   0.004 at step 1900. The DAVI optimization converged to a local
   fit-of-moving-targets long before macro-MAE bottomed out — confirming
   that train loss does not measure V*-fit, it measures "predict what
   V_target predicts +1 on the min child."

4. **Macro-MAE descent rate trajectory:** 1.53 / 1.30 / 0.62 / 0.12 /
   0.08 / 0.15 / 0.04 / 0.14 / 0.02 / -0.09 (per 1000 steps from
   step 1k onward). After step ~5000 macro-MAE oscillates in
   [3.07, 3.27] with no monotone trend. Plateau is real, not "still
   converging slowly."

5. **Wavefront extent reflected in greedy solve rates** at final
   eval: d1=1.00, d3=1.00, d5=1.00, d7=0.78, d9=0.74, d11=0.40,
   d13=0.20. Cleanly solvable up to ~d5, partially solvable d7–d9,
   degrading past that. Greedy solve drops where V_θ's relative
   ordering of children stops being reliable.

6. **Step time stable** at 218 ms throughout. No degradation, no
   memory issue. M4 Max at this network size handled the run fine
   (~109 minutes total wall).

7. **pred_std plateau at ~1.4** despite target std being ~4.0 (V*
   uniform over depths 1..14 has std ~4). The model has compressed
   its dynamic range — same shape as the V*-supervised collapse
   patterns, but different mechanism (here it's bootstrap-stalled,
   not loss-distribution-collapsed).

### Hypotheses

**H1 — Target sync interval is too long for deep-tail wavefront
propagation. (Medium-high confidence.)**

Evidence: train loss converges fast (~step 2000) inside each
sync-window — V_θ catches up to V_target quickly. After convergence,
gradient signal for the bulk drops to noise (target = V_target ≈
V_θ ⇒ MSE ≈ 0). With sync interval = 500, V_target gets ~60 update
events over 30k steps. For wavefront to reach depth 14, each of those
60 updates needs to push the V_target boundary outward by ~0.25 depth
on average. Empirically the wavefront made it to ~5 — that's
~5/60 = 0.08 depth per sync. The propagation rate is much slower than
the hypothetical "one depth per sync" because each sync only nudges
V_target slightly given the sync-interval-internal optimizer dynamics.

Verification plan: rerun with `target_sync_interval = 100` (5× more
syncs in same 30k steps). If wavefront reaches deeper (e.g. macro-MAE
plateaus at ~1.5 instead of ~3.1, or pred_std exceeds 2.0), H1
confirmed. If similar plateau → H1 wrong, deeper issue.

**H2 — K_max=18 dilutes per-depth gradient signal in batch-balanced
slicing. (Medium confidence.)**

Evidence: at K_max=18 with batch_size=4096, balanced slicing gives
~227 states per depth bucket per batch. Depths 15–18 are not in V*
(2x2 QTM diameter is 14) — they're real states whose true V* ≤ 14
but whose scramble-walk depth went past optimality. Their Bellman
target via V_target on their children is bootstrapped through the
same shallow V_target that's also unreliable on depths 11–14. Roughly
4/18 ≈ 22% of each batch carries gradient signal that pollutes V_θ
with the V_target's own self-consistent-but-wrong shallow
predictions.

Verification plan: rerun with `max_scramble_depth = 14` (exactly the
QTM diameter). Removes the 4 surplus depth buckets. If wavefront
propagates further, H2 contributes meaningfully. Crucially, this is a
distinct experiment from H1 — best to vary one axis at a time so
contributions are attributable.

**H3 — Network is over-parameterized, gradient signal at depth-d
predictions is dominated by its own batch's bulk modal predictions.
(Low-medium confidence.)**

Evidence: 17M params on a 3.6M-state 2x2 universe; not classic
underfit-friendly capacity ratio, though regression loss landscapes
care less about this than classification. The pred_std plateau at ~1.4
is suspicious — V_θ is happy to predict ~5 across all input states
when the true range is [0, 14]. A smaller net might be forced to
generalize better.

Verification plan: rerun at `(2048, 512) n=2` (~3M params) and check
whether pred_std reaches a higher fraction of V* std. Confounded by
H1+H2 unless those are held fixed.

**H4 — Greedy solve rates show V_θ's relative-order signal is more
robust than its absolute-MAE signal. (Medium confidence.)**

Evidence: at depth 9, MAE = 3.34 (V_θ predicts ~5.7 for true=9), yet
greedy solve = 0.74. At depth 11, MAE = 5.06 (V_θ predicts ~5.9 for
true=11), greedy solve = 0.40. The model isn't getting absolute V*
right but is partially preserving the inequality V(s_a*) < V(s_a≠*).
Implication for downstream: at this plateau, the model is *useful*
for greedy in shallow-to-medium scrambles even though it fails the
strict acceptance gate. If beam search (M6) is forgiving of absolute
MAE, this config might still be a reasonable beam-search input.

Verification plan: not strictly needed before next experiment, but
worth re-running greedy solve with `n_per_depth=200` (instead of 50)
to tighten the noise on solve rates. Currently each rate is
σ ≈ 0.07 from binomial. With 200 we'd get σ ≈ 0.035.

### Open questions

1. **What's the steady-state plateau as a function of target sync
   interval?** H1's verification (sync=100) is the cleanest single
   next experiment. Cost: same 30k steps × 218 ms = ~109 min.

2. **Does removing K_max=18 surplus help?** H2's verification
   (K_max=14) is the cleanest single next experiment. Cost: ~85 min
   (less per-step batch-gen work due to fewer depth buckets and
   shorter walks).

3. **Curriculum?** The current setup samples uniformly across depths
   1–18 from step 1. An alternative: ramp K_max from 1 to 18 over
   training. Would the wavefront propagate cleaner that way? This is
   a methodology-shape question, not a single hparam — best deferred
   until after H1 + H2 are settled.

4. **Loss formulation?** MSE here vs. Huber / L1 / weighted-by-depth.
   The V*-supervised work (archived) ruled out loss formulation as
   the dominant axis under that regime. Whether DAVI is similar is
   open — but no evidence yet that loss is the leading suspect, so
   defer.

5. **Should pred_std be a logged ceiling?** Right now we only see it
   in the eval record. Worth tracking pred_std-over-target_std ratio
   as a "dynamic range" metric. If a config has pred_std/target_std
   close to 1.0 *and* macro-MAE high, that's a different failure
   than this run (compressed range → mean collapse).

### What we haven't verified

- That step time at sync_interval=100 is the same as 500. The sync
  itself is just a state_dict copy; should be cheap, but 5× more of
  them in 30k steps is worth checking.
- That max_scramble_depth=14 doesn't break the depth-stratified eval
  set (which has depths 0..14). Quick re-check of `eval_set.npz`'s
  depth distribution before re-running.
- That the network's BN running stats are not the actual cause of
  the pred_mean shift (V*-supervised work suggested they weren't,
  but DAVI dynamics differ — quick eval-mode-during-training check
  could rule it out cheaply).

### Verdict

**This config falsifies the "first try acceptance" hypothesis** —
macro_mae 3.13 vs. gate < 1.0; greedy d11=0.40, d13=0.20 vs. gate
> 0.99. Failure shape is *informative-fail*, not catastrophic: the
DAVI loop runs end-to-end, the wavefront propagates from solved
outward, target sync drives extension, but the wavefront stalls at
~depth 5. The leading suspect is target_sync_interval (H1) — that's
the single next experiment to run.

This run is preserved as the baseline against which to compare H1
(sync=100) and H2 (K_max=14).
