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

---

### Cycle 2 (2026-05-05): sync rate is not the lever — plateau is structural

Three follow-up runs, each warm-started from `baseline-30k/net_final.pt`
(weights-only resume; Adam moments restart; target_net synced from net
at init). Same architecture, same batch, same K_max, same LR, same
sampler. The only varied axis: `target_sync_interval`.

**Run summary (eval at step 7000, except sync100 which was killed at
step ~13800 with last full eval at 13000):**

| run | sync | macro_mae | pred_std | d7 | d11 | verdict |
|---|---:|---:|---:|---:|---:|---|
| baseline-30k (origin) | 500 | 3.13 | 1.44 | 0.78 | 0.40 | plateau |
| sync100-13k (killed) | 100 | 3.83 | 0.96 | 0.50 | 0.12 | regressed |
| sync500-7k (control) | 500 | 3.16 | 1.36 | 0.84 | 0.28 | flat |
| sync2000-7k | 2000 | 3.30 | 1.33 | 0.78 | 0.26 | flat / mild drift |

**Per-step macro_mae trajectories (warm-start runs, every 1000 steps):**

```
step    1k    2k    3k    4k    5k    6k    7k
sync100 3.16  3.13  3.48  3.50  3.55  3.54  3.72   monotone regression
sync500 3.16  3.27  3.31  3.21  3.23  3.22  3.16   flat — round trip
sync2000 3.14 3.14  3.18  3.17  3.22  3.21  3.30   flat with mild drift
```

#### Key observations

1. **Sync rate is not the lever.** Faster sync (100) is destabilizing.
   Same sync (500) holds steady. Slower sync (2000) is essentially
   identical to same sync. None of the three advance the wavefront past
   depth ~5. The hypothesis from cycle 1 — "sync_interval=500 is too
   long, the inner-window optimizer converges fast and the gradient
   signal drops" — is falsified across the bracket.

2. **The plateau is genuine, not "needed more steps."** sync500-7k is
   the cleanest possible control: identical config to baseline-30k,
   warm-started from its converged state. After 7k more steps,
   macro_mae returns to 3.16 (within noise of the warm-start origin
   3.13), pred_std oscillates 1.30–1.50, deep-depth solve rates wobble
   within sample variance. The model has converged to a stable
   fixed-point distribution where additional optimization at the same
   config produces no new structure.

3. **sync100 actively regresses, doesn't just plateau.** macro_mae
   drifted 3.13 → 3.83 over 13k steps; pred_std collapsed
   1.44 → 0.96 (predictions compressing back toward the mean);
   solve rates fell across the board. Faster sync from a *converged*
   warm-start state destabilizes — V_target chases V_θ before V_θ has
   stabilized, V_θ then overshoots back toward V_target's new
   position, the inner-window gradient is noise-dominated, dynamic
   range collapses. (Confounded with the Adam-fresh handoff at warm-start
   init, but the regression continues monotonically across 13k steps —
   well past where Adam moments would have re-warmed up.)

4. **GPU saturation answer (from macmon probe during sync100-13k at
   step 5000):** B=4096 sustains **96% GPU busy / 47W / 76°C**. M4's
   `apply_moves` saturation onset was located between B=4096 and
   B=32768. We're already past the knee. **Wider batch is not a
   throughput lever** — at most ~4% headroom, likely less in practice.

#### Why the plateau is structural

The wavefront stalls because of a self-reinforcing fixed-point in the
data distribution × bootstrap dynamics:

- Random walks of depth ∈ [1, K_max=18] generate the training batch.
  The state distribution is heavily skewed toward deeper (BFS frontier
  grows ~7× per depth on the 2x2; depths 11–14 dominate the corpus).
  Deep-depth states get most of the gradient updates per batch.
- For a state at true depth 11, the Bellman target is
  `min_a (1 + V_target(child))`. The child is at true depth 10, 12, or
  back-tracking. V_target(child) is whatever V_θ predicts there — and
  V_θ at depth 11 is currently ~5. So the target for a depth-11 state
  is ~6. Optimizer moves V_θ(depth-11) toward 6.
- Same logic for depth-10 children: target is ~6 from V_θ(depth-9)
  which is ~5 → target 6, V_θ(depth-10) moves toward 6.
- Result: V_θ(d) → ~5 for all d ≥ 5, regardless of true depth. The
  net is *self-consistently wrong* in the bulk. Train loss is low
  (V_θ ≈ V_target ≈ 1 + V_target(child) on the optimal child path
  — locally consistent) but macro-MAE is flat at ~3 (globally wrong).
- Sync rate doesn't break this fixed-point — only the data
  distribution or the curriculum can. A faster sync just chases the
  same fixed-point faster; a slower sync leaves it equally stable.

#### Recommendations (ordered by predicted info gain)

The lever is on the **data side**, not the optimizer side.

**1. Audit the actual training-batch state distribution (cheap, ~10 min).**
What does `generate_adi_batch` actually produce? Are depths 1–14 evenly
sampled (per-depth slice in scrambles.py implies yes), or does the
random-walk-revisits-shallow effect collapse the effective depth
distribution? Instrument `generate_adi_batch` to log a depth histogram
on a few sample batches before designing the next training run. If
deep depths are under-represented in *trained* batches (vs the *target*
slicing), depth-balance sampling is the obvious next axis. If deep
depths are over-represented but the V_target signal there is junk,
curriculum is the obvious next axis.

**2. K_max=14 (warm-start, 7k steps, ~25 min).** Trim training distribution
to ≤ QTM diameter. States past depth 14 don't exist (the cube has 14
QTM diameter; depth-15 random walks visit some depth ≤ 14 state with a
detour). Currently 22% of each batch is at K_max ∈ [15, 18] and
bootstraps through inflated shallow V_target. Cleaning this up tests
whether the past-diameter pollution matters — H2 from cycle 1, never
verified. Falsifiable in either direction.

**3. Curriculum scheduling (fresh-start, ~2 hours).** Start K_max=2
for first 5k steps (only depths 1–2; net learns V*(d=1)=1 and
V*(d=2)=2 accurately), then ramp K_max=4, 8, 14 in stages. This
follows DAVI dynamics — V_target(d=1) needs to be accurate before
V_θ(d=2) can converge to 2. Standard ADI / DeepCubeA technique
(generic, not borrowed numbers — the schedule is the question to
explore). Costs a fresh run, but the lessons from cycles 1+2 about
"sync rate doesn't matter" carry forward.

**4. Network size: not the next move.** V*-supervised work in cycle 0
showed (1024, 256) n=2 = ~2M params learning V* to macro_mae ~0.85
under depth-balanced supervised regression. Capacity isn't the
bottleneck — bootstrap dynamics are. Defer size sweep until the
plateau breaks (then re-examine: maybe smaller works fine, maybe
bigger gives faster convergence; either way, only meaningful once
we know we *can* learn past depth 5).

**5. More-full-training-from-scratch: not yet.** Running 100k steps
of a config that doesn't learn past depth 5 burns compute without
information gain. Must first find a config that breaks the plateau,
then long-train.

#### Direct answers to the asked questions

- **"Should we try different sized networks?"** Not first. The
  V*-supervised work already located the capacity floor; we're well
  above it. The bottleneck is bootstrap fixed-point, not capacity.
  Revisit size after a config beats the plateau.

- **"Should we do more full training?"** Not at this config. The
  sync500 control run is direct evidence — 7k extra steps from
  converged state produces zero learning. 30k more would be 30k more
  of the same. Need to change something other than step count first.
