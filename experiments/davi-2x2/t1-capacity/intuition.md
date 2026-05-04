# t1-capacity intuition

*Datestamp:* 2026-05-04. *Run conditions:* Phase A on `m5-davi` branch at commit
`50c9d5e` (corrected two-phase methodology). M4 Max MPS, torch 2.11.

## Why we ran this

Tier 1 asks a yes/no capacity question: **what's the smallest network that
can fit V\* on the 2x2 to val-MAE < 0.5 by direct supervised regression?**
Decoupling capacity from training dynamics is leverage — if we know the
capacity floor in isolation, then any later DAVI failure with that
network is a *training-dynamics* problem (LR, target sync, curriculum),
not a *capacity* problem.

The methodology specified two phases: Phase A walks down widths at
`n_residual_blocks=0` to find `(h1*, h2*)`, then Phase B sweeps residual
counts at the chosen widths.

## What we expected

- A clean Pareto frontier on the (n_params, val_mae) plane: the
  `(4096, 1024)` anchor would clearly succeed; failure would emerge at
  some smaller cell as we walked down.
- Phase B would refine: maybe residuals would buy a small gain at the
  chosen widths, maybe not.
- T1 would close in ~30 minutes with a `(h1*, h2*, n*)` tuple in
  `_picks.json`, ready for T2 to start asking the LR question on the
  T1-comfortable network.

## Observations *(mechanical, from the runs)*

**Phase A — all 5 cells failed identically.** Every cell, from the
207K-param `(512, 256)` floor up to the 4.80M-param `(4096, 1024)`
anchor — a **23× param scale-up** — converges to **val_mae ≈ 0.90**.
The predict-the-mean baseline on V\* is 0.9286. So every cell ends up
within 0.03 absolute of the no-information baseline, and `n_params`
explains essentially none of the variance in final val_mae (range
0.8979–0.9037 across 23× scaling).

| cell | n_params | final val_mae |
|------|---------:|--------------:|
| 512 × 256 | 207K | 0.9017 |
| 1024 × 512 | 677K | 0.9037 |
| 2048 × 512 | 1.35M | 0.9023 |
| 2048 × 1024 | 2.40M | 0.8998 |
| 4096 × 1024 | 4.80M | 0.8979 |

**Train loss does decrease** — from ~1.43 at step 1k to ~1.22 at step 7k
across all cells. But that's the *same* train-loss decrease across all
five cells, suggesting the optimization is going somewhere identical
regardless of capacity.

**Diagnostic 1** — `(4096, 1024)` n=0, **30k steps + LR 3e-3**: val_mae
0.85 (improvement, but plateau still 0.7 above target). Loss continues to
drop slowly toward step 30k.

**Diagnostic 2** — `(1024, 512)` **n=2**, 7k steps, LR 1e-3: val_mae 0.87.
Adding 2 residual blocks at this width buys ~0.04 absolute over the n=0
counterpart.

**Diagnostic 3** — `(2048, 512)` **n=4, 30k steps, LR 3e-3**: val_mae
0.80. Combining bigger network + more residuals + more steps + higher LR
gets us further but still nowhere near 0.5.

**Diagnostic 3 prediction-distribution probe.** On 5000 random states,
the n=4 30k-step model's predictions ordered correctly by depth but
were systematically compressed:

| true depth | predicted mean | predicted std | per-depth MAE |
|-----------:|---------------:|--------------:|--------------:|
| 6 | 6.99 | 0.70 | 0.99 |
| 7 | 8.07 | 0.91 | 1.13 |
| 8 | 9.11 | 0.93 | 1.21 |
| 9 | 9.92 | 0.83 | 1.06 |
| 10 | 10.53 | 0.64 | 0.72 |
| 11 | 10.90 | 0.48 | 0.36 |
| 12 | 11.10 | 0.40 | 0.90 |
| 13 | 11.26 | 0.42 | 1.74 |

Predictions span 5.24–12.31 (std 0.79) against targets spanning 0–14 (std
1.16). The model is **systematically pulling all predictions toward the
mean** (10.67) — best at depth 11 (the modal class, 36.77% of states),
worst at the tails (depth 13 has 1.74 MAE — predictions cap around 11.3
when the truth is 13).

**Train and eval modes give nearly identical predictions.** On
diagnostic 1, eval-mode preds (mean 10.70, std 0.39) and train-mode preds
(mean 10.71, std 0.39) differ by < 0.02. So BatchNorm running statistics
**aren't drifted** vs batch statistics — that rules out one specific
class of BN bug.

## Hypotheses

*(These are **interpretive claims** with confidence levels. They are
**not** decided answers — the next-cycle work below verifies or rejects
them. Future agents should not anchor on these without running the
verification.)*

**H1 (high confidence): MSE loss × peaked target distribution × BN-MLP =
mean-collapse trap.** The depth distribution is severely peaked: 83% of
states at depths 10–12, only 0.01% at the tails (depths 0–4). Under MSE,
the gradient signal toward the tails is small (few samples, small
contribution to total loss); the signal toward the bulk is huge.
Combined with BatchNorm-everywhere (which bounds activation magnitudes
layer-by-layer) and a randomly-initialized head, the optimization
landscape near "predict the mean" is very flat — every direction from
mean-prediction looks roughly equal in loss. The model finds a slow
exit (per-depth ordering is correct, predictions span a wider range as
we train longer / scale up), but escape is **orders of magnitude slower
than the placeholder budget allowed**.

*Supporting evidence:* (a) all 5 Phase A cells converge to ~0.90 val_mae
regardless of capacity; (b) per-depth prediction ordering is correct
even in failed runs (model is not random); (c) prediction std is
compressed 0.39–0.79 vs target 1.16; (d) longer training + higher LR
slowly improves but doesn't break out of the regime; (e) the
predict-the-mean MAE baseline is 0.9286, and we're sitting 0.03 above
it.

*Verification plan:* run `(2048, 512)` n=2 with **L1 loss** instead of
MSE for 7k steps, fixed seed. L1 weights tail samples more equally with
bulk samples (the gradient is `sign(error)` not `error` itself). If
val_mae drops markedly (e.g., < 0.5), H1 is confirmed: MSE was the wrong
loss for this distribution. If it doesn't, H1 is weakened.

**H2 (medium confidence): BatchNorm-before-head compresses output range.**
The architecture has BN immediately before each ReLU through the body —
but the head is `Linear(h2, 1)` with no preceding normalization-undoing
step. If the body's final post-BN+ReLU activations are small (mean ~0,
std ~1 in pre-ReLU; ReLU clips negative half), the head must learn
weights that map this small-magnitude representation to a wide target
range (0–14). With Adam at LR 1e-3, the head gradient steps may be too
small relative to body steps to widen the output range fast.

*Supporting evidence:* eval-mode and train-mode predictions are
near-identical (BN running stats aren't drifted), but predictions are
still compressed toward the mean — suggesting the compression is not in
running stats but in the body's representation magnitude.

*Verification plan:* train an architecture variant without the BN before
the projection layer (or with LayerNorm replacing BN entirely), same
LR/steps. If val_mae drops dramatically, H2 has weight.

**H3 (low confidence): The shallow MLP without residuals lacks
inductive bias for cube algebraic structure.** V\*-to-state is a
function of corner positions and orientations — features that compose
non-linearly. A 2-layer MLP can in principle approximate any function
(universal approximation), but practically may need many more parameters
than we tested or much longer training to discover the right composition
without depth.

*Supporting evidence:* Diagnostic 2 (n=2 residuals at (1024, 512))
slightly improves on Phase A's matched cell (0.87 vs 0.90). Diagnostic 3
(n=4 at (2048, 512), 30k+3e-3) gets to 0.80. Adding depth seems to help.

*Verification plan:* contingent on H1/H2 — if those don't fix it, run a
deep-but-narrow architecture (e.g., n=8 residuals at (512, 512)) at
optimum LR for that shape and see if depth alone unlocks fitting.

## Open questions

These define the **debug cycle** that follows T1. Each is a single
focused experiment, not an axis of expansion:

1. **Does L1 loss break out of the mean-collapse plateau?** Single-cell
   test: `(2048, 512)` n=2, batch 1024, 7k steps, LR 1e-3, **MSE → L1**.
   This isolates the loss-formulation hypothesis from architecture.

2. **Does removing BN before the head fix output compression?** Same
   single cell, **drop BN on the projection layer** (keep BN after the
   first projection, drop it before head). Compare.

3. **Does an LR schedule with warmup change the picture?** Same cell,
   add 500-step linear warmup from 1e-5 to 1e-3, then constant. Does
   the model escape the mean-collapse plateau faster?

4. **Is there a numerical issue we missed?** Cheap sanity: train on a
   fixed 1k-state subset (small enough to memorize); does train-MSE go
   to ~0? If yes, machinery is sound and the issue is genuinely
   distributional. If no, there's a bug elsewhere.

5. **Should T1's framing change?** If H1+H2 are confirmed, the
   methodology's "T1 = capacity question" framing is the wrong frame
   for this problem at all. The bottleneck before capacity is
   *loss/normalization choice*. T1 may need to become a two-question
   tier: "what loss/normalization produces meaningful gradient signal
   on this distribution?" → "what's the capacity floor under those
   choices?"

## What we haven't verified

- The diagnostic prediction-distribution probe was on a single random
  5000-state sample. Numbers in the per-depth table could shift ~0.05
  with a different sample. The story holds, but specific values shouldn't
  be cited as "depth 13 MAE is exactly 1.74" — it's "around 1.7".
- The H1 hypothesis assumes MSE-with-peaked-distribution is the
  dominant cause. We haven't ruled out interactions with BN ordering,
  residual block initialization, or MPS-specific numerical quirks.
  Verification 4 (memorize 1k subset) is the cleanest sanity check
  before declaring the hypothesis sound.
- We did not test whether the W&B-style live training curves would
  reveal anything we missed in the JSONL post-hoc. That tooling is
  deferred per the methodology — but if the debug cycle drags, it may
  be worth pulling forward.

## Memorize-1k sanity check (2026-05-04)

Verification 4 from the open-questions list: train on a fixed 1k-state
depth-stratified subset, full-batch, MSE, same architecture/loss/optim
machinery as Phase A. Pass criterion: `train_mse <= 1e-3` within 10k
steps.

**Cell:** `(2048, 512)`, `n_residual_blocks=2`, batch=1000 (full-batch),
lr=1e-3, seed=0, subset_seed=7, MPS. 2.41M params.

**Observations.**

| step | train_mse | train_mae |
|-----:|----------:|----------:|
| 0 | 111.904091 | 10.4994 |
| 100 | 0.000233 | 0.01446 |
| 1000 | 0.000000 | 0.00000 |
| 5000 | 0.000000 | 0.00000 |
| 10000 | 1.2e-9 | 0.00003 |

**PASS at step 63** (`train_mse=9.78e-4`). Train MSE drives to numerical
zero by step 1000 and stays there. Wall time ~32s for 10k steps.

Courtesy val-MAE on a 2000-state held-out slice (excluded from the 1k
subset) hovers around 0.99–1.07 throughout — i.e., the model memorizes
the 1k subset perfectly while remaining ~at the predict-the-mean
baseline on unseen states. That's the expected memorization signature
(no mysterious generalization, no machinery weirdness).

**Verdict: machinery is clean.** Training loop, data path, model,
optimizer, MPS backend are all functioning as designed. Phase A's
~0.90 val_mae plateau is **not** a bug — it is the genuine behavior of
MSE + this depth distribution + this architecture on the full 2.94M
training set. H1 (MSE × peaked V\* depth distribution × BN-everywhere
mean-collapse trap) remains the leading hypothesis and the L1-loss
verification (open question 1) is the right next experiment.

**Open questions** unchanged — go run the L1 loss experiment next.

## L1-loss diagnostic (2026-05-04)

Open question 1 from above: does swapping MSE → L1 break out of the
~0.90 plateau? Two cells, matched-everything-else against
`phaseA_3_2048x512` and the cold-handoff n=2 cell:

| cell | loss | n_res | val_mae | pred_mean | pred_std |
|------|------|------:|--------:|----------:|---------:|
| `phaseA_3_2048x512` (MSE baseline) | mse | 0 | 0.8979 | (~10.66) | 0.79 (probe) |
| `l1_2048x512_n0` | l1 | 0 | **0.8819** | 10.82 | 0.19 |
| `l1_2048x512_n2` | l1 | 2 | **0.8667** | 10.86 | 0.25 |

**Constant-predictor baselines on V\*:**
- predict-the-mean MAE = 0.9286 (MSE-optimal constant)
- **predict-the-median MAE = 0.8583** (L1-optimal constant; median = 11)

**Observations.**

- L1 cells land at 0.88 and 0.87 — barely above predict-the-median's
  0.8583. They are **0.026 / 0.008** above L1's no-information floor.
  Compare: MSE Phase A's `phaseA_3` was 0.030 above MSE's
  predict-the-mean floor (0.8979 vs 0.9286). Both losses sit a similar
  small distance above their respective constant-predictor baseline.
- Prediction std collapsed **further** under L1 (0.19, 0.25) than under
  MSE (Phase A diagnostic 3 measured 0.79 on the 30k+3e-3 large run; the
  matched-budget MSE n=0 wasn't probed for std but train-loss curves
  imply similar compression). pred_mean sits at 10.82–10.86 — within
  0.15 of the median (11), not the mean (10.67). The model is
  **collapsing toward the L1-optimal constant**, just as MSE collapsed
  toward the mean.
- Residuals (n=0 → n=2) move val_mae from 0.8819 to 0.8667 (-0.015),
  similar in size to the MSE-side residuals delta (Phase A n=0 0.90 →
  n=2 0.87 in diagnostic 2, ~-0.03). Adding depth helps a little under
  both losses; it is not loss-dominated.

**Hypotheses.**

**H1 (HIGH → FALSIFIED): MSE × peaked V\* depth × BN-everywhere
mean-collapse trap.** L1 was supposed to break the plateau if MSE was
the bottleneck. It did not. Both losses collapse to the
loss-appropriate constant predictor and sit ~0.01-0.03 above that
floor. The bottleneck is not "MSE is the wrong loss for this
distribution" — it is **something more architectural**, likely the
combination flagged in H2 (BN compresses output range) plus the
optimization landscape near any constant-predictor baseline being
locally flat under either loss.

*Confidence on falsification: HIGH.* Same widths, n_res, lr, steps,
seed, split — only the loss differs. Outcomes are nearly symmetric
relative to each loss's own constant-predictor floor, with L1 if
anything compressing predictions *more* (0.19–0.25 vs MSE's 0.79).

**H2 (medium → promoted to leading): BatchNorm-before-head compresses
output range.** With H1 falsified, H2 inherits the plateau-cause role.
The pred_std numbers under L1 (0.19, 0.25) are dramatically smaller
than under MSE (0.79 in the large diagnostic 3 run). That is
loss-asymmetric in a direction H2 predicts: L1's `sign()` gradient
gives the head no magnitude signal scaled to the residual size, so
absent normalization-undoing layers the head cannot learn to widen
output range. MSE at least gives the head error-magnitude signal, but
BN before the head still bounds the body's representation magnitude.

*Updated verification plan:* drop BN immediately before the projection
layer (or replace BN with LayerNorm body-wide), keep MSE for clean
comparison against `phaseA_3_2048x512`. Same widths/lr/steps/seed.
If pred_std climbs significantly and val_mae drops below 0.7, H2 is
confirmed.

## Open questions (revised post-L1)

1. **Does removing BN before the head fix output compression?** Now
   the highest-leverage single experiment. `(2048, 512)` n=0, MSE,
   matched budget, BN-stripped projection. Compare pred_std.
2. **Is BN itself the issue, or BN-on-this-distribution?** If (1)
   helps a lot, follow up: replace BN with LayerNorm body-wide, same
   cell. If LN keeps the win, the issue was BN-statistics on a peaked
   target distribution, not normalization in general.
3. **Should T1's framing change?** Even more strongly than before —
   T1's "capacity question" framing assumed gradient signal would let
   capacity matter. With the loss swap not unblocking, the
   loss/normalization choice is upstream of capacity in a way the
   methodology didn't anticipate.
4. **Carryover, deferred:** the LR-warmup question and the
   median-aware sampling question remain open but lower-priority than
   the BN-removal experiment.

## What we haven't verified

- We did **not** measure pred_std on `phaseA_3_2048x512` directly under
  matched 7k-step budget — the 0.79 figure is from the *larger*
  30k-step diagnostic 3 run. The MSE-side compressed-pred-std story
  could be tighter (smaller std) at 7k steps; the L1 vs MSE pred_std
  comparison should ideally rerun MSE Phase A with the std logging in
  place. Cheap follow-up if needed.
- The L1 runs were single-seed. The 0.015 n_res delta could be
  seed-noise — replicate at seeds {1, 2} before treating it as signal.
  Not load-bearing for the H1-falsified verdict, which rests on the
  much-larger gap from the MSE plateau and from each loss's
  constant-predictor floor.

## BN ablation + autonomous investigation (2026-05-04)

Charged with: build a normalization toggle on `ValueNet`, run the
(BN/no-BN)×(small/big) primary contrast, then adapt based on what the
data says. Hour-budget.

Built `normalization: "bn" | "none" | "ln"` toggle on `ValueNet`
(commit `6337396`) and ran the 4 primary cells. Findings forced a
reframe to a sampling probe; depth-balanced sampling was added
(commit `301e429`) and run on the larger network with both
normalizations. Phase A's 7k-budget pred_std was finally measured under
matched conditions and is *much* tighter than the 30k+lr3e3 figure
implied — Phase A's pred_std at 7k is ~0.35, not ~0.79.

### Observations *(mechanical)*

All cells: 2x2 V\*, 80/20 split (split_seed 42), batch 1024, 7k steps,
MSE, seed 0, MPS. Final values reported.

| cell | norm | sampler | lr | n_params | val_mae | pred_mean | pred_std |
|------|------|---------|----|---------:|--------:|----------:|---------:|
| `bn_2048x512_n0_baseline` | bn | uniform | 1e-3 | 1.35M | 0.9023 | 10.62 | **0.350** |
| `nobn_2048x512_n0` | none | uniform | 1e-3 | 1.35M | 0.8928 | 10.73 | 0.316 |
| `bn_4096x1024_n2` | bn | uniform | 1e-3 | 9.01M | **0.8577** | 10.67 | 0.461 |
| `nobn_4096x1024_n2` | none | uniform | 1e-3 | 8.99M | 0.8923 | 10.70 | 0.321 |
| `bn_4096x1024_n2_lr3e3` | bn | uniform | 3e-3 | 9.01M | 0.8567 | 10.66 | 0.478 |
| `dbal_2048x512_n0` | bn | depth_balanced | 1e-3 | 1.35M | 1.1285 | 10.30 | **1.151** |
| `dbal_4096x1024_n2` | bn | depth_balanced | 1e-3 | 9.01M | 0.9545 | 10.60 | **1.125** |
| `dbal_nobn_4096x1024_n2` | none | depth_balanced | 1e-3 | 8.99M | 1.0806 | 10.40 | 1.073 |

**Per-depth predictions on 10k val states:**

| depth | uniform `bn_4k_n2` pred_mu | dbal `bn_4k_n2` pred_mu | truth |
|------:|---------------------------:|------------------------:|------:|
| 3 | 5.69 | 4.06 | 3 |
| 5 | 9.48 | 6.26 | 5 |
| 7 | 9.72 | 8.11 | 7 |
| 9 | 10.48 | 9.92 | 9 |
| 11 | 10.74 | 10.81 | 11 |
| 13 | 10.78 | 11.32 | 13 |

**Key mechanical observations:**

1. **No-BN does NOT break the plateau.** At small (2048,512) n=0,
   no-BN is 0.8928 vs BN's 0.9023 — basically identical. At big
   (4096,1024) n=2, BN is *better* than no-BN (0.8577 vs 0.8923).
2. **Phase A's pred_std at matched 7k budget is ~0.35** (not 0.79
   from the 30k+lr3e3 large diagnostic). The "compression" is even
   tighter than previously thought at the matched budget.
3. **Adding 2 residual blocks at 4096×1024 BN gives -0.04 val_mae
   over phaseA_1's n=0 result** (0.8577 vs 0.8979) — depth helps
   under uniform sampling but doesn't break the plateau.
4. **Depth-balanced sampling makes pred_std jump from ~0.35 to
   ~1.15** (target std 1.16) — within 0.01 of the target. Per-depth
   predictions track ground-truth across the full 0–14 range, not
   collapsed to ~10.7. The trade-off: val_mae *goes up* (0.95–1.13)
   because val is uniform-distributed and depth-balanced
   under-weights the bulk that uniform was already nearly-optimal on.
5. **Depth-balanced + no-BN works almost as well as depth-balanced +
   BN** (pred_std 1.07 vs 1.13; train loss 1.11 vs 0.72 at step 7k).
   The dominant factor is the sampler, not the normalization. BN is
   slightly *helpful* when sampling is fixed, not harmful.
6. **LR=3e-3 at the BN big cell gives val_mae 0.8567** vs 0.8577 at
   1e-3 — within seed noise. LR is not the bottleneck at this size.
7. **Train loss curves under depth-balanced are still falling
   sharply at step 7000** (1.71→0.72 for the big BN cell). The
   plateau is gone; this regime has not converged in 7k steps.

### Hypotheses

**H4 (HIGH confidence): The val_mae plateau under uniform sampling is
a *gradient-coverage* bug, not a *capacity* or *normalization* bug.**
The V\* depth distribution is severely peaked (83% of states at
depths 10–12, only 0.05% at depths 0–4). With uniform-with-replacement
sampling, gradient signal toward the tails is proportional to their
frequency — a 1024-batch step sees 0.4 expected samples at depth 4
combined and ~600 at depth 11. Under MSE, total tail contribution to
loss is tiny, so the optimization landscape rewards "predict the
modal class" — the model collapses to ~10.7 across all inputs. Once
sampling is rebalanced so each depth contributes equally to per-step
gradient, the model immediately learns to predict across the full
depth range (pred_std 0.35 → 1.15 in the same budget).

*Supporting evidence:* (a) under uniform sampling, **NO** axis tested
(capacity 23×, normalization {bn,none,ln-untested-here}, depth
{n=0,n=2}, lr {1e-3,3e-3}, loss {mse,l1}, training set size {1k
memorize, 2.94M full}) breaks the ~0.90 plateau. (b) Under
depth-balanced sampling with **all other things equal**, pred_std
jumps to within 0.01 of target std and per-depth predictions track
truth across the entire 0–14 range. (c) BN/no-BN are
near-equivalent under depth-balanced sampling (1.07 vs 1.13 std),
falsifying the BN-clamps-output-magnitude story (H2 from prior
section). (d) Under depth-balanced, the train-loss curve is still
falling fast at step 7k — the previous "plateau" was a stable point
of the optimization, not a capacity ceiling.

*Verification plan:* re-run `dbal_4096x1024_n2` for 30k steps. If
val_mae falls below 0.5 (uniformly weighted, against bulk-favored
val) the H4 story locks: depth-balanced sampling provides usable
gradient signal across all depths and the network can fit V\* given
enough compute. If it plateaus around 1.0 then there's a residual
distributional mismatch (depth-balanced loss favors tails which val
under-weights) and the *evaluation* methodology may also need
attention. Either outcome is informative.

*Falsification:* if a 30k-step run on `dbal_4096x1024_n2` *also*
plateaus at val_mae > 0.7 with pred_std stuck near 1.1, then there
**is** a remaining bottleneck below sampling — probably in the
depth-balanced loss formulation interacting with squared-error on the
rare tail buckets (each tail bucket of ~70 samples contains tons of
duplicates from depth 0/1/2/14 due to with-replacement sampling).

**H1 (FALSIFIED earlier, still falsified):** L1 vs MSE was a
red-herring. Both losses collapse under uniform sampling.

**H2 (now FALSIFIED):** BN-before-head was *not* the source of output
compression. BN and no-BN produce the same plateau under uniform
sampling. Under depth-balanced, BN is slightly *helpful* in
optimization speed (train loss 0.72 vs 1.11 at step 7k for the same
cell). The previous "compression" interpretation was a symptom of
the sampling-induced gradient distribution, not a normalization
artifact.

**H3 (still open, lower priority):** Residual depth helps modestly
under uniform sampling (~0.04 absolute val_mae gain at the same
widths), but not nearly enough to break the plateau. The mechanism is
likely "more depth = more capacity to learn the bulk-vs-tail
distinction" — but under depth-balanced sampling, the gradient signal
already rewards that distinction directly, so depth's marginal value
under depth-balanced is the relevant question. Unmeasured.

### What this means for T1's framing

T1's tier-by-tier methodology assumed gradient signal would be
present and that capacity was the right first-question. H4 says the
methodology has a hidden zeroth tier that needs to land before
capacity becomes a measurable axis: **sampling**.

The new T1 should be:

- **T1a — Sampling (yes/no).** Does depth-balanced sampling produce
  per-depth predictions that track truth? (Answered: YES at 4096×1024
  n=2 with pred_std 1.13 vs target 1.16.)
- **T1b — Sampling × evaluation.** Once tail predictions are real,
  what's the right evaluation metric? Uniformly-weighted val_mae
  rewards the bulk-collapse strategy that depth-balanced sampling
  intentionally walks away from — so the "did we improve" question
  needs a depth-aware aggregation. Likely candidates: per-depth MAE
  table; depth-weighted val_mae; "macro-MAE" averaging per-depth MAEs
  uniformly. **Not free** — picking the metric is itself a methodology
  decision.
- **T1c — Capacity floor under fixed sampling+metric.** Now run the
  width × depth ablations the original T1 was meant to run, with
  depth-balanced sampling and macro-MAE, with budgets long enough to
  see the still-falling loss curve converge.

The two-phase widths-then-residuals plan from the previous methodology
correction is still right shape, but it sits *under* T1a/T1b — they
have to land first.

### Open questions

1. **30k-step `dbal_4096x1024_n2`.** Does the still-falling loss
   curve actually hit a useful val performance? Predicted answer per
   H4: macro-MAE drops well below 1.0 by step 30k, uniform-weighted
   val_mae stays elevated (~0.9–1.0) because the val set is
   bulk-dominated.
2. **Macro-MAE evaluation.** Recompute Phase A and the new cells
   under macro-MAE to get a comparison that isn't biased by val's
   bulk-dominance. Cheap — just an analysis script, no retraining.
3. **Depth-weighted MSE loss.** Instead of (or in addition to) a
   depth-balanced sampler, try MSE weighted by `1/freq[depth]` per
   sample with uniform sampling. Mathematically related but lets the
   model see the bulk distribution it'll be evaluated on. May
   converge to a different point than depth-balanced sampling.
4. **Depth-balanced sampling + smaller capacity.** Does
   `dbal_512x256_n0` (207K params) also escape the plateau? Tests
   whether the original capacity-floor question is meaningful once
   sampling is fixed. Probably the right next experiment after the
   metric question is answered.
5. **DAVI implications.** The DAVI training distribution is
   *generated by random scrambles* of varying depths, not by
   sampling V\*. The sampler-vs-gradient story above is specific to
   supervised regression on V\*. Whether DAVI's natural curriculum
   (uniform-over-scramble-depth?) sees this same pathology is a
   separate question, but the answer informs whether the
   T1→T2→T3→… methodology even maps to the DAVI we'll eventually
   train.

### What we haven't verified

- The H4 "30k steps converges below 0.5 macro-MAE" prediction is
  literally a prediction — it has not been run. The hour budget
  ended after the 7k-step ablations finished. The 30k-step
  verification is the highest-priority follow-up.
- We did not test LayerNorm (`normalization: ln`). The toggle is
  built and tested in unit tests, but no training run used it. If
  the team wants the BN-vs-LN story for completeness, that's a
  no-code-needed two-cell run.
- The depth-balanced sampler over-samples the very-rare tail buckets
  (depth 0 has 1 train state, sampled with replacement ~68 times
  per batch — every step the model sees 68 copies of the same
  depth-0 state). At 7k steps that's 480k gradient updates against
  one state. We did not check whether this causes the head to
  *overfit* the tail — the trained model's depth-0/1 predictions
  look directionally correct (4.06 for depth 3, where d=3 has 102
  samples and is sampled ~68/step) but a held-out tail-bucket test
  (e.g. predict for the held-out depth-3 *val* states, which is
  exactly what was reported above) is the cleanest probe — and we
  have it: per-depth predictions on val show no obvious overfit, just
  appropriate generalization. Still, with-replacement sampling on
  buckets of size 1 is unusual; if the 30k-step run shows
  weirdness on the tails, this is the suspect to investigate.
- Single seed throughout. The val_mae numbers under depth-balanced
  could shift ~0.05 with a different seed, especially given the
  small bucket sizes. The qualitative finding (pred_std jumps from
  0.35 to ~1.15) is robust to seed; the absolute val_mae values are
  not.
- We measured per-depth predictions on a 10k-state val subset only
  for the BN big cells. We did not run per-depth probes on the
  no-BN cells under depth-balanced — confidence that "depth-balanced
  fixes the per-depth picture" rests on pred_std as a proxy plus the
  one detailed table; a no-BN per-depth probe is a cheap follow-up
  if challenged.

## T1c verification — 30k dbal_4096x1024_n2 (2026-05-04)

**Set up.** Same cell as `bn_4096x1024_n2_dbal` (4096×1024 BN n=2, batch
1024, lr 1e-3, MSE, sampler=depth_balanced, seed 0, split_seed 42)
extended from 7k → 30k steps. Eval logger now reports `macro_mae`
(uniform mean across per-depth MAEs — the T1b metric) and
`per_depth_mae` (dict[int, float]) at every 1000-step eval. Uniform
val_mae retained for backward comparability.

Methodology doc (`plans/m5-davi-methodology.md` §T1) was rewritten to
the T1a/T1b/T1c structure first, then this run was launched as the
T1c **pre-requisite** (gate for the widths sweep).

### Observations *(mechanical)*

**Headline numbers** at step 30k vs step 1k:

| step | train_loss | val_mae | macro_mae | pred_mean | pred_std |
|-----:|-----------:|--------:|----------:|----------:|---------:|
|  1000 | 1.71 | 1.148 | 1.437 | 10.072 | 0.995 |
|  7000 | 0.72 | 0.955 | 1.131 | 10.604 | 1.125 |
| 15000 | 0.52 | 0.938 | 1.085 | 10.713 | 1.160 |
| 30000 | 0.31 | 0.889 | 1.170 | 10.593 | 1.091 |

**macro_mae trajectory (eval-every-1000 noise visible).** From step 7k
through 30k, macro_mae oscillates in [1.02, 1.17] with **no descent**
— the final value (1.17) is *higher* than the 7k value (1.13). Train
loss in the same window falls 0.72 → 0.31 (2.3× reduction). This is
a classic train-val gap: model is fitting train-distribution structure
the val set doesn't share.

**Per-depth MAE trajectory** (val subset has depths 3–13 only — see
"What we haven't verified"):

| depth | step 1k | step 7k | step 15k | step 30k | pattern |
|------:|--------:|--------:|---------:|---------:|---------|
| 3 | 0.43 | 1.06 | 1.40 | **2.04** | ✗ getting worse 4.7× over training |
| 4 | 0.56 | 0.59 | 0.60 | 0.69 | drift up, mild |
| 5 | 1.92 | 1.47 | 1.06 | 1.68 | improves then regresses |
| 6 | 2.03 | 0.95 | 0.98 | 1.01 | improves then plateau |
| 7 | 1.92 | 1.33 | 1.37 | 1.14 | slow improve |
| 8 | 1.54 | 1.32 | 1.15 | 1.12 | slow improve |
| 9 | 1.13 | 1.32 | 1.24 | 1.14 | flat |
| 10 | 0.72 | 0.98 | 0.99 | 0.85 | volatile, no clear trend |
| 11 (modal) | 0.92 | 0.72 | 0.76 | 0.70 | best; slow improve |
| 12 | 1.78 | 1.02 | 0.93 | 1.04 | improves then plateau |
| 13 | 2.87 | 1.69 | 1.46 | 1.47 | improves then plateau |

**Depth 3 is the smoking gun.** 0.43 at step 1k → 2.04 at step 30k
— the model gets *worse* at depth 3 over training. Depth 3 has 102
train states; under depth-balanced sampling with batch 1024 and 15
buckets, each step draws ~68 with-replacement samples from those 102
states. Over 30k steps that's ~2M gradient examples against 102
unique inputs.

**Bulk depths (10, 11) show modest improvement.** Modal depth 11
goes from 0.92 → 0.70. Depth 10 drifts but ends marginally better.

**Pred std drifted slightly down** (1.16 at step 7k → 1.09 at step
30k) — model is *narrowing* its prediction range slightly, opposite
to what's needed.

**Train loss kept falling**: 1.71 → 0.72 → 0.31. Continued descent
through all 30k steps suggests the model has more capacity to fit
the training distribution; what it's fitting just doesn't transfer.

### Hypotheses

**H4 (FALSIFIED at this sampler implementation).** The earlier H4
predicted macro_mae < 0.5 at 30k. Empirically macro_mae **plateaus
around 1.10** with no descent; depth 3 *worsens* over training; train
loss keeps falling. Depth-balanced sampling is necessary (it broke
the uniform-sampling collapse — pred_std jumped from 0.35 to 1.15)
but **not sufficient** at the current with-replacement-from-bucket
implementation.

*Confidence:* HIGH. Single-cell, single-seed, but the falsification
mechanism (per-depth MAE for depth 3 going 0.43 → 2.04 while train
loss continues falling) is unambiguous overfitting and explains the
plateau directly.

**H5 (NEW, HIGH confidence): The with-replacement sampler causes
tail-bucket overfit.** Sampling ~68 examples per step from 102 unique
states (depth 3) means the model trains to memorize those 102 states.
By step 30k it has seen ~2M gradient updates on them. The
depth-balanced sampler intended "give equal gradient signal per
depth"; what it actually gives is "memorize each tail bucket
independently." Depth 11 has 1.08M train states (modal) so it's still
seeing fresh examples per batch — that's why the modal depth keeps
improving while tails regress.

*Supporting evidence:* (a) depth 3 (102 train states) regresses 4.7×
over training while depth 11 (1.08M train states) improves; (b)
train loss keeps falling while val plateaus — train fits the
oversampled tails, val tests on held-out tail states; (c) all the
"easy" tail depths (3, 4, 5) have either flat or upward trajectories
in their per-depth MAE.

*Verification:* the next experiment (sampler design fix) should
either resolve the depth-3 regression *or* falsify H5 if the
regression persists.

### What this means for T1c's pre-requisite

T1c was gated on this run hitting macro_mae < 0.5. **Gate fails.**
The widths-then-residuals Phase A/B sweep is *not* the right next
move — sweeping cells with a sampler that overfits tails just
characterizes the overfit at multiple capacities. The sampler design
needs to be fixed first.

### Open questions (proposed next experiments)

The methodology says "T1 is yes/no, don't expand" — so T1c stays one
question with a single sampler-design test. The cleanest single
alternative to with-replacement-from-bucket is:

1. **Frequency-weighted MSE under uniform sampling.** Sample uniformly
   from the train set (so every state is seen at its natural rate —
   no artificial repetition), but weight each sample's loss by
   `w(d) = N / (n_buckets · freq[d])` so each depth's expected
   gradient contribution is equalized. Mathematically related to
   depth-balanced sampling but no with-replacement pathology — the
   model sees the diversity of the training distribution directly.
   `w(d=3) ≈ 1920` (huge weight, but only 102 states ever get it),
   `w(d=11) ≈ 0.18` (tiny weight, but 1.08M states get it). Net
   gradient per depth bucket is still equal in expectation but the
   set of states contributing is the *full* bucket each batch, not
   ~68 with-replacement copies.

2. **Without-replacement sampling up to bucket size, with-replacement
   past it.** Each batch draws `min(take, bucket_size)` unique
   samples per bucket; if `take > bucket_size`, fills extras with
   replacement. Tail buckets still see repetition (depth 3 sees each
   state at most once per batch, then 0–1 extras), but the
   per-batch gradient diversity is maximized. Probably less clean
   than option 1 because depth 0 (1 state) still gets repeated.

3. **Bigger train set for tail depths via data augmentation.** The
   2x2 has 24 cube-rotation symmetries; depth 3 has 102 canonical
   states ⇒ ~2448 raw states. Currently we deduplicate to canonical
   form. Re-expanding to symmetric variants would 24× the depth-3
   bucket. Adds engineering surface (data path); probably overkill
   for what option 1 buys cheaply.

**Recommendation.** Option 1 is the cleanest single-axis test of H5.
Same cell (4096×1024 BN n=2, lr 1e-3, MSE, batch 1024), 30k steps,
seed 0. If macro_mae descends below 0.5 (or even just below 1.0),
H5 confirmed and T1c's widths sweep can proceed under the new
sampler. If it plateaus near current numbers, the limit is somewhere
deeper than sampler design — back to the drawing board.

4. **Subsidiary question (cheap):** re-evaluate the existing 7k cells
   using the new `_eval_val` function (loads checkpoint, recomputes
   macro_mae + per_depth_mae). Don't need to retrain — just need an
   analyze-existing-runs script. Adds the macro-MAE column to the
   prior comparison table. This is essentially free and worth doing
   in parallel with experiment 1.

### What we haven't verified

- **Val eval set contains only depths 3–13.** The 10k-state random
  subset of val (which is itself a random 20% of V*) ended up with
  zero examples of depths 0, 1, 2, 14 — they're rare enough that the
  random sample missed them. Macro-MAE is averaged over the 11
  depths actually present, not all 15. Future T1c runs should
  switch to a depth-stratified val subset (the methodology's
  long-flagged `eval_set.npz` from the pre-tier deliverable section)
  so macro-MAE is computed over a fixed, depth-complete set. Listed
  as the methodology pre-tier deliverable but never landed; this is
  an open methodology debt.
- Single seed (seed 0). The depth-3 regression is strong enough that
  seed-noise won't reverse it, but the *exact* trajectory shape
  could shift.
- We did not run the val-set checkpoint reload to compare against
  Phase A's val_mae plateau under the new metric — the
  retroactive-comparison follow-up (open question 4) hasn't run
  yet.
- We did not check whether depth 3's regression is also visible in
  *train* per-depth MAE (would tell us "model fits train depth-3
  perfectly but val depth-3 is held-out" vs "model can't fit train
  depth-3 either"). Cheap follow-up if challenged.

