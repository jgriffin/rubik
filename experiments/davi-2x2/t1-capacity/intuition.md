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
