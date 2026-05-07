# precision-sweep — intuition

**Datestamp:** 2026-05-07
**Run conditions:** see `results.md`. M4 Max / MPS, 5120/1024 ValueNet,
post-A1+A2 cross-scramble batched beam.py, baseline checkpoint
`net_final.pt`, n=100, 14 depths, 7 widths, seed=0.

## Observations

What the wall-and-rate tables actually show, separating mechanical
observation from interpretation.

- **Width=128 BF16 wall is 1.37× faster than FP32 (43.6s vs 59.6s).** The single-layer pre-experiment probe predicted 1.30×. The full-net win is *slightly larger* than the single-layer probe — opposite of the prior expectation that "memory bandwidth, batchnorm, etc." would dilute the gain. FP16 lands at 43.7s, essentially tied with BF16.
- **Solve rates at width=128 are bit-identical across all three precisions, at every depth.** Including avg_solve_len. FP32, BF16, FP16 all give `(0.640, 0.820, 0.970)` at d=14, d=13, d=12.
- **The 7-width FP32 sweep on the current HEAD took 25:35.** This is consistent with the LOG entry's "33 min on the current beam.py" prediction (the LOG number was rougher; this is the precise figure). Width=256 alone takes 9:38 and width=512 takes 13:39 — between them they're 91% of total wall.
- **The 7-width BF16 sweep took 16:43.** A 1.53× wall reduction across the whole sweep.
- **Width=256 is where BF16 wins biggest.** FP32 takes 578s; BF16 takes 113s — **5.13×** speedup. At the smaller widths (8..128) BF16 wins 1.17–1.76×. At width=512 BF16 only wins 1.03× — basically flat.
- **The width=256 FP32 cell behaves as a throughput-knee outlier in a way the BF16 cell doesn't.** FP32 ramp: 5.0 → 7.8 → 15.5 → 29.6 → 74.9 → **578.2** → 818.5 (the 7.7× jump from w=128 → w=256 is anomalous). BF16 ramp: 4.3 → 6.3 → 12.1 → 22.6 → 42.6 → **112.8** → 797.9 (smoother through w=256, then jumps 7.1× at w=512). The knee shifted right by one width tier.
- **No NaN, no kernel errors, no MPS fallbacks.** BF16 on Linear + BatchNorm + ReLU + topk + where on MPS all completed cleanly. The code change in `network.py` was a single line and turned out to be the only source-side change required.
- **At width=512 both precisions converge.** FP32 818s vs BF16 798s — BF16 is barely faster. At ~614k states (`100 × 512 × 12`) BF16's smaller working set is no longer enough to keep the per-step batch under whatever resource limit dominates.
- **Across all 98 cells, max |solve_rate Δ| is 0.02.** And BF16 at width=512 is *slightly better* than FP32 at d=13 (+0.02) and d=14 (+0.01). At smaller widths the deltas are noise around 0.

## Hypotheses

Interpretive claims with confidence + supporting evidence + verification plan.

### H1 — BF16 shifts the MPS throughput knee from ~150k states to ~300k states

**Confidence:** Medium-high.

**Evidence:** Every BF16 tensor in the forward pass occupies half the bytes of its FP32 counterpart. The throughput knee is bandwidth-bound (the LOG documented this when ruling out compute-bound explanations). Empirically the FP32 knee shows up sharply between w=128 (75s) and w=256 (578s) — a 7.7× jump for a 2× width — while BF16's equivalent jump (43s → 113s, 2.6×) is much closer to "expected linear in batch size." But BF16's w=256 → w=512 jump (113s → 798s, 7.1×) re-introduces the knee behavior. So BF16 hasn't *removed* the knee, just *shifted* it by ~one width tier.

**Verification plan:**
1. Sweep at finer width granularity around the candidate knees: FP32 widths {160, 192, 224} and BF16 widths {320, 384, 448}. The first width past the knee should show the same step-function jump; widths below should ramp smoothly.
2. Capture per-step `forward()` wall via the same instrumentation pattern used in M8 (Bash + perf_counter wrapping `net(...)` calls), confirming the wall jump comes from `net_fwd` time, not orchestration.
3. If true, the production-cycle eval can run width=256 BF16 in ~2 min instead of ~10 min — a quality-preserving 5× speedup on the canonical sweep.

### H2 — Solve-rate equivalence across precisions is a property of *this* net, not a general guarantee

**Confidence:** Medium-high.

**Evidence:** Beam search is a discrete selection algorithm fed by continuous scores. At each step it picks topk indices; the values themselves don't propagate. So as long as BF16's quantized scores preserve the same topk ordering as FP32's exact scores, the beam search is bit-exact-equivalent. With 7 mantissa bits BF16 can represent ~128 distinguishable values per binade — for the trained net's outputs (V values in roughly [0, 14] for a 3x3), that's score resolution of ~0.05–0.1, which appears to be coarser than the actual gaps between the topk and topk+1 entries. The gaps must be larger than that, on this trained network.

But this isn't a structural guarantee. A poorly-trained net (or one with regularization that compresses the value range) might have score gaps below BF16's resolution at deep cells, breaking equivalence. The +0.02 |Δ| at width=512 d=13 hints that some cell-level reordering *does* happen — it's just statistically inseparable from sampling noise here.

**Verification plan:**
1. Repeat this sweep on a *different* checkpoint (e.g. an earlier-step checkpoint with worse training) — does max |Δ| stay ≤ binomial SE?
2. Inspect the actual topk gap distribution: at depth d=14, width=128, what's the median gap between the 128th and 129th score? If it's > 0.1 we're comfortably above BF16 resolution; if < 0.01 we're getting lucky.
3. If a future net trained at a different scale shows |Δ| > binomial SE, that's a signal the net's value distribution has tightened — keep the precision flag, fall back to FP32 for that net, and document the inflection.

### H3 — At width=512, FP32 and BF16 converge because both blow past whatever limit dominates at ~600k states

**Confidence:** Low-medium.

**Evidence:** FP32 wall at w=512 is 818s; BF16 is 798s — only 1.03×, while every other width tier gave 1.17–5.13×. If the bottleneck were pure floating-point throughput we'd expect BF16's win to grow at higher batches (more arithmetic per launch); seeing it *shrink* says something else dominates. Candidates: (a) MPS Metal command-buffer scheduling overhead that scales with tensor count, not byte count; (b) memory pressure from accumulator buffers (the `dense_v` tensor of shape `(N_active, B*12)` and the topk index buffer); (c) thermal throttling kicking in mid-run.

**Verification plan:**
1. Run `mactop` or `macmon` during a width=512 sweep to check thermal throttling. If the GPU is at 100% steady through both runs and frequencies don't drop, rule out (c).
2. Profile a single width=512 forward call via Metal Performance Shaders trace tools to confirm whether the bottleneck is kernel launch count or memory bandwidth.
3. If H3 holds, width=512 isn't a productive target for further inference-perf work — accept the floor and either move to a smaller cube or push elsewhere (training-side perf, beam-tree pruning).

## Open questions

Well-defined next experiments worth doing.

- **Should we quant-aware fine-tune for INT8?** With BF16 already at the "no quality loss" mark and width=256 already 5× faster, the marginal value of INT8 is limited to width=512 cases (where BF16 didn't help). Probably not worth the complexity unless we find a downstream cycle where w=512 capability is mandatory and w=256 isn't enough.
- **Does BF16 inference work the same on a smaller-net checkpoint?** The 5120/1024 net has FP32 BatchNorm running stats accumulated during training — when we cast to BF16 those stats lose precision. For a smaller net (say 1024/256) the running stats might be more sensitive to rounding. Worth a quick sniff.
- **Should training move to BF16 too?** **DECISION DEFERRED — this experiment is inference-only.** Training in BF16 has different tradeoffs (gradient underflow at small magnitudes, batchnorm stat drift). The user explicitly scoped this experiment to inference; training-side precision is its own block.
- **What's the actual MPS throughput knee in absolute terms?** The Phase 2 ramp gives a strong hint but doesn't pinpoint it. One narrow-grain sweep (widths {96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256} for FP32; {200, 240, 280, 320, 360, 400} for BF16) would localize the FP32 knee to a width tier and give a ground-truth states/sec curve for the GPU-intuition writeup.

## What we haven't verified

Caveats on the most speculative claims.

- **Solve-rate equivalence is shown only for n=100, seed=0, on this one checkpoint.** The "max |Δ| = 0.02" headline is one realization. Multi-seed runs (n=100 with seeds 0..4) would give a per-cell variance estimate; we have one sample per cell. The binomial SE bound says any single-seed |Δ| ≤ 0.046 is "within noise" — we're well under that — but we haven't sampled the seed-distribution.
- **The width=256 5.13× wall reduction is one observation.** Another sweep could come in at 4× or 6× depending on system load, thermal state, and MPS scheduler pressure. The fact that it's *qualitatively* a step-function shift (knee moved a tier right) is the load-bearing claim; the exact factor isn't pinned to the second decimal.
- **avg_solve_len being identical at all 14 depths × 3 precisions is somewhat surprising.** Even with identical solve-rate, you'd expect *some* paths to differ across precisions when topk ties are broken differently. The fact that avg_solve_len matches exactly suggests the trained net produces no exact ties in the topk window — every selection is unambiguous, so quantization can't reorder. This is consistent with the LOG's earlier C4 observation that "no tied-V states observed at width=128 on the trained net."
