# beam-search-2x2 — results

_Run dir: `experiments/beam-search-2x2/runs/sync500_kmax20-baseline`._

- **checkpoint:** `experiments/davi-2x2/davi-baseline/runs/sync500_kmax20-30k/net_final.pt`
- **device:** `mps`
- **params:** 13.21M
- **arch:** `body_widths=[4096, 1024]`, `n_residual_blocks=4`, `normalization=bn`
- **grid:** depths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] × beam_widths=[1, 4, 16, 64, 256] × n_per_depth=200 = 70 cells
- **max_steps:** 20
- **wall:** 334.1s

## Solve rate (depth × beam_width)

| depth \ width | w=1 | w=4 | w=16 | w=64 | w=256 |
|---:|---:|---:|---:|---:|---:|
| d=1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| d=2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| d=3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| d=4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| d=5 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 |
| d=6 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |
| d=7 | 0.860 | 0.970 | 1.000 | 1.000 | 1.000 |
| d=8 | 0.800 | 0.940 | 0.985 | 1.000 | 1.000 |
| d=9 | 0.690 | 0.810 | 0.930 | 1.000 | 1.000 |
| d=10 | 0.520 | 0.710 | 0.865 | 0.955 | 0.990 |
| d=11 | 0.480 | 0.620 | 0.815 | 0.935 | 0.990 |
| d=12 | 0.345 | 0.560 | 0.760 | 0.865 | 0.985 |
| d=13 | 0.325 | 0.535 | 0.725 | 0.875 | 0.990 |
| d=14 | 0.365 | 0.485 | 0.665 | 0.805 | 0.970 |

## Mean V*-excess (depth × beam_width)

_Mean over solved attempts only. Below the M6 acceptance gate of `mean_v_star_excess ≤ 1.0` per depth = good._

| depth \ width | w=1 | w=4 | w=16 | w=64 | w=256 |
|---:|---:|---:|---:|---:|---:|
| d=1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| d=2 | 0.210 | 0.210 | 0.210 | 0.210 | 0.210 |
| d=3 | 0.360 | 0.360 | 0.360 | 0.360 | 0.360 |
| d=4 | 0.420 | 0.420 | 0.420 | 0.420 | 0.420 |
| d=5 | 0.563 | 0.560 | 0.560 | 0.560 | 0.560 |
| d=6 | 0.444 | 0.440 | 0.440 | 0.440 | 0.440 |
| d=7 | 0.430 | 0.412 | 0.410 | 0.400 | 0.400 |
| d=8 | 0.412 | 0.340 | 0.325 | 0.320 | 0.320 |
| d=9 | 0.406 | 0.407 | 0.344 | 0.270 | 0.270 |
| d=10 | 0.385 | 0.338 | 0.347 | 0.220 | 0.192 |
| d=11 | 0.375 | 0.290 | 0.331 | 0.203 | 0.182 |
| d=12 | 0.319 | 0.500 | 0.342 | 0.162 | 0.162 |
| d=13 | 0.338 | 0.374 | 0.193 | 0.103 | 0.111 |
| d=14 | 0.438 | 0.206 | 0.256 | 0.149 | 0.113 |

## Avg solve length (depth × beam_width)

_For solved attempts only. Compare to depth d (random scramble length) and to V*-optimal length (= avg_solve_len − mean_v_star_excess)._

| depth \ width | w=1 | w=4 | w=16 | w=64 | w=256 |
|---:|---:|---:|---:|---:|---:|
| d=1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| d=2 | 2.000 | 2.000 | 2.000 | 2.000 | 2.000 |
| d=3 | 3.000 | 3.000 | 3.000 | 3.000 | 3.000 |
| d=4 | 3.850 | 3.850 | 3.850 | 3.850 | 3.850 |
| d=5 | 4.628 | 4.630 | 4.630 | 4.630 | 4.630 |
| d=6 | 5.515 | 5.520 | 5.520 | 5.520 | 5.520 |
| d=7 | 5.942 | 6.052 | 6.090 | 6.080 | 6.080 |
| d=8 | 6.237 | 6.447 | 6.518 | 6.540 | 6.540 |
| d=9 | 6.565 | 6.852 | 7.075 | 7.150 | 7.150 |
| d=10 | 6.731 | 7.155 | 7.561 | 7.665 | 7.727 |
| d=11 | 6.812 | 7.081 | 7.626 | 7.824 | 7.980 |
| d=12 | 6.928 | 7.768 | 8.105 | 8.208 | 8.497 |
| d=13 | 7.369 | 7.953 | 8.145 | 8.440 | 8.747 |
| d=14 | 7.781 | 7.814 | 8.226 | 8.472 | 8.907 |

## Cost (n_expansions, elapsed seconds)

**n_expansions per cell:**

| depth \ width | w=1 | w=4 | w=16 | w=64 | w=256 |
|---:|---:|---:|---:|---:|---:|
| d=1 | 2,400 | 2,400 | 2,400 | 2,400 | 2,400 |
| d=2 | 4,800 | 12,000 | 31,200 | 31,200 | 31,200 |
| d=3 | 7,200 | 21,600 | 69,600 | 184,800 | 307,200 |
| d=4 | 9,240 | 29,760 | 102,240 | 315,360 | 854,820 |
| d=5 | 11,292 | 37,248 | 132,240 | 435,792 | 1,313,292 |
| d=6 | 13,584 | 45,792 | 166,368 | 571,872 | 1,865,640 |
| d=7 | 18,984 | 54,912 | 188,256 | 657,888 | 2,199,552 |
| d=8 | 21,576 | 62,496 | 212,448 | 728,544 | 2,482,176 |
| d=9 | 25,752 | 82,560 | 260,832 | 822,240 | 2,856,960 |
| d=10 | 31,440 | 97,248 | 309,216 | 986,592 | 3,287,040 |
| d=11 | 32,808 | 107,904 | 335,136 | 1,047,264 | 3,440,640 |
| d=12 | 37,176 | 119,040 | 375,264 | 1,229,280 | 3,790,848 |
| d=13 | 38,148 | 122,928 | 392,352 | 1,242,336 | 3,907,584 |
| d=14 | 37,296 | 128,064 | 421,728 | 1,370,592 | 4,142,748 |

**Elapsed seconds per cell:**

| depth \ width | w=1 | w=4 | w=16 | w=64 | w=256 |
|---:|---:|---:|---:|---:|---:|
| d=1 | 0.65 | 0.45 | 0.49 | 0.45 | 0.44 |
| d=2 | 0.87 | 1.06 | 1.02 | 1.01 | 0.93 |
| d=3 | 1.20 | 1.36 | 1.46 | 2.06 | 2.63 |
| d=4 | 1.55 | 1.77 | 2.00 | 2.77 | 3.94 |
| d=5 | 1.80 | 2.12 | 2.38 | 4.09 | 5.05 |
| d=6 | 2.17 | 2.41 | 2.89 | 4.89 | 6.74 |
| d=7 | 2.99 | 2.98 | 3.30 | 5.71 | 7.98 |
| d=8 | 3.48 | 3.24 | 3.58 | 5.87 | 8.87 |
| d=9 | 3.99 | 4.20 | 4.12 | 6.98 | 10.22 |
| d=10 | 4.87 | 5.08 | 5.03 | 7.54 | 11.36 |
| d=11 | 5.16 | 5.61 | 5.36 | 8.82 | 11.82 |
| d=12 | 5.81 | 6.10 | 5.20 | 9.60 | 12.97 |
| d=13 | 5.95 | 5.89 | 6.18 | 8.99 | 13.56 |
| d=14 | 5.93 | 6.06 | 6.66 | 10.14 | 14.15 |

## Acceptance-shape summary (max-width column)

At `beam_width=256` (sweep's largest cell — surrogate for the C5 acceptance gate):

| depth | n_solved | solve_rate | avg_solve_len | mean_excess | elapsed_s |
|---:|---:|---:|---:|---:|---:|
| d=1 | 200/200 | 1.000 | 1.000 | 0.000 | 0.4 |
| d=2 | 200/200 | 1.000 | 2.000 | 0.210 | 0.9 |
| d=3 | 200/200 | 1.000 | 3.000 | 0.360 | 2.6 |
| d=4 | 200/200 | 1.000 | 3.850 | 0.420 | 3.9 |
| d=5 | 200/200 | 1.000 | 4.630 | 0.560 | 5.0 |
| d=6 | 200/200 | 1.000 | 5.520 | 0.440 | 6.7 |
| d=7 | 200/200 | 1.000 | 6.080 | 0.400 | 8.0 |
| d=8 | 200/200 | 1.000 | 6.540 | 0.320 | 8.9 |
| d=9 | 200/200 | 1.000 | 7.150 | 0.270 | 10.2 |
| d=10 | 198/200 | 0.990 | 7.727 | 0.192 | 11.4 |
| d=11 | 198/200 | 0.990 | 7.980 | 0.182 | 11.8 |
| d=12 | 197/200 | 0.985 | 8.497 | 0.162 | 13.0 |
| d=13 | 198/200 | 0.990 | 8.747 | 0.111 | 13.6 |
| d=14 | 194/200 | 0.970 | 8.907 | 0.113 | 14.2 |

## Charts

See `beam_curves.html` — per-depth small multiples (5/5/4 banding) of solve-rate, V*-excess, and avg-solve-len vs beam_width, plus the (depth × beam_width) heatmap overview.

---

# beam-search-2x2 — intuition

*Datestamp: 2026-05-05. Written after the C3 sweep run
(`runs/sync500_kmax20-baseline`, 70 cells, 334s wall, n_per_depth=200).
Will be extended after the C5 acceptance run.*

## Observations

*(mechanical, from this run's data)*

- **Width=256 capability map.** d=1..9 solve at 1.000 (200/200). d=10..13
  solve in [0.985, 0.990] — 2–3 fails per 200 each. **d=14 solves at 0.970
  (194/200) — the clear weak point.** Acceptance gate (≥0.999) is met at
  d=1..9 and missed at d=10..14.
- **Width-vs-capability curve is monotone-and-not-saturated at deep
  depths.** d=14: w=1→0.365, w=4→0.485, w=16→0.665, w=64→0.805,
  w=256→0.970. Each 4× width step adds roughly +9pp to +14pp solve rate
  at d=14. At w=256 the curve hasn't visibly bent — there's room above.
- **V\*-excess is well under the 1.0 gate at every (depth, width) cell.**
  At width=256 the per-depth excess range is [0.00, 0.56], mean across
  14 depths ≈ 0.27. Gate would pass with margin to spare — capability,
  not solution length, is the binding constraint.
- **V\*-excess shrinks with width at deep depths** (counter-intuitive at
  first glance — wider search "should" only help find solutions, not
  shorten them). d=11: w=1→0.375, w=256→0.182. d=14: w=1→0.438,
  w=256→0.113. The wider beam carries more candidate prefixes through
  each step and the back-pointer pick lands on shorter paths.
- **V\*-excess is identical across widths for d≤6.** All five width
  cells return the same per-depth excess (e.g. d=4: 0.420 across all
  widths). The algorithm finds the same best path at any width when the
  state space within `max_steps=20` of the start is small enough for
  greedy to traverse it.
- **Greedy (w=1) capability ranking from this sweep matches M5
  cycle-3's N=200 capture closely** (sanity check that we're scoring
  the same checkpoint with the same scramble distribution). M5 cycle-3
  N=200 reported d=11 ≈ 0.50, d=13 ≈ 0.40 for sync500_kmax20-30k under
  greedy. This sweep at w=1 reports d=11 = 0.480, d=13 = 0.325. Within
  binomial SE (~0.035 per cell) for d=11; d=13 here is somewhat lower
  than the M5 capture but still in the same ballpark — different
  scramble seeds.
- **Cost grows sublinearly in width.** d=14 wall: w=1→5.9s, w=4→6.1s,
  w=16→6.7s, w=64→10.1s, w=256→14.2s. Cost from w=1 to w=256 is 2.4×
  for a 256× nominal width increase. `n_expansions` scales closer to
  linear with width at deep depths (37k → 4.14M, 112×) but the GPU
  inner-batch is fixed at `beam_size × n_moves`, so launch overhead
  dominates at small widths and per-step compute dominates at large
  widths.
- **Within-beam dedup is doing real work.** d=2 `n_expansions` saturates
  at 31,200 from w=16 onward — the reachable state space at d=2 is
  small enough that wider beams hit the same unique states. The
  raw-bytes dedup keys away the duplication.
- **Apparent V\*-excess non-monotonicity in d=12, d=14 is a
  selection-bias artifact, not a real anomaly.** d=12 V\*-excess goes
  w=1→0.319, w=4→0.500, w=16→0.342. But d=12 solve rate goes
  w=1→0.345, w=4→0.560, w=16→0.760. Each width is averaging excess
  over a *different* set of solved attempts; wider beams solve more
  scrambles including ones that V\* says have a longer optimum, so the
  numerator and denominator both shift. Same effect at d=14. Means
  cross-width V\*-excess comparisons need to account for sample
  population, or be restricted to scrambles every width solved.

## Hypotheses

*(interpretive — confidence + evidence + verification plan each)*

### H1 — Width=256 is roughly 1–2 doublings short of the 99.9% gate at d≥10. Confidence: MEDIUM-HIGH.

The width-vs-solve-rate curve at d=14 is roughly linear in `log4(width)`:
slopes of +0.12, +0.18, +0.14, +0.17 per 4× step. Linearly extrapolating
the last two steps (≈ +0.15 per 4×), to climb from 0.970 → 0.999 at d=14
would need ≈ +0.029 / 0.15 = 0.19 more 4× doublings — call it 1×–2×
more 4× steps, i.e. width 1024–4096. At d=10..13 (currently ≥0.985) one
4× step (w=1024) is plausibly enough.

**Verify**: rerun width sweep at `beam_widths=[256, 1024, 4096]` ×
depths 10..14 × n_per_depth=200, same checkpoint. Walltime estimate:
each cell ≈ width × 14s/256 in worst case (d=14), so a w=4096 cell is
~225s; the new sweep is ~5 cells × 3 widths ≈ 15 cells × ~150s avg ≈
40 min wall.

If width=4096 still misses 0.999 at d=14, the value-net's ordering
signal at deep depths has a hard ceiling that more search cannot break,
and the next move is back to training (V\*-stratified resampling or
curriculum) rather than wider search.

### H2 — Failures at d=14 are concentrated on scrambles whose true V\* is also large (i.e. natural-state-distribution edge cases that the random-walk training under-sampled). Confidence: MEDIUM.

The M5 cycle-2 sampler audit established that random walks of length
K_max=20 still under-represent true depths 13–14 (0.21% of walk-d=18
hit true-d=13, 0% hit true-d=14). The d=14 random-scramble cells in
this sweep contain a mix of true depths < 14 (most) and true depth = 14
(few) — the trained net should ace the former and struggle on the
latter, producing the observed 6/200 failure rate.

**Verify**: for each w=256 failure case at d=10..14, look up `V*[state]`
for the scrambled state and check the failure-rate-vs-true-V\*
distribution. If failures cluster at true-V\* = 12–14, H2 confirmed and
the diagnosis is "V_θ has weak ordering on natural-distribution-tail
states." If failures are uniform over true-V\*, H2 falsified and the
issue is something else (e.g. specific symmetry classes the net mis-
orders).

Cheap to run: `solve_histograms_beam.json` already carries the per-cell
`solve_lens` array but not the per-attempt scrambles. A small extension
to `run.py` saving the failure-state bytes (or seed+index) would let a
follow-up script index into V\* and produce a failure-vs-true-V\*
histogram. Add to `analyze.py` if H2 looks worth investigating after
the C5 results land.

### H3 — Beam at width=256 with a *better-trained* V_θ would meet the gate without going wider. Confidence: MEDIUM.

The sync500_kmax20-30k checkpoint plateaued at macro_mae 3.13 in M5;
its pred_std (1.51) is wider than baseline (1.44) but still well below
the V\* target spread (~3 across [0,14]). The ordering signal is "good
enough" but not saturated. A net trained against V\*-stratified or
curriculum-extended samples could plausibly halve the d=14 failure
rate at width=256 without touching beam parameters.

**Verify**: Out-of-scope for M6. This hypothesis is the logical link
between this sweep's outcome and the M5 forward-backlog item
"V\*-stratified resampling." If H1's wider-beam path hits a hard
ceiling, H3 becomes the leading recommendation.

### H4 — Beam search would extract more capability from a *less compressed* V_θ even at the same width. Confidence: MEDIUM-LOW.

Beam consumes ordering and is invariant to monotonic transforms of
V_θ, so compression *per se* shouldn't matter. But ordering quality is
correlated with predict spread: a net that says "all states are
between 5 and 7" has compressed mass and easily-confused close-call
ordering, where a net spanning [0, 14] with broad within-class variance
likely has clearer per-state separation. Cheap to test.

**Verify**: extra w=256 × depths 10..14 × n=200 sweep against the
cycle-1 `baseline-30k/net_final.pt` checkpoint (pred_std 1.44) for
direct comparison with this sweep's sync500_kmax20 checkpoint
(pred_std 1.51). If beam *amplifies* the cycle-3-vs-baseline gap at
deep depths beyond the gap greedy already showed, H4 supported. Cost:
~1 min of compute.

## Open questions

*(well-defined next experiments)*

- **Q1 — Where exactly does width=256 fail?** What `(scramble, true V\*)`
  pairs show up in the 6 failures at d=14, the 3 at d=12, the 2 at
  d=10..13 each? Per-cell seed is reproducible
  (`config.seed + int(depth)`), so a follow-up script can regenerate
  the same scrambles, look up V\*, and tabulate failure-vs-true-V\*.
  **Required for H2 verification.**
- **Q2 — Width-cost frontier.** What's the actual wall-time vs solve-
  rate Pareto for w ∈ {256, 512, 1024, 2048, 4096} at d=14? The H1
  extrapolation suggests w=4096 might meet the gate at d=14 in ~225s
  per cell. If so, what's the n_per_depth=1000 acceptance run time at
  the chosen width — does it fit in 30 min, an hour?
- **Q3 — Wider sweep needs `max_steps` headroom.** At w=256, d=14 the
  longest solved path was 8.91 moves average (some individual paths
  could be longer). At wider widths, beam may explore deeper before
  finding a solution; `max_steps=20` was set with +6 headroom over the
  QTM diameter 14. Check the longest individual solved path in the
  width=256 cell at d=14 to confirm headroom isn't being squeezed.
- **Q4 — The two seeded-but-still-failed scrambles per cell at d=10–13.**
  Are they the same scramble across widths? If a single hard scramble
  is failing at every width, that's a hint about either the
  failure-state distribution (H2) or a per-state path-length cap
  interacting badly with `max_steps`.

## What we haven't verified

- **The 99.9% acceptance projection from H1.** Linear-in-log4 width
  extrapolation assumed the last 4× step's slope continues. The curve
  could bend either direction past w=256: width could saturate (further
  doublings buy diminishing returns because the beam already covers
  "the right" candidates) or could open up (a wider beam unlocks paths
  that a narrower one truncates early). The M6 acceptance run is the
  empirical test — if width=256 lands closer to 0.95 at d=14 (not 0.97
  as the n=200 sweep suggests), H1's 1024–4096 estimate is too low.
- **The C3 sweep used n=200 per cell.** Binomial SE at solve_rate=0.97
  is √(0.97 × 0.03 / 200) ≈ 0.012 — so the d=14 w=256 cell's "0.970"
  is really 0.97 ± 0.024 95%-CI. The C5 run at n=1000 will tighten this
  to ±0.011. Decisions about whether width=256 "barely misses" or
  "clearly misses" the 0.999 gate should wait for n=1000.
- **The per-depth V\*-excess values at high width.** At width=256, d=10
  V\*-excess is 0.192 — averaged over 198/200 attempts. The ≤1.0 gate
  is "mean V\*-excess across all depths" (per the SPEC wording in the
  plan). All cells well under 1.0 means the gate-as-mean and
  gate-as-max are both met regardless. But H2's diagnosis depends on
  whether the *failures* would have had high V\*-excess if they had
  been solved — by definition we don't have that data.
