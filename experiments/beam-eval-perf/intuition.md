# beam-eval-perf — intuition

_Datestamp: 2026-05-07. Written after the first sweep
(`sweep_full_train_final`, widths {8,16,32,64,128,256,512},
n_per_depth=100, ~8 min total wall on M4 Max / MPS, seed=0)._

## Observations

_(mechanical, from this run's data)_

- **Total sweep wall ≈ 8 minutes for 7 widths.** Far cheaper than the
  block estimate of 30–60 min. Sweeps are now in "casual diagnostic"
  cost territory, not "overnight job."
- **Width=8 and width=16 take identical wall time (30.6s each).** The
  per-step beam compute at these widths fits inside fixed scheduling
  overhead — doubling the work is free up to some inflection point.
- **Width=64 → 128 is nearly free (1.05× wall) for +15pp at d=14.**
  The single most efficient doubling in the sweep. After 128 the wall
  cost grows roughly with width (1.31×, 1.72× for the next two
  doublings).
- **Capability discrimination lives entirely at d=11..14.** d=1..9
  saturates at 1.00 across all widths; d=10 saturates at width=64.
  Anything cheaper than width=64 throws away the d=10 signal too.
- **d=14 capability climbs monotonically with width:** 0.19, 0.26,
  0.40, 0.49, 0.64, 0.71, 0.79. Each doubling adds capability; the
  slope per `log2(width)` is roughly +0.10 across the sweep.
- **avg_solve_len at d=14 climbs slightly with width:** 12.84 (w=8) →
  13.67 (w=512). Wider beams find more d=14 walks AND those additional
  walks tend to have longer solves. Makes sense — easy d=14 walks
  (short optimal solve) get found at every width; hard ones (longer
  optimal solve) require more beam width to keep the right candidate
  alive.
- **This run's d=14 at width=256 is 0.71, vs the prior baseline 0.78
  on the same checkpoint.** 7pp gap at n=100 ≈ 1.5σ — within plausible
  walk-sample variance. The two runs use different random walk
  generators (this one is seed=0 explicit; the prior post-run eval
  used the script's default behavior).
- **Width=512 lands at d=14 = 0.79 — matches the prior 0.78 baseline
  within 1pp.** The two seeds happen to disagree at width=256 but
  agree at width=512. Plausible interpretation: at higher widths the
  solve-rate stabilizes earlier (fewer "marginal-walk" cells in the
  binomial denominator), so seed-to-seed variance shrinks.

## Hypotheses

_(interpretive — confidence + evidence + verification plan)_

### H1 — The width=64 → 128 "free doubling" is a hardware sweet-spot artifact, not a property of beam search. Confidence: MEDIUM.

The wall ratio sequence 1.00, 1.59, 1.34, **1.05**, 1.31, 1.72 has a
clear minimum at the 64 → 128 step. A pure-compute view of beam search
would predict roughly linear wall growth with width (the per-state
forward pass scales with batch_size = beam_width × n_moves, modulo
batched-MPS efficiency curves). The dip at 64 → 128 is consistent with
beam_width × n_moves = 128 × 12 = 1536 fitting one MPS launch where
beam_width × n_moves = 64 × 12 = 768 fits the same launch with 50%
idle SIMD lanes. If true, the sweet spot would shift on different
hardware (Apple M3, M2, or non-MPS backends).

**Verify:** repeat the sweep on a different machine (or with `--device
cpu` for a contrast point) and check whether the 64 → 128 dip
reproduces. Cheap.

### H2 — The d=14 = 0.71 vs 0.78 discrepancy at width=256 is walk-sample variance, not a real ordering shift. Confidence: HIGH.

Binomial SE at p=0.7, n=100 is √(0.7 × 0.3 / 100) ≈ 0.046, so the gap
between 0.71 and 0.78 is ~1.5σ. The two runs draw walks from
independent generators (this one explicit seed=0; the prior's
generator state was not pinned). The fact that width=512 in this run
matches the prior 0.79 baseline within 1pp — and that the per-walk-depth
× per-width matrix shows the expected monotone shape — argues against
a real regression in checkpoint behavior between the two runs.

**Verify:** rerun the prior post_run_beam_eval pathway with explicit
`generator=torch.Generator().manual_seed(0)` and check whether width=256
lands at 0.71 (matches this sweep). Or re-run this sweep at multiple
seeds (5–10 seeds × n=100 each) and compute a real CI. Cheap (~1 hr).

### H3 — The "screening / production / deep-diagnostic" tier structure generalizes across checkpoints. Confidence: MEDIUM-LOW.

I assigned width=64 / 256 / 512 as the three operational tiers based
on the d=10 / d=12 / d=14 saturation thresholds for THIS checkpoint.
On a checkpoint with worse deep capability (e.g. warm_continue
net_final at d=14 = 0.24 / d=12 = 0.79 per LOG), d=10 might saturate
later (width=128?) and the screening tier might need to shift up.
Conversely, on a much better checkpoint (hypothetical: d=14 ≥ 0.95),
width=128 might suffice for production and width=64 might not be
enough screening signal.

**Verify:** sweep on `warm_continue/net_step_120000.pt` (the LOG's
collapse endpoint) and on `full_train/net_step_5000.pt` (early-training
checkpoint). If the saturation depths scale with checkpoint quality —
i.e. the curve shape is preserved but the saturation knee shifts — the
tier mapping is checkpoint-specific and the production tier
recommendation needs a calibration step per major checkpoint. If the
saturation depths are roughly checkpoint-invariant, the tier mapping
is universal and we lock it in. ~25 min wall for two more checkpoints.

### H4 — At larger widths (≥1024), the d=14 curve will bend (saturate) before reaching 1.00. Confidence: LOW.

The slope-per-`log2(width)` is steady around +0.10 in the sweep but
the curve from width=256 → 512 (+0.08) is slightly shallower than
width=128 → 256 (+0.07) and width=64 → 128 (+0.15) — too noisy to
declare a trend. The natural interpretation: for a fixed network, beam
search eventually exhausts the "ordering signal can find the right
state" ceiling and additional width buys nothing because the network
genuinely can't distinguish the right candidate from a wrong one.

**Verify:** extend the sweep to {1024, 2048} on the same checkpoint.
Width=1024 ≈ 270s wall, width=2048 ≈ 460s — call it 12 min total. If
d=14 climbs to 0.85+, slope persists; if it lands ≤0.82, ceiling is
imminent and "raise checkpoint quality" beats "widen the beam."

## Open questions

_(well-defined next experiments)_

- **Q1 — Multi-seed CI for the d=14 cells.** What's the actual 95% CI
  on solve@d=14 at each width? Re-run the sweep with seeds 0..9 (10
  re-runs ≈ 80 min wall) and compute mean + sample SD per cell. Would
  let us state cell-comparison verdicts with calibrated confidence
  rather than hand-waving "1.5σ."
- **Q2 — Does the 64 → 128 sweet spot shift across checkpoints?** Run
  the same width sweep on 2–3 other checkpoints (warm_continue final,
  full_train step 5000, full_train step 10000) and check whether the
  wall ratio curve has the same minimum at 64 → 128.
- **Q3 — How does this sweep scale on Apple M3 / a different
  generation?** Out of scope here, but the H1 sweet-spot hypothesis is
  testable if/when the project gets data from another machine.
- **Q4 — `--include-v-star` worth enabling at smaller widths?** V*=1..6
  saturates at every width on the current 3x3 model, but if the
  bounded oracle extends to K=8 the deep V* layers might show the same
  width-bound erosion as deep walk-depth. Would tell us whether
  "deeper V*" and "deeper walks" carry the same ordering signal.

## What we haven't verified

- **The "states/s" column** treats `n_per_depth × len(walk_depths) =
  1400` as the work measure. That's the count of root scrambles, not
  the count of (state, action) evaluations the beam actually scored —
  the latter scales with the depth-budget × beam_width per scramble.
  The states/s ratios are useful for relative comparison across widths
  but should NOT be read as raw model-throughput numbers; the right
  throughput metric would be `total_states_scored / wall`, which the
  beam helpers don't currently emit. (`scripts/beam_eval_run.py` has a
  `states_scored` field but it's the same `1400`. Future improvement:
  thread the actual node count through `beam_eval_walk`'s return.)
- **Whether the production-tier recommendation generalizes to 3x3
  models with different architectures** (e.g. the smaller_net
  comparison run still on the bench). Different arch → different
  forward-pass cost → different absolute walls but possibly the same
  qualitative shape. Re-validate when the smaller_net comparison
  lands.
- **The pre-committed `d=14 ≥ 0.75` gate decision is a one-cell
  gate** based on the n=100 number for ONE seed. With Q1's multi-seed
  CI in hand, the gate should be restated as "lower 95% CI bound at
  d=14 ≥ 0.75" — a more defensible operational criterion.
