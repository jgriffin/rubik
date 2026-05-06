# davi-2x2

DAVI training experiments toward M5 acceptance on the 2x2 cube.

**M5 acceptance gate** (per SPEC.md):
- MAE vs `V*` < 1.0
- Greedy solve rate ≥ 99% on depth ≤ 14 (QTM diameter; V* covers all
  3,674,160 reachable states).

**Methodology — tiered hand-curated.**

Every config we ship is *earned* — picked from our own measurements, not
borrowed from prior work. There is no starting reference config in the
repo on purpose: the question of "what hyperparameters work for 2x2 DAVI
on this hardware" is what these experiments answer.

| Tier | Question                                                         | Script                       | Cost              |
|------|------------------------------------------------------------------|------------------------------|-------------------|
| 0    | What does a DAVI step *cost* across (batch, body, blocks)?       | `calibrate_step_time.py`     | one-shot, ~30 min |
| 1    | Does DAVI converge *at all* on a downscaled cube?                | `run.py` w/ tiny configs     | 5–10 min/run      |
| 2    | Single-axis sweeps at mid-scale — LR / sync / curriculum / batch | `run.py` w/ swept configs    | ~5 min/run        |
| 3    | Champion full-scale run informed by tiers 0–2.                   | `run.py` w/ tier-3 config    | hours             |

Without tier 0 every downstream budget is blind. So tier 0 ships first.

---

## Tier 0 — step time calibration

**Status:** harness landed. Calibration run pending.

**Script:** [`calibrate_step_time.py`](calibrate_step_time.py).

**Grid (initial; refine after first pass):**
- `batch_size`: TBD
- `body_widths`: TBD
- `n_residual_blocks`: TBD

**Acceptance for tier 0:** one JSONL row per cell with `step_ms_median`,
`step_ms_ci_low`, `step_ms_ci_high`, `n_params`. Numbers feed tier 1 +
tier 2 budget decisions.

(table to be appended once the calibration runs)

---

## Tier 1 — sanity runs (downscaled)

**Status:** pending tier 0.

Tiny net (`body_widths=(256, 64)`, `n_residual_blocks=2`) on a depth ≤ 5
curriculum. Goal: DAVI loss curve reaches near-zero on a problem small
enough to be obviously learnable. If it doesn't, no amount of tier 2/3
tuning will help.

---

## Tier 2 — single-axis sweeps (mid-scale)

**Status:** pending tier 1.

One axis at a time, fixed budget per cell, on a mid-scale net. Axes:
`learning_rate`, `target_sync_interval`, `max_scramble_depth`,
`batch_size`. Goal: identify working regimes per axis without compounding
changes (each cell starts from the same baseline so attribution is
clean).

---

## Tier 3 — champion run

**Status:** pending tier 2.

Single full-scale run informed by tier 2 winners. Reports M5 acceptance
gate evaluation (MAE vs V*, greedy solve rate by depth).

---

## Baseline-30k run rationale

This is the first end-to-end DAVI training run on the 2x2. The
V*-supervised detour (everything that lived under `t1-capacity/` and
`v-star-weighted/`) has been archived to branch
`m5-davi-vstar-supervised-archived`; this directory restarts fresh on
the actual DAVI loop. V* is the eval oracle here, never the training
signal.

### Question

Does DAVI on a comfortable network with first-try-defensible
hyperparameters reach M5 acceptance — `macro_mae < 1.0` AND
`greedy_solve > 99% at d ≤ 14` — in 30k steps?

If yes, we have a concrete config to do the M5 close on. If no, the
failure shape (loss diverges / wavefront stalls at low depth / value
collapses to mean / etc.) tells us which axis to sweep first.

### What's an informed pick vs. a placeholder

- **`max_scramble_depth: 18`** — user-directed. QTM God's number on the
  2x2 is 14 (verified empirically by the M5 V\* enumerator across all
  3,674,160 reachable states); sampling out to 18 means the curriculum
  reaches every depth bucket plus four extra buckets where the random
  walk wanders past optimality. The state distribution past depth 14 is
  drawn from the same 3.6M states as depths ≤ 14 (just via longer
  walks), so this is harmless surplus, not extension.
- **`body_widths: [4096, 1024]`, `n_residual_blocks: 4`,
  `normalization: bn`** — comfortable-by-construction. Per the user's
  steer of "I just hate to spend time on a network that's too small
  while we're still proving things out," this lands at ~17M params
  (input 144 → 4096 → 1024 → 4 × residual(1024,1024) → 1, body depth ~12
  layers). The V\*-supervised work showed BN was innocent at fixed
  sampler, even slightly helpful, so BN stays. Going wider/deeper than
  the prior placeholder `(4096, 1024) n=2` (~9M params) is intentional
  spare capacity for a first-pass run.
- **`batch_size: 4096`** — first-try-defensible. User flagged 1024 felt
  narrow. M4's batch sensitivity sweep found `apply_moves` saturation
  onset at B=4096–32768; at B=4096 we're at the start of the saturation
  knee, well into pipelined-throughput territory. 4× the prior
  placeholder. Not yet earned by DAVI-side measurement; if step time
  gates progress we revisit.
- **`learning_rate: 0.001`** — Adam framework default. Not a
  cube-literature value. Adam's default is robust across MLP regression
  problems out of the box; we keep the framework default rather than
  tune blind.
- **`target_sync_interval: 500`** — **earned by cycle 3** (cycle-3
  comparison sync500 vs sync1000 at K_max=20). At N=200 post-hoc, the
  sync500 cell strictly dominates the sync1000 cell at every test depth
  (~+5pt absolute on d11/d13). Smaller intervals (sync100 in cycle 2)
  destabilize from a converged state. Larger (sync2000 in cycle 2;
  sync1000 in cycle 3) tighten *calibration* but not *capability*.
  500 is the project default for fresh-start DAVI runs going forward
  unless something specific changes the regime.
- **`n_steps: 30000`** — wall-clock budget. Long enough to see a
  wavefront propagate from solved outward (30k optimizer steps with
  target sync every 500 means ~60 target generations). Short enough to
  fit in a few hours. Not earned; if the run is converging at 30k we
  extend, if it's plateaued well before then we shorten future runs.
- **`seed: 0`, `device: mps`, `log_every: 100`, `eval_every: 1000`,
  `checkpoint_every: 5000`** — bookkeeping defaults.

The earned values are: `max_scramble_depth` (user-directed, justified
by 2x2 QTM diameter); `normalization: bn` (V\*-supervised work
established as innocent → keep). Everything else is a defensible first
try, with the understanding that the post-run intuition section calls
out which knob the data points to next.

### Acceptance + plan

- **Pass:** final-eval `macro_mae < 1.0` AND every
  `solve_rate_d{d} > 0.99` at the test depth grid {1..13} (contiguous
  since the eval.py change in commit `2a0e95f` — was odd-only before).
  Per M5 SPEC.
- **Informative-fail shapes worth writing up** (each is a well-shaped
  next experiment):
  - macro-MAE plateaus while train loss keeps falling → target_sync or
    curriculum is the bottleneck.
  - Greedy solve flat at small depths only → wavefront propagated from
    solved but didn't reach the bulk; curriculum cap or depth
    weighting needs attention.
  - Train loss diverges → LR / batch / sync interval combination
    unstable for DAVI; tighten one.
  - Per-depth MAE follows the depth distribution (high MAE at modal
    depths, low at tails) → predict-the-mean collapse, this time
    inside DAVI.
- **Writeup:** `results/intuition.md` is hand-written per project
  convention (Observations → Hypotheses with confidence + verification
  plan → Open questions). `analysis/analyze.py` regenerates
  `results/results.md` with curves + tables and appends the intuition
  section verbatim.

### Files

- `configs/baseline.yaml` — DAVIConfig YAML with the values above.
- `run.sh` — one-line wrapper invoking `run.py`.
- `analysis/analyze.py` — reads `runs/baseline-30k/metrics.jsonl`,
  writes `results/results.md` with train-loss / macro-MAE / per-depth /
  solve-rate tables, and appends `results/intuition.md`.

---

## Intuition

(Hand-written in `results/intuition.md` once tier 0 has data; appended
by an analyzer in a later commit. Per project convention: Observations
→ Hypotheses w/ confidence + evidence + verification plan → Open
questions.)
