# davi-2x2 results

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

**Status:** harness landed (this commit). Calibration run pending.

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

## Intuition

(Hand-written in `intuition.md` once tier 0 has data; appended by an
analyzer in a later commit. Per project convention: Observations →
Hypotheses w/ confidence + evidence + verification plan → Open
questions.)
