# M5 — DAVI experimentation methodology

## Purpose

This is the **meta-plan** for the rest of M5. It does not specify hyperparameter values — it specifies the *process* by which we discover them. Per project convention (`CLAUDE.md`: "earn every hyperparameter — do not borrow"), we have committed scaffolding (env, oracle, ValueNet, DAVI loop, MetricLogger, run.py, T0 step-time calibration) but **no defaults** for `body_widths`, `n_residual_blocks`, `learning_rate`, `batch_size`, `target_sync_interval`, `max_scramble_depth`, or `n_steps`. This plan describes how we close each of those questions.

The tiered approach extends the M4 perf methodology: each tier asks one yes/no or "what value of X" question, opens with `intuition.md` stating *why* and *what we expect*, closes with answers committed before the next tier opens.

## Guiding principles

- **One question per tier.** Tier scope is the question, not the artifact. "Tier 1 is yes/no, don't expand" — when a tier surprises you, question the framing rather than widening the investigation inside it.
- **V\* as instrument, never as gradient.** V* (`src/rubik/oracle/v_star_2x2.py`) is used for capacity probes (T1), eval MAE (all tiers), and post-hoc analysis. It is **never** mixed into the DAVI loss. The DAVI training path stays identical to what would run on 3x3.
- **No borrowed values.** Every numeric choice in a config YAML traces back to a tier whose `intuition.md` justifies it. If a value appears with no tier behind it, that's a bug in the methodology.
- **Static-preview discipline.** Loss/MAE-vs-step curves rendered to `experiments/davi-2x2/<tier-slug>/plots/*.html` (matplotlib avoided per project convention; use Plotly or hand-rolled SVG). Eyeball the plot in a browser; commit if right.
- **Branch per tier, commit per variant.** Each tier opens a `cc-process` block on its own branch (`m5-tier1-capacity`, `m5-tier2-lr`, …). Variants within a tier are individual commits on that branch. The closing commit of a tier merges to `m5-davi` with the answer in `results.md`.

## Pre-tier deliverable: `eval_set.npz`

Before T1 opens, generate and freeze the universal eval set. This is the X-axis every downstream tier shares.

- **Spec.** Depth-stratified: 100 canonical states per depth d ∈ {1, 2, …, 14}. Sampled with a fixed seed via reservoir sampling over canonical orbits at each BFS layer.
- **Storage.** `experiments/davi-2x2/eval_set.npz` — `states` (1400, 24) int8, `depths` (1400,) int8, `seed` (scalar). Committed to git (~35 KB).
- **Loader.** `experiments/davi-2x2/eval.py` — `load_eval_set() -> (states, depths, v_star_targets)`. Loads V* lazily from cache, computes targets once.
- **Usage.** Every run (T1 onwards) calls eval at fixed step counts (e.g. every 1000 steps), logs `eval/mae_overall` and `eval/mae_at_depth_{d}` to JSONL. `analyze.py` reads these.
- **Why fixed-and-frozen.** Comparable cross-experiment metrics. Run A's eval-MAE-at-step-5000 means the same thing as run B's because they hit the same states.

## Tier sequence

### T0 — step-time budget calibration *(DONE — commit `357f466`)*

Question: *what's the per-step wall-time at various (batch_size, network_shape) points on M4 Max MPS?*

Output: `experiments/davi-2x2/calibrate_step_time.py` + a step-time grid in `results.md`. Sets the wall-time budget for sizing later tiers.

### T1 — capacity floor (supervised V* regression)

**Question.** What's the minimum `(h1, h2)` body width — and how many (if any) residual blocks at that width — needed to fit V* on the 2x2 to val-MAE < 0.5 with direct supervised regression?

**Why.** Decouples capacity from training dynamics. If the smallest sufficient network is known, then any failure of DAVI with that network is a *training-dynamics* problem, not a *capacity* problem — that separation is leverage for every later tier.

**Two phases — widths first, residuals second.** Bundling width and residual-block count into one grid confounds two questions: "is the network wide enough to fit V*?" and "do residual blocks earn their keep at sufficient width?" These are separable; we keep them separate.

#### Phase A — width sweep (fix `n_residual_blocks=0`, vary widths)

Walk down from "embarrassingly large" to "near input dim." All cells use plain MLP body — no residuals. The only first-principles floor on `h1` is `h1 ≥ input_dim = 144` (otherwise the first projection compresses information). Cells:

| h1 × h2     | h1/input | role |
|-------------|---------:|------|
| 8192 × 2048 |     56.9× | "should clearly succeed" — anchor from above |
| 4096 × 1024 |     28.4× | half the anchor |
| 2048 × 1024 |     14.2× | quarter, h2 unchanged from previous |
| 2048 × 512  |     14.2× | h1 fixed, half h2 — does h2 matter at fixed h1? |
| 1024 × 512  |      7.1× | smaller still |
| 512 × 256   |      3.6× | smallest "meaningfully wider than input" |

All at `n_residual_blocks=0`, batch 1024, fixed seed, 7000 steps, LR 1e-3 (placeholder — T1 isn't tuning LR, just reading capacity). Top-end widths verified against T0 calibration data before commit (extrapolation from T0's `(5000, 1000) × 4 blocks` measured grid).

**Phase A closing condition.** Pick the smallest cell with `val_mae < 0.5`. Call it `(h1*, h2*)`. If multiple cells succeed, smallest one is the answer. If only the top cell succeeds, that's the answer (and Phase B operates there). If nothing succeeds at `(8192, 2048)`, the failure isn't capacity — it's optimization, batch-norm, eval-loop, or data — and we stop and debug instead of going to Phase B.

#### Phase B — residual sweep (fix widths at `(h1*, h2*)`, vary residual count)

Only runs if Phase A found a passing cell. The question: *do residual blocks at the chosen width buy us anything for plain supervised V* regression?* Cells:

| h1* × h2* | n_residual_blocks | role |
|-----------|-------------------|------|
| h1* × h2* | 0 | already done in Phase A — reused |
| h1* × h2* | 1 | one block |
| h1* × h2* | 2 | two blocks |
| h1* × h2* | 4 | four blocks (geometric spread, not because anyone published this) |

All same batch/seed/steps/LR as Phase A. Three new runs.

**Phase B closing condition.** If `val_mae` flat across all four `n` values, residual blocks do nothing for this problem and **we ship `n=0`** — the simplest model that works. If `val_mae` improves with `n`, ship the smallest `n` that captures most of the gain (knee-of-curve). Either way, the answer to T1 is now a specific `(h1*, h2*, n*)` tuple.

#### T1 outputs

- `experiments/davi-2x2/t1-capacity/supervised.py` — thin trainer (no DAVI, MSE loss, Adam). Reuses `ValueNet` unchanged.
- Training data sampled uniformly from the 3.67M canonical V* keys, seed-pinned 80/20 train/val split — held-out val ≠ the depth-stratified eval set, since this tier is asking a capacity question and depth-stratification could mask it.
- Per-cell `runs/<cell>/log.jsonl` via `MetricLogger`.
- `results.md` with both phases' tables + a Pareto-style frontier plot.
- `_picks.json` with the chosen `(h1*, h2*, n*)` tuple — call this the **T1 architecture**. Cache the next size up too as the **T1 comfortable** for tiers that need headroom.
- `intuition.md`: did widths scale how we expected? Did residuals help (Phase B)? Where exactly is the floor?

### T2 — LR range test

**Question.** What learning rate is in the live zone for DAVI training on the T1-comfortable network?

**Why.** LR is the most universally-tuned hyperparameter and the cheapest to probe. Pinning it before T3 means the T3 sweep isn't fighting an LR pathology.

**Method.**
- Smith-style range test inside the real DAVI loop. Network = T1-comfortable. Curriculum pinned to a sane middle (`max_scramble_depth=7`, balanced per-depth — half the 2x2 diameter, neither trivial nor saturating). Sync interval pinned reasonably (e.g. 200 steps — refined in T3).
- Single config, exponential LR ramp from 1e-6 → 1e-1 over 1000 steps. Log loss and grad-norm per step.
- Plot loss-vs-LR (X-log, Y-loss) — pick the LR at the "elbow" (steepest loss descent before divergence). Also note where grad-norm explodes — that's the LR ceiling.

**Closing condition.** A single chosen LR (call it **T2-LR**) plus a 3x range around it (e.g. 0.3× and 3×) reserved for T3 robustness checks if needed.

**Output.** `t2-lr/results.md` with the range-test plot; `intuition.md`: where's the elbow, how wide is the live zone, is grad-norm telling us anything that loss isn't?

### T3 — DAVI dynamics (joint sweep)

**Question.** What `(target_sync_interval, max_scramble_depth)` combination produces a DAVI loop that converges to eval MAE < 1.0 on the T1-comfortable network at T2-LR?

**Why.** These two hyperparameters control the bootstrap target's stability. They interact — an aggressive curriculum (deeper scrambles) only works with a sufficiently slow target sync, and vice versa. Sequential one-at-a-time would miss interactions; joint surfaces them.

**Method.**
- Joint 2D grid. Initial proposal:

  | target_sync_interval | max_scramble_depth = 4 | 7 | 11 | 14 |
  |---------------------:|:----------------------:|:--:|:--:|:--:|
  | 50  | ✓ | ✓ | ✓ | ✓ |
  | 200 | ✓ | ✓ | ✓ | ✓ |
  | 1000| ✓ | ✓ | ✓ | ✓ |
  | 5000| ✓ | ✓ | ✓ | ✓ |

  16 cells. Each runs for 30k steps (T0 budget tells us wall-time per cell — confirm this fits in a session before launching).
- Curriculum shape pinned to **balanced per-depth slicing** (B/max_depth states per depth) — already in `generate_adi_batch`. Curriculum *shape* (fixed vs ramp vs sampled) is deferred; this tier varies depth ceiling only.
- Eval every 1000 steps against `eval_set.npz`; log per-depth MAE.

**Closing condition.** A heatmap of final eval-MAE per cell + a per-cell loss curve grid. Identify:
- The "live zone" — connected region of (sync, depth) cells where MAE < 1.0.
- The "fastest convergence" cell — lowest MAE per wall-clock.
- Any pathology cells (oscillating loss, MAE plateau above 1.0, grad blowup).

Pick a **T3 champion** (sync, depth) and a **T3 neighbor** (one cell away, similar performance) — having two configurations carry into T4 protects against the champion being a single-seed lucky run.

**Output.** `t3-dynamics/results.md` with heatmap + loss grid; `intuition.md`: which axis dominated? Where's the boundary? Any surprising interactions?

### T4 — stopping signals

**Question.** What early indicator (which step, which metric, which threshold) at step k predicts whether a config will reach eval MAE < 1.0 by step 30k?

**Why.** T1–T3 ran every config to completion. From T5 onwards we want to **kill bad configs early** — but only if we know what early-bad looks like. This tier calibrates that kill criterion against the T3 results we already have, so we don't pay for new runs.

**Method.**
- Reuse T3's per-cell logs. For each cell, label the run "succeeded" (final MAE < 1.0) or "failed".
- Compute candidate early signals at step k ∈ {1k, 2k, 5k, 10k}: (a) train-loss absolute, (b) train-loss slope, (c) eval-MAE absolute, (d) eval-MAE-at-depth-1 (shallow first?), (e) gradient norm.
- For each (signal, step k, threshold) triple, compute confusion matrix vs the success label. Pick the earliest-step signal with high precision (few false kills) at acceptable recall.

**Closing condition.** A documented kill rule of the form "if `eval/mae_at_depth_1` > X at step Y, abort". Implement as `experiments/davi-2x2/early_stop.py` — a hook the runner can call.

**Output.** `t4-stopping/results.md` with confusion matrices per candidate rule; `intuition.md`: did shallow-first-learning emerge as predicted? How many wall-hours does the kill rule save us in T5?

### T5 — duration + scaling

**Question.** Given the T3 champion config, what `n_steps` reaches the M5 acceptance criteria (loss decreases monotonically over 100k, MAE vs V* < 1.0, greedy solve rate ≥ 99% at depth ≤ 11)? And: does scaling network size beyond T1-comfortable still buy MAE improvement, or are we at diminishing returns?

**Why.** This is the tier where M5 actually ships — every prior tier was instrumentation. By here we have a config we trust and a kill rule that protects against waste.

**Method.**
- Run the T3 champion at three durations: 30k (sanity check, should match T3), 100k (acceptance), 300k (over-train check).
- Independently, run T3 champion + 2 larger network sizes (T1-comfortable × 1.5, × 2) at 100k.
- Greedy solve eval added: per scramble depth 1..11, 100 random scrambles, follow argmin V_θ child until solved or depth-budget exhausted.

**Closing condition.** M5 acceptance gate met (per `plans/m5-davi.md`). Final config YAML committed to `experiments/davi-2x2/configs/m5-final.yaml` with full provenance: every value cross-references the tier that earned it.

**Output.** `t5-final/results.md` with acceptance numbers; `intuition.md`: did over-training help? Did over-provisioning the network help? What's the M6 baseline?

## Per-experiment template

Each tier's directory follows the same shape so future agents (and future-you) can navigate uniformly:

```
experiments/davi-2x2/<tier-slug>/
├── README.md           # one paragraph: the question, the answer (filled at close)
├── intuition.md        # hand-written: why, expectations, observations, hypotheses, open questions
├── configs/
│   └── <variant>.yaml  # one DAVIConfig per variant — every field explicit
├── run.sh              # reproducible launcher: loops configs/, calls run.py
├── runs/               # gitignored; per-variant JSONL + checkpoints
├── analyze.py          # reads runs/, produces results.md tables + plots/
├── plots/              # generated HTML/SVG, eyeballable via `open`
└── results.md          # generated table + summary; intuition.md appended at bottom
```

`intuition.md` structure (per project convention from M4):

```markdown
# <tier-slug> intuition

## Why we ran this
<1–3 paragraphs: the question this tier answers and why the framing is yes/no>

## What we expected
<predictions before running — pinning these forces honesty>

---
*(everything below filled after runs complete)*

## Observations
<mechanical, from this run's data>

## Hypotheses
<interpretive claims with confidence + supporting evidence + verification plan>

## Open questions
<well-defined next experiments; these feed the next tier's framing>

## What we haven't verified
<caveats on the most speculative claims>
```

## Agent boundaries

Agentic intelligence between tiers, never within. Concretely:

- **Within a tier.** No agent. The user runs the variants (or a deterministic `run.sh` does), `analyze.py` aggregates, the user writes `intuition.md`'s hypotheses + open questions section by hand. This is the part that builds intuition; outsourcing it is the cheat.
- **Between tiers.** After tier N closes, an agent may:
  - Read `tier-N/results.md` + `tier-N/intuition.md`.
  - Draft `tier-(N+1)/intuition.md`'s "why we ran this" + "what we expected" sections.
  - Draft `tier-(N+1)/configs/*.yaml`.
  - **The user reviews and edits before any variant runs.**
- **Observation drafting.** After variants run, an agent may draft `intuition.md`'s Observations section from logs (mechanical claims only — what the numbers say). The user writes Hypotheses, Open questions, What we haven't verified.

This seam exists because of the project's own warning ("Tier 1 is yes/no, don't expand"). Agents widening framings mid-tier is exactly the failure mode this guards against.

## Definition of M5 done

When T5 closes with the acceptance gate met, M5 ships. The deliverables:

1. `experiments/davi-2x2/configs/m5-final.yaml` — the production config, every field traced to a tier.
2. Tiered results in `experiments/davi-2x2/t1-capacity/`, `t2-lr/`, `t3-dynamics/`, `t4-stopping/`, `t5-final/` — each with `results.md` + `intuition.md`.
3. A trained checkpoint meeting the M5 gate, committed (or stored via Git LFS if size warrants).
4. `LOG.md` block closed; `m5-davi` branch merged.

The intuition documents collectively are the M5 artifact for *learning* — readable end-to-end, a future agent (or human, or 3x3 transfer effort) gets the full epistemic chain from "we knew nothing" to "here's why these values".

## Cross-references

- `SPEC.md` §M5 — acceptance gate.
- `plans/m5-davi.md` — milestone plan; this methodology is its commits-5–6 replacement.
- `plans/m4-perf-1.md` — the iteration-loop pattern this extends.
- `CLAUDE.md` — "earn every hyperparameter", "experiment results.md format", project conventions.
- `src/rubik/oracle/v_star_2x2.py` — V* enumerator.
- `experiments/davi-2x2/calibrate_step_time.py` — T0 (done).
