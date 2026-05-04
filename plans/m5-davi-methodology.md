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

### T1 — what does the network need to fit V*?

**Original framing assumed gradient signal would exist and capacity was the first question.** The autonomous investigation (`experiments/davi-2x2/t1-capacity/intuition.md`, commits `6337396` → `72ce969`) showed gradient signal *itself* is the upstream blocker: V\*'s peaked depth distribution (83% of states at depths 10–12, 0.05% at 0–4) under uniform-with-replacement sampling collapses every cell tested (capacity 23×, normalization {bn,none}, residual depth, LR, loss formulation, training-set size 1k→2.94M) to predict-the-mean. Depth-balanced sampling escapes the trap immediately. So T1 splits into three questions that must land in order:

#### T1a — sampling [DONE]

**Q.** Does depth-balanced sampling (with-replacement, ~equal samples per depth bucket per batch) produce per-depth predictions that track ground truth across 0–14?

**A.** Yes — at `(4096, 1024)` n=2 BN, batch 1024, 7k steps, pred_std jumped 0.35 → 1.13 (target 1.16). Per-depth predictions track truth (depth 3 → 4.06, depth 13 → 11.32). BN/no-BN are near-equivalent under this sampler; LR and loss-formulation are not bottlenecks. Logged in `t1-capacity/intuition.md` §"BN ablation + autonomous investigation".

**Caveat carried into T1c.** With-replacement sampling on tiny tail buckets (depth 0 has 1 train state, depth 1 has 9) means the model sees ~68 repeats of the same 1–9 states per batch. Whether this overfits the tails at longer budgets is the first thing to verify (next experiment: 30k step `dbal_4096x1024_n2`).

#### T1b — evaluation metric [DONE]

**Q.** Under depth-balanced sampling, uniformly-weighted val-MAE rewards the bulk-collapse strategy T1a deliberately walks away from (val is bulk-dominated by construction — same V\* distribution as train). What's the right metric?

**A.** **Macro-MAE = uniform mean across per-depth MAEs.** Treats each depth equally regardless of frequency in val. A model that predicts 11 for everything gets macro-MAE ~3.7 (penalized for the tails it's wrong on) where uniform val-MAE is ~0.93 (rewarded by bulk overlap). Implemented additively in `supervised.py`'s `_eval_val_mae` — uniform val-MAE stays in metrics for backward comparability; per-depth-MAE table also logged so the per-depth picture isn't reduced to one number.

**Cross-experiment retroactive comparison.** macro-MAE is a function of `(preds, val_depths)`, not of training. Phase A and earlier cells can be re-evaluated by reloading checkpoints and recomputing on the same val set. Cheap follow-up; not load-bearing for T1c.

#### T1c — capacity floor under fixed sampling + metric [OPEN]

**Q.** Now that gradient signal flows uniformly across depths and the metric isn't gameable by bulk-collapse, what's the smallest network that hits **macro-MAE < 0.5**?

**Pre-requisite.** The 30k-step `dbal_4096x1024_n2` run must drive macro-MAE below 0.5. H4 predicts this. If macro-MAE plateaus above 0.5 instead, the with-replacement-on-tiny-buckets caveat from T1a needs investigation (likely candidates: floor-cap on bucket repetition; weighted loss instead of resampling; oversample-without-replacement up to bucket size, then with-replacement past it) before sweeping cells.

**Method (assumes pre-requisite holds).** The original widths-then-residuals two-phase plan moves here verbatim — only the closing metric (macro-MAE < 0.5 vs val-MAE < 0.5) and the sampler (`depth_balanced` not `uniform`) change. Bundling width and residual-block count would confound two separable questions ("is the body wide enough?" and "do residuals earn their keep at sufficient width?"); we keep them separate.

##### Phase A — width sweep (fix `n_residual_blocks=0`, vary widths)

Walk down from "embarrassingly large" to "near input dim." All cells use plain MLP body — no residuals. The only first-principles floor on `h1` is `h1 ≥ input_dim = 144` (otherwise the first projection compresses information). Cells:

| h1 × h2     | h1/input | role |
|-------------|---------:|------|
| 8192 × 2048 |     56.9× | "should clearly succeed" — anchor from above |
| 4096 × 1024 |     28.4× | half the anchor (T1a verified at n=2) |
| 2048 × 1024 |     14.2× | quarter, h2 unchanged from previous |
| 2048 × 512  |     14.2× | h1 fixed, half h2 — does h2 matter at fixed h1? |
| 1024 × 512  |      7.1× | smaller still |
| 512 × 256   |      3.6× | smallest "meaningfully wider than input" |

All at `n_residual_blocks=0`, `sampler: depth_balanced`, `normalization: bn` (T1a result: BN slightly helpful under this sampler), batch 1024, fixed seed, step count = whatever the 30k pre-requisite established as "sufficient for convergence at the anchor" (likely 30k unless the 30k run still shows a falling curve at the end). LR 1e-3 (T1a result: not a bottleneck). Top-end widths verified against T0 step-time data before commit.

**Phase A closing condition.** Pick the smallest cell with `macro_mae < 0.5`. Call it `(h1*, h2*)`. If multiple cells succeed, smallest one is the answer. If only the top cell succeeds, that's the answer (and Phase B operates there). If nothing succeeds at `(8192, 2048)`, the failure isn't capacity — and the methodology has *another* upstream miss to find (it would mean T1a's sampler buys depth-coverage but not enough fitting power; investigate before extending Phase B).

##### Phase B — residual sweep (fix widths at `(h1*, h2*)`, vary residual count)

Only runs if Phase A found a passing cell. The question: *do residual blocks at the chosen width buy us anything for supervised V\* regression under depth-balanced sampling?* Cells: `n_residual_blocks ∈ {0 (reused from Phase A), 1, 2, 4}`. Three new runs at same batch/seed/steps/LR/sampler/normalization as Phase A.

**Phase B closing condition.** If macro-MAE flat across all four `n`, residual blocks do nothing here and **we ship `n=0`** — the simplest model that works. If macro-MAE improves with `n`, ship the smallest `n` capturing most of the gain. The answer to T1 is now `(h1*, h2*, n*)` plus the closed-form choices `sampler*=depth_balanced`, `normalization*=bn`.

#### T1 outputs

- T1a closed: `experiments/davi-2x2/t1-capacity/intuition.md` §"BN ablation + autonomous investigation". Sampler toggle in `src/rubik/training/...` and `supervised.py`.
- T1b closed: macro-MAE + per-depth-MAE in `supervised.py`'s `_eval_val_mae`. `metrics.jsonl` schema gains `macro_mae` (float) and `per_depth_mae` (dict-of-float keyed by depth) on `event:"eval"`. Uniform val-MAE retained.
- T1c outputs (pending verification + sweep):
  - `experiments/davi-2x2/t1-capacity/supervised.py` — thin trainer (no DAVI, configurable loss/sampler/normalization, Adam). Reuses `ValueNet` unchanged.
  - Per-cell `runs/<cell>/metrics.jsonl` via `MetricLogger`.
  - `results.md` with both phases' tables (macro-MAE) + a Pareto-style frontier plot.
  - `_picks.json` with `(h1*, h2*, n*, sampler*, normalization*)` tuple — the **T1 architecture**. Cache the next size up too as **T1 comfortable** for downstream tiers.
  - `intuition.md`: did widths scale how we expected under depth-balanced? Did residuals help (Phase B)? Where exactly is the floor?

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
