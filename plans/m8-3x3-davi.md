# M8 — 3x3 enablement (training surface)

> **Active scope:** phases 1 + 2 only. Goal is "does the 3x3 training + eval stack work end-to-end?" — not acceptance-grade champion runs. Phase 3+ (axis sweeps → champion training → beam acceptance gates → 10M tx/sec verification) gets replanned after phase 2 lands.

## Context

The M8 env layer is ✅ (LOG 2026-05-06): `CUBE_3X3`, cubie oracle with 8 corners + 12 edges + fixed centers (U/D-axis edge-flip convention), `MOVE_PERM_3X3` snapshot, oracle↔tensor equivalence at 10k random sequences, all standard 3x3 identities green.

The training/search/beam stack is `CubeSpec`-parameterized end-to-end. P1a's audit confirmed no source-level 2x2 hardcodes blocking 3x3 use.

3x3 has **4.3 × 10¹⁹ states**. A full BFS V* oracle is impossible. This plan accepts that as a permanent constraint and scopes the early M8 training work as smoke-testing the stack. Acceptance-grade evaluation against SPEC.md's M8 criteria (greedy ≥95% on d≤15, beam=4096 100% on d=20, ≤24 mean moves, <10s/scramble, 10M tx/sec) is **deferred to a later plan extension**.

## Approach overview

Two phases. Each is a discrete cc-process block on its own branch off main. Sub-blocks within a phase stack on the parent branch.

| # | Phase | Goal | Branch |
|---|---|---|---|
| 1 | **Eval scaffold + bounded V\* oracle + lean eval API** | Stand up `experiments/davi-3x3/`; build BFS V\* up to K=6; implement three-function eval API (`value_eval` live + `beam_eval_walk` + `beam_eval_v_star` post-hoc) | `m8-3x3-eval-scaffold` |
| 2 | **Smoke training run with early-stop infra** | Hand-pick arch (T0 deferred); add sync-period early-stop to DAVIConfig; train ~5–10k steps on `rubik-3x3` wandb project; run post-training beam evals | `m8-3x3-smoke-train` |

## Phase 1 — Eval scaffold + bounded V\* oracle + lean eval API

### P1a — `experiments/davi-3x3/` skeleton — ✅ DONE (commit `9899a8f`)

Mirrored `davi-2x2/` directly. Audit found no source-level `CUBE_2X2` hardcodes blocking 3x3 use; analysis layer ports unchanged. 14 new files / 1939 insertions; analysis scripts produce graceful empty-state artifacts on empty `runs/`. Smoke checks all green; pytest 500 passed (parent's 480 reference was stale by 5 — no regression).

### P1b — Bounded BFS V\* up to K=6 — ✅ DONE (commit `11cac57`)

`src/rubik/oracle/v_star_bounded_3x3.py` mirrors `v_star_2x2`'s API shape but **skips canonicalization** — 3x3 centers are positionally fixed under QTM, raw `state.tobytes()` is canonical for this encoding. Build wall: **4.8s** (well under 30–120s plan estimate). Cache **14.9 MB** at `data/v_star_bounded_3x3_k6.npz` (gitignored per `v_star_2x2.npz` pattern). Test count 500 → 515 (+15). Per-layer counts hit the published QTM table exactly: 1 / 12 / 114 / 1068 / 10011 / 93840 / 878880 → total **983,926**.

### P1c — Lean three-function eval API

**Pivot from original plan (per 2026-05-06 conversation).** Original P1c was a 1400-state frozen `eval_set_3x3.npz` with walk-length depth bins. Reshaped per user direction to a lean three-function eval API in `experiments/davi-3x3/eval.py`. **No committed npz** — sampler is deterministic-from-seed + oracle, regenerates on demand.

**Three functions:**

#### 1. `value_eval(net, walk_set, oracle_arrays, *, ...)` — LIVE during training

Single forward pass on a fixed random-walk eval set (deterministic seed, regenerated from oracle at run start). Reports:

- **`per_walk_depth/d{1..K_walks}`** — predicted-V\* distribution stats per walk-depth bin (mean, std, optionally full histogram). The "forever metric": works at any walk-depth, no oracle needed.
- **`v_star_mae/d{1..6}`** — true MAE against oracle V\* for the d≤6 subset where the oracle has ground truth (lookup via `oracle_arrays` → state-bytes → V\*; samples without oracle entry skipped). Disappears when we go beyond K=6.
- **`macro_v_star_mae`** — mean of `v_star_mae[1..6]`. **This scalar drives early-stop.**

Cost: ~< 1s per call (forward pass only, no search). Fits in every-sync cadence.

#### 2. `beam_eval_walk(net, spec, *, n_per_depth, max_depth, beam_width, seed)` — POST-TRAINING

Random-walk states × beam search. Per-walk-depth solve rate + mean solution length. **The forever capability eval** — scales to any depth without an oracle. Greedy is recovered by `beam_width=1` (no separate greedy implementation).

#### 3. `beam_eval_v_star(net, spec, oracle_arrays, *, n_per_layer, beam_width)` — POST-TRAINING

V\*-stratified states (sampled per V\*-layer from oracle) × beam search. Per-V\* solve rate + mean solution length + per-V\* prediction MAE. **Temporary ground-truth eval** (d≤6 only, while oracle exists).

#### 4. `scripts/post_run_beam_eval.py` — post-hoc trajectory

Given a run dir, iterates `checkpoints/*.pt`, runs both beam evals on each, writes trajectory JSON + HTML for retrospective "beam-solve-rate vs training-step" charts. Optional, on-demand. Means we can save intermediate checkpoints during training (cheap) and post-process them later without slowing training.

**Tests** in `tests/experiments/test_eval_3x3.py`:
- `value_eval` returns dict with the right keys/shapes on a random-init `ValueNet`.
- `beam_eval_walk` produces solve-rate per depth (sanity: ~0% on random-init).
- `beam_eval_v_star` produces per-V\* solve rate + MAE on a random-init net.
- `post_run_beam_eval.py` smoke-runs on an empty checkpoint dir without erroring.

### Phase 1 acceptance gate

- P1a + P1b ✅ done.
- P1c three functions live in `experiments/davi-3x3/eval.py` + `scripts/post_run_beam_eval.py` exists + tests green.
- W&B namespacing for live eval: `eval/value/per_walk_depth/d{1..14}`, `eval/value/v_star_mae/d{1..6}`, `eval/value/macro_v_star_mae`. Verified by smoke-run wiring (Phase 2).

## Phase 2 — Smoke training with early-stop infra

**Goal.** Train a single 3x3 DAVI run with proper instrumentation. Goal isn't to pick a champion config or hit SPEC's acceptance criteria — it's to (a) verify the wired-up training loop works on 3x3 (loss decreases, no NaN, no MPS errors), (b) watch `macro_v_star_mae` evolve over training to confirm the network is learning the value function, (c) characterize how early-stop behaves in practice on 3x3, and (d) produce a final-checkpoint beam-eval as our first 3x3 capability data point.

T0 capacity calibration is **deferred to backlog** in favor of a hand-picked architecture.

### Hand-picked architecture (acknowledged guess, not earned)

```yaml
body_widths: [5120, 1024]
n_residual_blocks: 4
normalization: bn
```

Per CLAUDE.md "Earn every hyperparameter": this is an explicit guess captured for traceability. **Rationale (NOT borrowed values):** 5120 = 5×1024, MPS-aligned, larger than 2x2's first hidden (4096) to acknowledge 3x3's wider input (324 vs 144 one-hot dims) without over-scaling. Second hidden 1024 + 4 residual blocks + BN match 2x2's structural shape. Clearly NOT the DeepCubeA `[5000, 1000]` borrowed shape. T0 capacity sweep (deferred to backlog) earns the real config.

### Early-stop infrastructure (sync-period units)

**Five new `DAVIConfig` fields** (all required, no defaults — per project rule):

```yaml
early_stop_enabled: true
early_stop_metric: macro_v_star_mae
early_stop_patience_evals: 12      # = 12 sync periods of no improvement → stop
early_stop_min_evals: 4            # warmup: don't even check until 4 syncs in
early_stop_min_delta: 0.001        # absolute min improvement to count as progress
```

**Constraint validation at config-load:** if `early_stop_enabled=true`, require `eval_every == target_sync_interval` (or be a clean multiple). Errors at load time if misaligned. This guarantees patience-in-evals = patience-in-syncs.

**Semantics.** Every `eval_every` steps, run `value_eval`, log `macro_v_star_mae`. Track `best = min(history_so_far)`. If `current_step ≥ early_stop_min_evals * eval_every` AND last `early_stop_patience_evals` evals haven't improved by ≥ `early_stop_min_delta` against `best` → write final checkpoint, log `event=early_stop`, exit cleanly.

**Conservatism rationale (per user direction):** at sync_interval=500, 12 sync patience = 6000 steps of plateau before stopping. At sync_interval=1000, 12000 steps. Generous — matches "let it run a bit longer than we think" while still catching the cycle-3/cycle-4 "10k+ flat steps" failure mode. `min_delta=0.001` means tiny improvements still count as progress; only true plateaus trigger stop.

**Why sync-period units (recorded for future-me):** between target-net syncs, the Bellman target is frozen — loss is just settling toward a stationary target. The "real" progress signal is sync-to-sync, not step-to-step. Patience measured in steps is fooling yourself.

### W&B configuration

**Separate project `rubik-3x3`** (default in `experiments/davi-3x3/run.py`). 2x2 stays on `rubik` until renamed (drive-by → backlog). Run names from run-dir stem (existing wandb integration). Auto-tag wandb runs with `cube=3x3`, `phase=smoke`. Run dirs descriptive: e.g. `experiments/davi-3x3/runs/smoke-bw5120-d8-001/`.

**W&B namespacing:**
- Live during training: `eval/value/per_walk_depth/d{1..14}`, `eval/value/v_star_mae/d{1..6}`, `eval/value/macro_v_star_mae`.
- Post-training (one-shot summary at end + post-hoc trajectory): `eval/beam_walk/{solve_rate,mean_len}/d{1..14}`, `eval/beam_v_star/{solve_rate,mean_len,mae}/d{1..6}`.

Three independent panel groups in the wandb workspace.

### Smoke run config

Single config `experiments/davi-3x3/configs/smoke.yaml`:

```yaml
# arch (hand-picked guess, not earned)
body_widths: [5120, 1024]
n_residual_blocks: 4
normalization: bn

# scrambles — slightly beyond V* oracle reach to exercise bootstrap on out-of-V* states
max_scramble_depth: 8
max_scramble_depth_initial: 0
max_scramble_depth_ramp_steps: 0

# optimizer
batch_size: 4096
n_steps: 10000             # hardcap; expect early-stop to fire earlier
learning_rate: 0.001
target_sync_interval: 500

# logging / eval / checkpoints
log_every: 100
eval_every: 500            # = target_sync_interval per early-stop alignment rule
checkpoint_every: 2500     # save 4 intermediates in 10k steps for post-hoc trajectory
seed: 0
device: mps

# early-stop
early_stop_enabled: true
early_stop_metric: macro_v_star_mae
early_stop_patience_evals: 12
early_stop_min_evals: 4
early_stop_min_delta: 0.001
```

### Sub-blocks

#### P2a — Early-stop + value_eval wiring + W&B project switch

- Add 5 early-stop fields to `DAVIConfig` + YAML round-trip + tests.
- Wire `value_eval` call into `experiments/davi-3x3/run.py` at every `eval_every`. Track `macro_v_star_mae` history; fire early-stop when criteria meet.
- Switch `experiments/davi-3x3/run.py` wandb project default from `rubik` to `rubik-3x3`.
- Tests for the early-stop logic on synthetic histories (improving / flat / decreasing-then-flat / never-improves).

#### P2b — Smoke training run + post-training beam eval

- Author `experiments/davi-3x3/configs/smoke.yaml`.
- Run training: `uv run python experiments/davi-3x3/run.py --config configs/smoke.yaml`.
- After training: invoke `beam_eval_walk` + `beam_eval_v_star` on `net_final.pt`; results to `runs/<run>/results/beam_eval.json` + summary metrics to wandb.
- Optional follow-up: run `scripts/post_run_beam_eval.py` on intermediate checkpoints to produce the beam-trajectory chart.
- Writeup in `experiments/davi-3x3/results/results.md` with the project's intuition convention (Observations → Hypotheses with verification plans → Open questions).

### Phase 2 acceptance gate

1. `DAVIConfig` extended with 5 early-stop fields; tests green; `eval_every == target_sync_interval` validation enforced at config-load.
2. Smoke run completes — either by `n_steps` cap or by early-stop firing. No NaN/MPS errors.
3. `macro_v_star_mae` trajectory shows clear downward trend over training (not flat from step 0; not exploding).
4. Post-training beam evals produce per-V\* and per-walk-depth solve-rate numbers.
5. Results writeup includes intuition section per project convention.
6. wandb run shows up under `rubik-3x3` project (NOT `rubik`) with three panel groups populated.

## Decisions captured here

1. **No borrowed hyperparameters from 2x2** (CLAUDE.md). Architecture for phase 2 is **acknowledged hand-picked guess** at `[5120, 1024] × 4 bn`, not earned. T0 capacity sweep deferred to backlog.

2. **Bounded V\* at K=6** chosen for cheapness over depth coverage (K=8 expansion → backlog). Build was 4.8s; cache 14.9 MB; total 983,926 states matching the published QTM layer-count table.

3. **`experiments/davi-3x3/`** mirrors `experiments/davi-2x2/`. Analysis pipeline ports unchanged. "Extend the existing pipeline, never parallel-build."

4. **Three-function eval API**, named per user direction: `value_eval` (live during training) + `beam_eval_walk` (post-hoc, forever-eval) + `beam_eval_v_star` (post-hoc, ground-truth while oracle exists). **Beam search runs post-training only** — beam cost would slow training meaningfully. Greedy = `beam_width=1`, no separate implementation. Intermediate checkpoints saved (`checkpoint_every`) for post-hoc trajectory analysis.

5. **Early-stop measured in target-sync periods** (not steps). Eval cadence aligned to sync cadence by validation rule. 12 sync patience + 4 sync warmup + 0.001 min_delta on `macro_v_star_mae`.

6. **W&B: separate `rubik-3x3` project** for 3x3 work to prevent cross-contamination with 2x2's `rubik`. 2x2 rename to `rubik-2x2` deferred to backlog drive-by.

7. **Phase 3+ replanned later.** This plan does not commit to T2/T3 cycles, beam acceptance against SPEC, or 10M tx/sec verification — those depend on phase 2 outcomes.

## Critical files

**New (P1c):**
- `experiments/davi-3x3/eval.py` — three-function API expanded (`value_eval` + `beam_eval_walk` + `beam_eval_v_star`)
- `scripts/post_run_beam_eval.py` — post-hoc trajectory analysis
- `tests/experiments/test_eval_3x3.py` (or similar location)

**Modified (P2a):**
- `src/rubik/training/config.py` — 5 new early-stop fields + validation rule
- `experiments/davi-3x3/run.py` — `value_eval` wiring + early-stop logic + wandb default project flip to `rubik-3x3`

**New (P2b):**
- `experiments/davi-3x3/configs/smoke.yaml`
- `experiments/davi-3x3/runs/smoke-bw5120-d8-001/` (gitignored, run output)
- `experiments/davi-3x3/results/results.md` — writeup with intuition section

**Done (P1a + P1b):**
- `experiments/davi-3x3/{run,eval,calibrate_step_time_3x3}.py` + analysis layer
- `src/rubik/oracle/v_star_bounded_3x3.py`
- `scripts/build_v_star_bounded_3x3.py`
- `tests/oracle/test_v_star_bounded_3x3.py`

## Closed questions (resolved this conversation)

- **Q-P1a (analysis script audit):** No source parameterization needed — all 2x2 hardcodes are legitimately category-(a) (experiment-specific). Path A taken in P1a.
- **Q-P1b (cache format):** `.npz` chosen, mirrors `v_star_2x2.npz`. Cache gitignored.
- **Q-P1c (eval set sample size and shape):** Reshaped entirely. No committed npz; deterministic-from-seed. Live `value_eval` uses ~600 V\*-stratified states (100/layer × 6) for cheap forward-pass MAE; post-training beams use larger samples (1400 walks, 1200 V\*-stratified).
- **Q-P2a (calibration grid):** MOOT — calibration deferred to backlog.
- **Q-P2b (smoke run length):** 10k step hardcap with early-stop firing earlier on plateau.

## References

- M5 plan shape: `plans/m5-davi.md` — mirror this depth of detail when phase 3+ gets planned.
- CLAUDE.md sections: "Earn every hyperparameter — do not borrow", "Cycle reporting pipeline (don't parallel-build)", "Evaluating training cycles correctly", "Cube / cubie naming".
- SPEC M8: `SPEC.md:328–358` — full M8 acceptance gates (deferred until phase 3+).
- LOG 2026-05-06: M8 env-layer block + M8 P1 block (P1a + P1b done).
- LOG 2026-05-05 (M5-followup cycle 4): the "macro_mae alone hides regressions" cautionary tale that motivates the post-training beam-eval discipline carried forward here (cheap MAE ≠ capability; beam answers capability).
