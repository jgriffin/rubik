# M8 — 3x3 enablement (training surface)

> **Active scope:** phases 1 + 2 only. Goal is "does the 3x3 training + eval stack work end-to-end?" — not acceptance-grade champion runs. Phase 3+ (axis sweeps → champion training → beam → 10M tx/sec verification) gets replanned after phase 2 lands.

## Context

The M8 env layer is ✅ (LOG 2026-05-06): `CUBE_3X3`, cubie oracle with 8 corners + 12 edges + fixed centers (U/D-axis edge-flip convention), `MOVE_PERM_3X3` snapshot, oracle↔tensor equivalence at 10k random sequences, all standard 3x3 identities green (M⁴=I, sexy⁶=I, Sune⁶=I, T-perm²=I). Test count 264 → 480.

The training/search/beam stack is `CubeSpec`-parameterized end-to-end. The context dump that opened this plan confirmed:
- `training/davi.py:compute_targets` / `davi_step` use `spec.n_moves` / `spec.n_stickers`
- `model/network.py:ValueNet` input dim derives from `spec.n_stickers * spec.n_colors`
- `search/beam.py:beam_solve_batch` is generic on `spec`
- `oracle/v_star_2x2.py` is appropriately size-suffixed

**No 2x2 hardcodes in any code path that matters for 3x3 training.** The CubeSpec bet has paid off.

3x3 has **4.3 × 10¹⁹ states**. A full BFS V* oracle is impossible. This plan accepts that as a permanent constraint and scopes the early M8 training work as smoke-testing the stack — does loss decrease, does the eval pipeline produce charts, what does the cost surface look like vs 2x2. Acceptance-grade evaluation against SPEC.md's M8 criteria (greedy ≥95% on d≤15, beam=4096 100% on d=20, ≤24 mean moves, <10s/scramble, 10M tx/sec) is **deferred to a later plan extension**.

## Approach overview

Two phases. Each is a discrete cc-process block on its own branch off main. Sub-blocks within a phase stack on the parent branch (the M8 env-layer pattern from LOG 2026-05-06).

| # | Phase | Goal | Suggested branch |
|---|---|---|---|
| 1 | **Eval scaffold + bounded V\* sanity oracle** | Stand up `experiments/davi-3x3/`; build BFS V\* up to K=6 as a sanity-check ground truth; port the analysis layer | `m8-3x3-eval-scaffold` |
| 2 | **T0 capacity calibration + smoke training** | Port `calibrate_step_time.py` to 3x3; sweep `batch × body_widths × n_residual_blocks` for ms/step; run a short training cycle to verify loss decreases | `m8-3x3-t0-capacity` |

## Phase 1 — Eval scaffold + bounded V\* sanity oracle

**Goal.** Mirror the 2x2 experiment dir into `experiments/davi-3x3/`. Port the analysis pipeline. Build a *small* bounded V\* oracle covering depths 1..K (K=6) as sanity-check ground truth. Explicitly **not** trying to enable v\_star\_stratified eval at SPEC's d=15+ — that gap is accepted.

### Sub-blocks

#### P1a — `experiments/davi-3x3/` skeleton

Mirror the 2x2 layer structure exactly:

```
experiments/davi-3x3/
  run.py                       # entry point, --config <path>
  configs/                     # YAML configs (none authored yet)
  runs/                        # gitignored
  calibrate_step_time_3x3.py   # T0 calibrator (P2a)
  build_eval_set_3x3.py        # eval set generator (P1c)
  eval.py                      # checkpoint → per-depth solve-rate report
  eval_set_3x3.npz             # frozen eval set (P1c)
  analysis/
    analyze.py                 # JSONL → text summary + charts
    capture_solve_histograms.py
    render_error_trajectories.py
    audit_sampler.py
  results/                     # canonical writeups + charts
  README.md                    # methodology overview
```

**`run.py` and `eval.py` swap `CUBE_2X2` → `CUBE_3X3` and adjust eval-set load path. Analysis scripts must already be spec-agnostic** — if they have any `CUBE_2X2` literal hardcoded, the right fix is at the source (accept a `spec` argument or read it from the run's config), not a copy-modify into davi-3x3. Drive-by parameterization, not a separate refactor.

**Acceptance:** dir scaffolded; `analyze.py` runs against an empty `runs/` without erroring.

#### P1b — Bounded BFS V\* up to K=6

New module: `src/rubik/oracle/v_star_bounded_3x3.py` (size-suffixed per CLAUDE.md naming convention).

**Approach.** Reverse-BFS from `CUBE_3X3.solved_state`, layer-by-layer. At each layer, expand by all 12 QTM moves, dedup against already-seen states (raw bytes, no canonicalization). Store `dict[bytes, int8]` mapping state → V\*.

**Layer counts (QTM, well-known from cube literature):**

| d | Layer | Cumulative |
|---|---|---|
| 0 | 1 | 1 |
| 1 | 12 | 13 |
| 2 | 114 | 127 |
| 3 | 1,068 | 1,195 |
| 4 | 10,011 | 11,206 |
| 5 | 93,840 | 105,046 |
| 6 | 878,880 | 983,926 |

So K=6 caches **~1M states**. Estimated build cost ~30–120s wall on M4 Max; estimated cache size ~60–80MB packed (state bytes uint8[54] + v\_star int8). Fast enough to rebuild on demand; small enough to commit to disk.

**Why K=6 not K=5 not K=8.** User's call (this conversation): K=8 jumps to ~9M states (8x larger), multi-minute build, ~600MB+ on disk. K=5 caps at 105k states — too small for a reasonable sanity check. K=6 is the sweet spot of "fast and cheap, but actually exercises the eval pipeline against a non-trivial sample."

**Cache location.** `data/v_star_bounded_3x3_k6.npz` (or `.pkl` if dict-shape works better). Decision deferred to block-open; mirrors whatever `oracle/v_star_2x2.py` does for `data/v_star_2x2.npz`.

**Tests** (`tests/oracle/test_v_star_bounded_3x3.py`):
- Total state count matches the layer-count table above (sentinel: 983,926 for K=6).
- `V*[CUBE_3X3.solved_state] == 0`.
- Random-walk round-trip: for d ∈ {1..6}, generate a length-d random walk, look up V\* of the endpoint, assert `V* ≤ d` (often `<` due to walk redundancy — informative, not a failure).
- Identity sanity: applying any QTM move to a V\*=k state lands on a state with V\* ∈ {k-1, k, k+1}.

**Acceptance:** module + cached file + tests green. Test count goes 480 → ~485.

#### P1c — Eval set construction

New script: `experiments/davi-3x3/build_eval_set_3x3.py`. Output: `experiments/davi-3x3/eval_set_3x3.npz` — 100 states × 14 depths (d=1..14) random-walk samples, frozen seed. Stored fields:

- `states[14, 100, 54]` uint8 — sticker-form states
- `walk_depth[14, 100]` int8 — walk length used to generate (this is the "depth bin")
- `v_star[14, 100]` int8 — true V\* if available (i.e. walk_depth ≤ 6 and endpoint is in the bounded oracle), else `-1` sentinel

**Decision: depth bin = walk-length, not true-V\*.** Walk-length scales beyond K=6 (true-V\* doesn't), and we want one consistent depth-bin scheme across the whole eval set. The bounded oracle's V\* is recorded as a side channel for hazard analysis (the cycle-4 lesson on 2x2: walk redundancy means "walk-depth-d" populations are biased toward V\*\<d states; we want to be able to surface that on 3x3 too at d≤6).

Frozen seed → deterministic eval set, committed to git. ~80KB packed, fine to track.

**Acceptance:** `eval_set_3x3.npz` committed; `eval.py` loads it and reports per-depth solve rate against a random-init checkpoint (sanity: should report ~0% solving everywhere).

### Phase 1 out of scope

- v\_star\_stratified sampling at d=7+ (no oracle there).
- Pattern-database-style admissible heuristics for full-depth ground truth.
- Per-cycle results.md writeups beyond the README.md scaffold (those come at phase 3+).

### Phase 1 acceptance gate

All three sub-blocks landed and tests green. Concretely:

1. `experiments/davi-3x3/` dir exists, scripts run on empty `runs/` without error.
2. `src/rubik/oracle/v_star_bounded_3x3.py` + cached file + tests green; bounded V\* loads in <1s after cache warm.
3. `eval_set_3x3.npz` committed; `eval.py` produces a populated 14-row per-depth report on a random-init `ValueNet`.

## Phase 2 — T0 capacity calibration + smoke training

**Goal.** Port the 2x2 calibrator shape to 3x3 and run a short training cycle. Goal isn't to pick a champion config — it's to (a) confirm 3x3 *trains at all* (loss decreases, no NaN, no MPS dispatch errors), (b) measure the cost surface vs 2x2 (54-sticker input is 2.25× wider — does that translate linearly to step time, or is dispatch still bottlenecking?), and (c) earn the first-cut "reasonable" config that phase 3 will start from.

### Sub-blocks

#### P2a — Port `calibrate_step_time.py` to 3x3

New file: `experiments/davi-3x3/calibrate_step_time_3x3.py`. Mirror `experiments/davi-2x2/calibrate_step_time.py` cell-for-cell, swapping `CUBE_2X2` → `CUBE_3X3`.

**Sweep grid (lean: match 2x2's grid for direct comparison).** Open question for block-open — whether to match 2x2's exact grid or pick a 3x3-appropriate one. Default: match. Document any cells that OOM.

**Output:** `experiments/davi-3x3/results/t0_calibration_3x3.md` (table of cell × ms/step, with CIs from `hyperfine`-style repeats and `torch.mps.synchronize()` discipline) + a small HTML chart. Earnings logged in the results.md `## Intuition` section per the project's hand-written intuition convention.

**Out-of-scope (carried over from M5/M7):** no comparison to DeepCubeA / EfficientCube reference values. The 2x2 calibration earned `[4096, 1024]` × 4 residual blocks × batch=4096 the hard way; 3x3 earns its own.

#### P2b — Smoke training run

After P2a publishes a cost surface, pick the cheapest cell that's not pathologically tiny and run **500 steps**. Goal:

- Loss decreases monotonically (or close to it — no NaN, no spike).
- Greedy solve rate at d ∈ {1, 2, 3} climbs above 0% (network learned *something* about 3x3 in 500 steps).
- Eval pipeline produces a populated chart on the smoke-run checkpoint.

**Config:** new `experiments/davi-3x3/configs/smoke_500.yaml`, full `DAVIConfig` round-trip, all fields explicit (no defaults — project rule).

**This is not a science run.** It's a smoke test of the wiring. If loss doesn't decrease, the bug is in the wiring (state encoding, model construction, scramble generation, target net sync, optimizer step), not the hyperparameters. Phase 2's value is precisely catching those wiring bugs cheaply.

### Phase 2 out of scope

- Champion-class training (≥30k steps).
- Axis sweeps beyond P2a's calibration grid.
- Warm-start cycles or curriculum schedule (`max_scramble_depth_initial` / `_ramp_steps`).
- Throughput verification (the SPEC 10M tx/sec gate). That's M4-shaped batch-sensitivity work and goes in `experiments/batch-sensitivity-3x3/` at phase 3+.
- W&B integration (already wired generically; will Just Work, but smoke run can stay JSONL-only to keep cycle short).

### Phase 2 acceptance gate

1. `t0_calibration_3x3.md` + chart published.
2. 500-step smoke run shows loss decreasing monotonically; no NaN; no MPS errors; greedy solve at d≤3 > 0%.
3. The cost surface vs 2x2 is documented in the results.md intuition section — not as a quantitative target, but as the conceptual takeaway ("3x3 is N× more expensive at fixed batch and fixed model size, dispatch knee shifted by M, etc.").

## Decisions captured here

1. **No borrowed hyperparameters from 2x2** (CLAUDE.md "Earn every hyperparameter"). The 2x2 cycle-3 winning config is a 2x2 result; its values do not transfer. Phase 2's calibration produces the 3x3 starting config the same way M5's tier 0 produced the 2x2 one.
2. **Bounded V\* at K=6** — explicitly chosen for cheapness over depth coverage. ~1M states, ~60MB cache, ~30–120s build. SPEC's deep-eval question (d=15+) is accepted-as-gap, deferred to a later plan extension.
3. **`experiments/davi-3x3/`** mirrors `experiments/davi-2x2/` directly. The analysis pipeline (analyze/capture/render layer) ports unchanged — the project pattern is "extend the existing pipeline, never parallel-build" (CLAUDE.md "Cycle reporting pipeline"; reinforced by the C3 misfire in M5 cycle-4).
4. **Cycle-eval discipline carries forward** (CLAUDE.md "Evaluating training cycles correctly") with one caveat: "both sampling strategies" collapses to **one** (random\_walk\_depth) for d>6 since v\_star\_stratified requires V\* ground truth we don't have. At d≤6 we *can* run both — and the bounded oracle exists partly to surface walk-redundancy hazards on 3x3 the way the V\*-stratified eval did on 2x2.
5. **Phase 3+ replanned later.** This plan deliberately does not commit to T2/T3 axis sweeps, beam search acceptance, or the 10M tx/sec verification. Those depend on phase 2's cost surface and on whether phase 2 surfaces any structural surprises about 3x3 training.
6. **One block per phase, branched off main.** P1 and P2 each get their own LOG block; sub-blocks (P1a/b/c, P2a/b) stack on the parent branch in the M8 env-layer pattern.

## Critical files

**New:**
- `experiments/davi-3x3/` (whole directory; mirrors `davi-2x2/`)
- `src/rubik/oracle/v_star_bounded_3x3.py`
- `data/v_star_bounded_3x3_k6.npz`
- `tests/oracle/test_v_star_bounded_3x3.py`

**Modified (probably zero, possibly a small parameterization):**
- `experiments/davi-2x2/analysis/*.py` — only if any contains a `CUBE_2X2` literal that should accept `spec` instead. Drive-by fix at source.

**Reference, not changed:**
- `experiments/davi-2x2/{run,calibrate_step_time,eval}.py` — shape templates
- `experiments/davi-2x2/configs/sync500_kmax20.yaml` — DAVIConfig field reference (NOT value reference; values stay 2x2-only)
- `src/rubik/oracle/v_star_2x2.py` — module shape for the bounded variant

## Open questions (resolve at block-open time, not now)

- **Q-P1a:** Do `experiments/davi-2x2/analysis/*.py` contain any `CUBE_2X2` literals? Subagent didn't trace this. If yes → drive-by parameterization. If no → straight scaffold.
- **Q-P1b:** Cache format — `.npz` (mirrors v\_star\_2x2) or `.pkl` (preserves dict shape natively)? Probably `.npz` for symmetry.
- **Q-P1c:** Eval set sample size — 100 per depth × 14 depths = 1400 (matches 2x2). Bigger gives tighter solve-rate CIs at the cost of slower eval. Lean: keep 1400 for symmetry; revisit if eval becomes a bottleneck.
- **Q-P2a:** Match 2x2's exact calibration grid (3 × 3 × 2 = 18 cells) or pick a 3x3-aware grid? Lean: match. Document OOMs.
- **Q-P2b:** Smoke run length — 500 steps or 1000? Lean: 500 if loss is dropping cleanly by step ~200, else 1000 to give the curve more shape.

## References

- M5 plan shape: `plans/m5-davi.md` — mirror this depth of detail when phase 3+ gets planned.
- CLAUDE.md sections: "Earn every hyperparameter — do not borrow", "Cycle reporting pipeline (don't parallel-build)", "Evaluating training cycles correctly", "Cube / cubie naming".
- SPEC M8: `SPEC.md:328–358` — full M8 acceptance gates (deferred until phase 3+).
- LOG 2026-05-06: M8 env-layer block — what we did to get to this surface.
- LOG 2026-05-05 (M5-followup cycle 4): the "macro\_mae alone hides regressions" cautionary tale that motivates the cycle-eval discipline carried forward here.
