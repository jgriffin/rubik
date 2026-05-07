# M8 — `beam_solve_batch` perf overhaul

## Context

The width-sweep block (closed `0d2f3f8`) traced where wall-time goes inside `beam_eval_walk`. Headline: at width=128, 70s of total wall is roughly half forward-pass compute (~30s) and half orchestration (~25s Python loop overhead + ~10s GPU↔CPU sync + ~5s misc). The orchestration half is structural — three architectural choices in `src/rubik/search/beam.py` create it:

1. **Per-scramble outer loop.** `beam_solve_batch` is `for i in range(n): _beam_solve_single(states[i])`. 100 scrambles per depth × 14 depths = **1400 sequential beam runs**. Each step inside each beam is a separate Python iteration. Total Python iterations across the sweep: ~21,000. Forward-pass batches max out at width × 12 = 1536 at width=128 — small enough that MPS is launch-latency-bound, not throughput-bound.
2. **GPU↔CPU sync every step.** The within-beam dedup uses `dict[bytes(state)] = (v, idx)`, which forces `.detach().cpu().numpy()` on the children tensor every step. The early-exit-when-solved check forces `.any().item()` every step. Together: 21,000 forced CPU↔GPU round-trips per sweep.
3. **CPU-side selection.** Top-k is `sorted(dedup.values())[:width]` — a Python sort over the deduped entries.

Smoking gun: **width=8 and width=16 wall-time tie at 30.6s** (full sweep). Forward-pass compute at these widths is tiny (1-4s); the rest is the constant Python+sync floor. Forward-pass FLOPs would not change in the rewrite — what changes is the orchestration cost, which today dominates at small widths and is meaningful at every width.

`beam_solve_batch` is the **shared primitive for both eval (`beam_eval_walk`, `beam_eval_v_star`) and production solve.** Optimizing it once benefits both. Out of scope: training-loop perf — different inner loop, different bottlenecks (target-net inference, replay sampling, optimizer step), gets its own block later.

## Goal & success criteria

- **Wall-time targets** (post-overhaul, full eval grid):
  - `default` profile (current uniform n=100, d=1..14): **≤2 min** total wall (currently ~8 min). Width=128 individual cell ≤15s.
  - `fast` profile (variable schedule): **≤30s** total wall — the cycle-screening operating point.
  - `thorough` profile (variable schedule, large n at deep): **≤5 min** total wall — the milestone-quality operating point.
- **Equivalence**: aggregate per-(walk_depth, beam_width) solve_rate within binomial SE of the existing `sweep_full_train_final.json` baseline. At n=100 / p≈0.7 the SE is ±5pp, so cells must agree within ±10pp (2σ) on the same seed for the gate to pass; outliers beyond that need investigation. **No cell-by-cell trace matching** — beam search at d=14 is statistical and individual outcomes are noise.
- **API**: callers of `beam_solve_batch` (production solve, `beam_eval_walk`, `beam_eval_v_star`) compile and pass with no behavioral regression. Per-state `max_steps` is added as a permitted call shape; uniform `max_steps` remains the default. `n_per_depth` becomes `int | Sequence[int]` — scalar backward-compat preserved.
- **Validated cycle-eval recommendation**: after C6, the project has a documented answer to "what profile do I use for cycle decisions vs milestone verdicts?" — backed by per-cell agreement-within-SE between `fast` and `thorough` on the canonical checkpoint. The `default` profile (current n=100 uniform) is checked against `thorough` to surface any systematic bias relative to the new larger-n reference.
- **Sampling-allocation insight**: at d=1 only 12 walks are reachable (one per QTM move); at d=2 ~132. Current uniform n=100 oversamples d=1 by ~8× and undersamples d=11..14 (where binomial SE at p≈0.7 is ±5pp at n=100). The variable-n schedule reallocates the sample budget toward depths where SE actually binds.
- **Test count delta**: existing tests in `tests/search/test_beam.py` and `tests/experiments/test_eval_3x3.py` continue to pass at every commit. New equivalence + regression + profile-shape tests added.

## Approach

Single branch `beam-solve-perf` from main HEAD `0d2f3f8` (already created and block-open committed, `8d6c77b`). **Six atomic commits** below — C1-C5 are the perf rewrite of `beam_solve_batch`; C6 adds variable-n sampling profiles on top of the now-fast primitive. Each commit is testable independently and keeps the existing test suite green. Per-commit checklist:

- [ ] Existing `tests/search/test_beam.py` passes
- [ ] Existing `tests/experiments/test_eval_3x3.py` passes
- [ ] Existing `tests/scripts/test_beam_eval_run.py` passes
- [ ] `uv run ruff check src/rubik/search/ tests/search/` clean
- [ ] Atomic commit message + co-author footer

### C1 — Cross-scramble batching (architectural foundation)

**File**: `src/rubik/search/beam.py`. **Internal change only — no API change.**

Replace the per-scramble outer loop with a single batched beam tensor:

- Beam state: `(N, beam_width, n_stickers)` where `N = states.shape[0]` — all input scrambles share one beam.
- Per-step expansion: `apply_all_moves(beam_states.reshape(N * beam_width, ...), spec).reshape(N, beam_width * 12, ...)` → score with one `net(...)` call of batch size `N × beam_width × 12`. At the typical eval shape (N=100 per walk-depth call, width=128) batch becomes 153,600 — well above the MPS efficiency knee. For the V*-stratified call (N=200) and width=512 the batch hits 1.2M; split into 2-4 chunks if memory binds (see C5).
- Per-scramble first-solved tracking: `solved_at_step[i] = first step k where any beam slot of scramble i is goal`. Tensor of shape `(N,)`, initialized -1, written via `where(is_solved & (solved_at_step == -1), step, solved_at_step)` per step. Stays on-device.
- Early-exit logic: keep it for now (C2 will revisit), but adapt to "exit when ALL scrambles have either solved_at_step != -1 OR exceeded their max_steps." This still has a `.any().item()` per step but at least the per-step cost is now amortized over N scrambles.

**Keep for this commit (will be replaced in later commits):**
- `dict[bytes(state)] = (v, idx)` dedup, now operating per-scramble within the batch (`for i in range(N)` in Python over batched-children — yes, this is still Python overhead, but it's removed in C3)
- `sorted()` top-k (replaced in C4)

**Tests**:
- All existing `tests/search/test_beam.py` tests pass without modification (same external behavior).
- New: `test_beam_solve_batch_n_invariant_to_split` — running `beam_solve_batch(states[:50])` and `beam_solve_batch(states[50:])` produces the same `solve_lens` as `beam_solve_batch(states)` on a fixed-seed corpus. Confirms the batched implementation handles arbitrary N correctly.

**Expected speedup**: 2-3× at width=128. Most of the gain comes from amortizing forward-pass kernel-launch overhead across all scrambles. Remaining cost: Python per-step dedup loop (still in this commit) and GPU↔CPU sync per step.

**Commit message**: `m8: beam_solve_batch — batch all input scrambles in one beam tensor`

### C2 — Run-to-completion (kill `.any().item()` early-exit sync)

**File**: `src/rubik/search/beam.py`.

Remove the early-exit-when-all-solved check. Run **all** beam steps for `max(max_steps_per_state)` iterations regardless. Track `solved_at_step` on-device throughout. At the end, for each scramble, check `solved_at_step <= max_steps[i]` to determine final solve verdict.

**Per-state `max_steps` (additive API change)**: add a new optional kwarg `max_steps_per_state: torch.Tensor | None`. When provided, each scramble is gated by its own budget; when `None`, fall back to the scalar `max_steps`. `beam_eval_walk` will start passing per-state budgets in C5; production callers continue passing the scalar.

**Why this is correctness-preserving**:
- Solved beams emit a goal-state at step k; the goal-state propagates via `apply_all_moves` to itself + 12 (mostly non-goal) children at step k+1. The first-solved-step tracker captures step k regardless of what happens later.
- The cost of running steps after a beam has solved is small: it's at most `max(max_steps) - k` extra steps per scramble, but these all run in parallel with the unsolved scrambles' work, so wall-time impact is near-zero.

**Tests**:
- All existing tests pass (the public solve_len for solved beams is the *first* solve step, which is what they assert).
- New: `test_beam_solve_batch_no_early_exit_same_solve_lens` — fixed seed, fixed scrambles, verify solve_lens match the C1 result with-vs-without the early-exit (uses an internal flag for the test, removed before commit).
- New: `test_beam_solve_batch_per_state_max_steps` — pass two scrambles where one budget is 5 and the other is 10; assert the 5-budget scramble's solve verdict considers only steps 1-5.

**Expected speedup**: 1.5-2× on top of C1. ~10s of GPU↔CPU sync removed.

**Commit message**: `m8: beam_solve_batch — run-to-completion, drop .any().item() sync per step`

### C3 — On-device dedup via int64 hash

**File**: `src/rubik/search/beam.py`. New helper: `src/rubik/search/state_hash.py` (or inline in beam.py if small enough).

Replace `dict[bytes(state)] = (v, idx)` with a GPU-native dedup:

1. **Hash each child state to int64.** State is `(B, n_stickers)` int8 (n_stickers=54 for 3x3). Pack-and-hash to an int64 per state — e.g. interpret first 8 bytes as int64, XOR with hash of next 8 bytes, etc. Or use a stable polynomial hash on int8 view. Collision-rate target: <1 per 10M states. (For 3x3, only ~3.7M states are reachable in QTM-≤14 anyway — collision risk is tractable to bound.)
2. **`torch.unique` for dedup.** `unique, inverse = torch.unique(hashes, return_inverse=True)`. Works on MPS.
3. **Scatter-min values.** `unique_values = full(unique.shape, +inf); scatter_reduce(unique_values, dim=0, index=inverse, src=values, reduce="amin")` (or equivalent — there's `scatter_reduce_` on MPS). Lowest-V wins per unique state.
4. **Map back to original indices.** Build `unique_idx` via `torch.unique` with `return_inverse=True` then a small index trick to get a representative original index per unique state.

**Hash collision handling**: a collision would cause two distinct states to be deduped together (the lower-V one survives, the higher-V one is dropped). For a well-chosen 64-bit hash on a 432-bit state (54 × 8), collision probability per pair is ~2⁻⁶⁴ ≈ 5×10⁻²⁰. Across all (state, state) pairs in a sweep (~30M states × 30M pairs), expected collisions: still vanishingly small. **Acceptable.** Document in a comment in `state_hash.py`.

**Tests**:
- `test_state_hash_no_collisions_on_oracle_corpus` — hash all states reachable in QTM-≤6 from the goal (the bounded V* oracle), assert all hashes are distinct.
- `test_state_hash_invariant_to_input_layout` — hashing a state from a contiguous tensor vs a strided view produces the same hash (catches subtle layout bugs).
- All existing beam tests pass (dedup is correctness-preserving up to hash collisions; the equivalence test in C5 will catch any aggregate divergence).

**Expected speedup**: 1.5-2× on top of C1+C2. The `.cpu().numpy()` round-trip is gone, and the Python dedup loop is replaced by 3-4 GPU ops.

**Commit message**: `m8: beam_solve_batch — on-device hash dedup (drops .cpu().numpy() per step)`

### C4 — On-device top-k

**File**: `src/rubik/search/beam.py`.

Replace `sorted(dedup.values())[:beam_width]` with `torch.topk(dedupped_values, k=beam_width, largest=False)`. Returns `(values, indices)` — use indices to gather the next beam frontier from the deduped child tensor.

**Edge case**: when fewer unique survivors than `beam_width` exist (rare — only at very early steps with small starting state space), pad with sentinel `+inf` values or shrink the beam. Production solve already handles short beams correctly via `beam_states.shape[1]` reads, so the rest of the code adapts naturally.

**Tests**:
- All existing tests pass.
- `test_beam_top_k_handles_short_beam` — synthetic case with 5 unique children at width=10; assert beam shrinks to 5 without error.

**Expected speedup**: 1.2-1.5× on top of prior. Gain comes from removing the CPU sort and the host→device transfer of selected indices.

**Commit message**: `m8: beam_solve_batch — torch.topk replaces CPU sorted() for selection`

### C5 — Equivalence check + benchmark + writeup

**Files**: `experiments/beam-eval-perf/results/sweep_full_train_final_post_overhaul.json` (new), `experiments/beam-eval-perf/results.md` (append second sweep row + perf delta table), `experiments/beam-eval-perf/intuition.md` (append "post-overhaul addendum" section), `experiments/beam-eval-perf/results/sweep_post_overhaul.html` (regenerate via existing renderer).

Steps:

1. Re-run the canonical width sweep against `full_train net_final.pt` with the new impl, same seed=0:
   ```
   uv run python scripts/beam_eval_run.py \
     --checkpoint experiments/davi-3x3/runs/20260507T043533Z_full_train/net_final.pt \
     --widths "8,16,32,64,128,256,512" \
     --output-json experiments/beam-eval-perf/results/sweep_full_train_final_post_overhaul.json \
     --seed 0
   ```
2. **Equivalence check**: compute per-(walk_depth, beam_width) solve_rate delta between baseline JSON and post-overhaul JSON. **All cells must agree within ±10pp (2σ at n=100, p≈0.7).** If any cell exceeds this, freeze and investigate before merging.
3. **Perf check**: total wall ≤ 2 min. Width=128 cell wall ≤ 15s.
4. **Writeup additions**:
   - `results.md` — append a "Post-overhaul sweep" subsection: side-by-side table of (width, old wall, new wall, speedup, old solve@d=14, new solve@d=14, |Δ|). Update the production-tier recommendations if the speedup changes the screening/production/diagnostic boundaries.
   - `intuition.md` — append a "Post-overhaul addendum" section (hand-written, project convention) with Observations on what the speedup actually shaped (did the small-width tie disappear? did the curve flatten? did MPS efficiency knees shift?), Hypotheses on remaining bottlenecks, and Open questions for the next perf block (likely candidate: training-loop perf).
5. **Render HTML**: `uv run python scripts/render_beam_eval_report.py --input experiments/beam-eval-perf/results/sweep_full_train_final.json experiments/beam-eval-perf/results/sweep_full_train_final_post_overhaul.json --output experiments/beam-eval-perf/results/sweep_comparison.html` — multi-input mode, baseline + post-overhaul overlaid.

**Tests**: no new unit tests in this commit; the equivalence + perf gates ARE the test, run as part of execution.

**Commit message**: `m8: beam_solve_batch — equivalence verified + sweep re-run; <Nx> speedup at width=128`

### C6 — Variable `n_per_depth` + sampling profiles + validation

**Files**: `experiments/davi-3x3/eval.py`, `scripts/beam_eval_run.py`, `experiments/beam-eval-perf/profiles.py` (new), `experiments/beam-eval-perf/results/sweep_fast.json` + `sweep_thorough.json` (new), `experiments/beam-eval-perf/results.md`, `experiments/beam-eval-perf/intuition.md`.

**Sub-task A — `beam_eval_walk` API change.** Signature accepts `n_per_depth: int | Sequence[int]`. When sequence, `len(n_per_depth) == len(walk_depths)`; element i is the count for `walk_depths[i]`. Same change for `beam_eval_v_star`'s `n_per_layer`. Scalar-int form remains supported (current callers don't break).

**Sub-task B — Profile presets in `experiments/beam-eval-perf/profiles.py`.** Three named profiles. **Specific numbers below are starting points; final values are tuned by the validation experiment in sub-task D and recorded in `intuition.md`.**

```python
PROFILES = {
  "fast": {                                                            # cycle-screening: ~10s wall
    "walk_depths": (1, 2, 4, 6, 8, 10, 12, 14),
    "n_per_depth": (12, 30, 50, 50, 80, 80, 100, 100),                 # 502 root scrambles
  },
  "default": {                                                         # current behavior: ~1 min wall
    "walk_depths": tuple(range(1, 15)),
    "n_per_depth": 100,                                                # 1400 root scrambles
  },
  "thorough": {                                                        # milestone-quality: ~3 min wall
    "walk_depths": tuple(range(1, 15)),
    "n_per_depth": (12, 50, 100, 100, 200, 200, 200, 200, 300, 300, 500, 500, 500, 500),  # 3662 scrambles
  },
}
```

The 12-at-d=1 entry exists because only 12 walks of length 1 are reachable; one per move enumerates the space exactly. The variable schedule front-loads samples where SE matters (d=11..14) and trims where the answer is deterministic (d=1..2).

**Sub-task C — CLI plumbing in `scripts/beam_eval_run.py`.** Add `--profile {fast,default,thorough}` (mutually exclusive with `--n-per-depth` and `--max-walk-depth`). When `--profile` set, both `walk_depths` and `n_per_depth` are pulled from `PROFILES`. The existing `--n-per-depth` is extended to accept either an int (uniform, current) or a comma-separated list of ints with length matching `walk_depths`.

**Sub-task D — Validation experiment.** Run all three profiles on `full_train net_final.pt` at the chosen production beam_width=256:

1. `--profile thorough` — establishes the new ground-truth reference. n=500 at d=11..14 → SE ≈ 2.0pp at p=0.7. Records `sweep_thorough.json`.
2. `--profile fast` — establishes the fast verdict. Records `sweep_fast.json`.
3. `--profile default` — re-runs the n=100 baseline for comparison. Records into the existing `sweep_full_train_final_post_overhaul.json` (already produced in C5; reuse it).

Compare per-cell solve_rate at d=11..14 across the three:
- `fast` vs `thorough`: cells must agree within `±2 × max(SE_fast, SE_thorough)`. If they do, **fast is validated as a legitimate cycle-screening proxy** — use it for "did this cycle improve" decisions when the answer is obvious.
- `default` vs `thorough`: surfaces whether the current n=100 was systematically biased relative to the n=500 reference at deep depths. Probably not — but worth checking.

If `fast` agreement fails at d=14 specifically (the most likely failure point), document the failure mode in `intuition.md` and propose a tuned schedule (e.g. raise n at d=14 to 200 in `fast`). The "fast" profile is a recommendation, not a contract — adjust based on data.

**Sub-task E — Writeup.**
- `results.md`: append "Sampling profiles" subsection with three-way comparison table (per-cell solve_rate, wall, sample count) and the validated profile recommendation. Update the production-tier guidance from C5: now there are TWO axes (beam_width × sampling_profile), so the recommendation matrix is screening = `fast × width=64`, production = `default × width=256`, milestone = `thorough × width=256` (or width=512 if d=14 confidence is the binding question).
- `intuition.md`: append "Profile validation addendum" — Observations on which depths benefit most from increased n (specifically: how much does d=14 SE shrink from n=100 → n=500?), Hypotheses on whether the variable schedule generalizes across checkpoints (probably checkpoint-dependent at the screening level, since per-depth saturation knees move — link to the prior block's H3), Open questions on whether shallow depths (d=1..3) carry any signal at all and could be dropped from `default`.

**Tests**:
- `test_beam_eval_walk_per_depth_n_sequence` — pass `n_per_depth=[10, 20, 30]` with `walk_depths=(1, 2, 3)`; verify each depth gets the right number of root scrambles. Mirror for `beam_eval_v_star`.
- `test_beam_eval_walk_n_scalar_backward_compat` — passing `n_per_depth=100` (int) still works exactly as before.
- `test_beam_eval_walk_n_sequence_length_mismatch` — passing a sequence of wrong length raises a clear error.
- `test_profiles_dict_well_formed` — for each profile, `len(walk_depths) == len(n_per_depth)` (when sequence) and all counts > 0.
- `test_beam_eval_run_profile_loads_correct_schedule` — invoke CLI with `--profile fast`, verify output JSON's per-cell `n` field matches the `fast` schedule.
- `test_beam_eval_run_n_per_depth_csv` — CLI with `--n-per-depth "12,50,100,..."` works.
- `test_beam_eval_run_profile_and_n_mutually_exclusive` — passing both `--profile fast` and `--n-per-depth` is an argparse error.

**Expected wall-time** (post-overhaul, M4 Max / MPS, beam_width=256):
- `fast`: ~10s
- `default`: ~60s (the C5 target)
- `thorough`: ~3 min

**Commit message**: `m8: beam_eval — variable n_per_depth, fast/default/thorough profiles, validation`

## Out of scope

- **Training-loop perf** (DAVI step time on M4 Max). Different bottlenecks: target-net forward pass, replay sampling, optimizer step. Gets its own block once we want to push training throughput.
- **Wider widths beyond 512.** The H4 hypothesis from the prior block (does d=14 saturate before width=2048?) is a different question and a different experiment.
- **Multi-seed CI** for the post-overhaul sweep. Single-seed equivalence is enough to gate the rewrite; the multi-seed CI question (Q1 from the prior block's intuition) is a separate followup.
- **`scripts/post_run_beam_eval.py` rewrite.** It calls `beam_eval_walk` which calls `beam_solve_batch`, so it gets the speedup for free. No code change needed there.

## Risks & open questions

- **Memory pressure at the largest batch size.** N × beam_width × 12 × n_stickers × 1 byte at N=200 (V*-eval) × width=512 × 12 × 54 ≈ 66 MB per tensor — fine on 128GB unified memory but the network forward then produces (N × width × 12,) float32 values plus intermediate activations through the [5120, 1024] × 4 residual blocks, easily 5-10 GB transient. **Mitigation**: chunk the batch into pieces of N=50 if memory becomes an issue; the cross-scramble batching still wins big at chunked batches.
- **Hash collision at higher cube sizes.** For 3x3 the collision rate is well-bounded; for hypothetical 4x4 (54 × M stickers) the per-state byte width grows. Document the collision-rate analysis in `state_hash.py` so we revisit if/when the project scales.
- **Test for the perf gate is not deterministic.** Wall-time variance run-to-run is ~5-10%. The gate "≤2 min total" allows ample headroom; no flakiness expected, but if ambient load on the M4 Max is high during the run, re-run.
- **`torch.topk` and `torch.unique` and `torch.scatter_reduce` MPS support.** All three should work on MPS as of recent PyTorch; verify support before C3/C4 by writing a 5-line probe at the top of each commit's branch. If any has bugs, fall back to a CPU-side implementation for that op (one round-trip per step is still better than the current many).

## Sequencing

**C1 → C2 → C3 → C4 → C5 → C6.** Each commit lands independently and keeps the test suite green. C5 produces the user-visible perf and equivalence verdict (rewrite is safe). C6 builds the sampling profiles on top of the now-fast primitive and produces the validated cycle-eval recommendation.

Hard dependencies:
- C2-C4 depend on C1 (cross-scramble batching is the architectural foundation).
- C5 depends on C1-C4 (it verifies the rewrite as a whole).
- C6 depends on C5 (the `thorough` profile would be impractical without the perf gain).

Total estimated time: 1–2 days of work + ~30 min to run the equivalence sweep (C5) + ~30 min to run the validation experiment (C6).
