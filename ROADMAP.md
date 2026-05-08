# ROADMAP

Forward-looking. See `LOG.md` for what's been done, `SPEC.md` for the
full project spec and per-milestone acceptance criteria. Per-milestone
plans get written to `plans/m<N>-<name>.md` when each block opens (plan
mode produces them; we don't pre-create stubs).

## Milestones

- **M0** — Repo skeleton, `CubeSpec` abstraction, notation hub scaffolding — ✅ done ([plan](plans/m0-skeleton.md))
- **M1** — Slow cubie oracle (2x2), hand-rolled, generic enough for 3x3 later — ✅ done ([plan](plans/m1-cubie-oracle.md))
- **M2** — Fast tensor cube (2x2), correctness only (oracle equivalence + identities) — ✅ done ([plan](plans/m2-tensor-cube.md))
- **M3** — Visualization stack: ASCII + HTML/SVG, first human-verifiable checkpoint — ✅ done ([plan](plans/m3-viz.md))
- **M4** — Perf-1: MPS measurement methodology + batch sensitivity + experiment-loop pattern — ✅ done ([plan](plans/m4-perf-1.md))
- **M5** — Scramble pipeline + DAVI training (2x2, smoke-tested against BFS V\*) — ✅ done ([plan](plans/m5-davi.md))
- **M6** — Beam search (2x2 vs BFS-optimal) — ✅ done with documented gap ([plan](plans/m6-beam-search.md)). At width=256, n=1000: d=1..9 perfect, d=10..14 in [0.952, 0.997] — under SPEC's ≥0.999 per-depth gate. Mean V\*-excess 0.248 (gate ≤1.0) — ✅. Capability is binding constraint, not solution length; cause traced to V\_θ ordering signal at deep states (random-walk training under-samples the natural-state distribution mode at d=11–13 — M5 cycle-2 finding). Gap revisitable via the M5-followup retraining backlog items below.
- **M7** — Perf-2 / hyperparam experiment loop on 2x2 training — *not yet planned*
- **M8** — 3x3 enablement — *active*
  - 3x3 env: CubeSpec + cubie oracle + tensor cube + viz + equivalence — ✅ done (LOG 2026-05-06)
  - 3x3 DAVI scaffold + T0 capacity calibration (M8 phases 1+2, "does 3x3 train?") — ✅ done (LOG 2026-05-06)
  - 3x3 DAVI champion cycles (M8 phase 3+) — *active; two attempts completed*: (a) full_train K=20 from random init reached d=14=0.78 at step 16k before early-stop misfired; (b) warm_continue K=20 from step 16k regressed to d=14=0.24 over 100k steps (slow-drift collapse, see LOG 2026-05-07). Next experiment: warm-continue with K_max raised to 25 or 30 to test "escape the local max" hypothesis.
  - 3x3 beam search + perf verification (10M transitions/sec target) — *upcoming*
- **M9 (stretch)** — 3D / web frontend + solution-trace analysis — *not yet planned*

> **Sequencing note (revised 2026-05-06).** Original plan was 2x2 end-to-end (env → train → search → perf-2) before 3x3. We hit diminishing returns on the 2x2 training side after M5 cycle-4: M6 has a documented deep-depth gap, M5-followup has three plausible levers, but each costs hours per cycle and the marginal scientific yield is shrinking. Pivoting to 3x3 now — broader information per hour, exercises the `CubeSpec` bet at the training/search level (it already paid off at the env level in M8 bringup), and the 2x2 followup work stays revisitable. The 2x2 V\* oracle remains a unique asset for ground-truth eval, just not the active surface.

## Backlog

<!--
When an idea surfaces that isn't the current block's goal, append here:

### <short title>
<1-3 lines of context>
Surfaced: YYYY-MM-DD
-->

### M8 backlog: checkpoint metadata bundle (.pt files carry step + config)
Today the eval tooling parses filenames (`net_step_26000.pt` → step=26000) and resolves the sibling `config.yaml` for arch metadata. This is fragile — easy to lose the step if a checkpoint gets renamed/copied, and parameter values that change across training (K_max ramp, learning-rate schedule, sync interval) aren't recoverable from a bare `.pt`. Idea: write checkpoints as a dict containing `{net_state: ..., step: int, config_snapshot: dict, training_metadata: {wall_time_so_far, optimizer_state_summary, ...}}` instead of a bare state-dict. Eval scripts then read the bundle directly — no filename parsing, no sibling-config dependency. Migration: `_load_net` already handles both bare and dict checkpoints, so backwards-compat is free; new code path writes the richer bundle on save. Affects the training-loop save site (`src/rubik/training/davi.py` or wherever the save happens) and the eval scripts that consume `step` (`beam_eval_run.py`'s filename regex).
Surfaced: 2026-05-07

### M8 backlog: chunked forward at high beam widths
Cross-scramble batching from m8 perf rewrite pushes the forward batch past the MPS throughput knee (~150-200k states for the [5120, 1024]×4 ValueNet) at widths ≥256. BF16 mostly resolved width=256 (knee shifts to ~300k), but width=512 (614k states/step at N=100) is still ~5× slower than the pre-rewrite per-scramble baseline. Fix: wrap `net(flat_children)` in `src/rubik/search/beam.py` with a chunking helper that splits inputs >150k states into ≤150k chunks, runs each, concatenates outputs. Likely 5× speedup at width=512 with no algorithmic change. Width=512 is diagnostic-tier per prior block, not production cycle decisions, so this is a quality-of-life fix rather than a cycle blocker.
Surfaced: 2026-05-07

### M8 backlog: multi-run trajectory heatmap layout
`render_beam_eval_report.py` trajectory mode renders only the first run's matrix in the heatmap section when multiple runs are passed (banded line chart and walltime line do overlay correctly). Today's `beam_eval_run.py` only ever produces single-run trajectory inputs so this didn't matter, but if we later want to compare two runs' trajectories side-by-side (e.g. cycle-N vs cycle-N-1), the natural extension is one heatmap per run vertically stacked or side-by-side.
Surfaced: 2026-05-07

### M8 backlog: K=8 bounded V\* oracle expansion (warm-start from K=6)
Bounded V\* currently at K=6 (~1M states, 14.9 MB cache, 4.8s build). Extending to K=8 means adding ~80M more states (depth-7 layer 8.2M new, depth-8 layer 70.9M new). Estimated ~10–15 GB working memory in dict form during build, ~6–15 min build time. Warm-start from existing K=6 cache (use the layer-6 frontier as the BFS seed) avoids re-doing layers 0..6. Validation: an "optimal-descent" test (greedy V\* descent from a random V\*=k state should reach solved in exactly k moves) — discussed but not landed yet, lift here when relevant. Useful when 3x3 training pushes solve rates near saturation at d≤6 and we want ground truth at d∈{7,8} too.
Surfaced: 2026-05-06

### M8 backlog: T0 capacity calibration on 3x3
Sweep `batch_size × body_widths × n_residual_blocks` for ms/step on 3x3 (mirrors `experiments/davi-2x2/calibrate_step_time.py`). Earned 3x3-specific architecture replaces the M8 phase-2 hand-picked guess (`[5120, 1024] × 4 bn`). Stub already in place at `experiments/davi-3x3/calibrate_step_time_3x3.py` raising `NotImplementedError` until populated. Likely runs after Phase 2 smoke completes — at that point we have signal on whether the hand-picked guess is reasonable or undersized.
Surfaced: 2026-05-06

### Drive-by: rename 2x2 wandb project from `rubik` to `rubik-2x2`
3x3 work goes to a separate `rubik-3x3` project to prevent cross-contamination. For symmetry, rename the existing 2x2 project from `rubik` to `rubik-2x2`. Note: wandb project renames may break links in old run dirs / README references. Decide whether to rename via UI (lossless for the project; existing run URLs may auto-redirect) or freeze old runs in `rubik` and create `rubik-2x2` afresh for new work.
Surfaced: 2026-05-06

## Paused — 2x2 M5-followup (post-3x3)

Parked 2026-05-06 when we pivoted the active surface to 3x3 (see Sequencing note above). 2x2 still has its V\* oracle, the M6 capability gap is real but documented, and any of these items remains a coherent block to open if/when we return. Not abandoned — just out of the active path.

### M5-followup Plan A: continue training with deeper walks (K_max=25+) from sync500_kmax20-30k
User-preferred next experiment for closing the M6 deep-depth gap. Warm-start from `experiments/davi-2x2/runs/sync500_kmax20-30k/net_final.pt` (best 30k checkpoint to date), keep all other hyperparameters identical, raise K_max from 20 → 25 (or further). Hypothesis: deeper random walks during DAVI expose V_θ to natural-state-distribution-tail depths (true-d=12..14) that the M6 acceptance run failed at, without restarting training from scratch. Cheaper than a full retrain (~1–2h from a warm start vs 3h+ from random init); the K_max=20 net is already well-formed. Calibration: M6 `intuition.md` H1 (width=256 is 1–2 doublings short of capability) implies deeper K_max alone may not close the gap without also widening the beam at solve-time. H2 (failures cluster at true-V\*=12..14 due to under-exposure during training) is the lever this plan attacks directly. Run the failure-state diagnostic (below) first to weigh H1 vs H2 before committing the train run. **Caution: cycle-4 (K_max=28 flat warm-start) ran this hypothesis at extreme and broke deep-V\* ordering — see LOG 2026-05-05. Plan A at K_max=25 is milder but the failure mode is on the path; consider V\*-stratified resampling first.**
Surfaced: 2026-05-05

### M5-followup: V*-stratified scramble resampling (close M6 gap)
M6 acceptance gate at d=10..14 missed by 0.3pp (d=10) up to 4.7pp (d=14). M5 cycle-2 sampler audit located the structural cause: random walks of length K_max=20 under-represent natural-state-distribution-mode true depths (mode is true-d=12 in 2x2's 3.6M states; walks land mostly on true-d=8..11). One concrete lever: at training time, draw scramble batches *with depth resampled against the natural V\* distribution* — index into `data/v_star_2x2.npz` to weight per-state batch composition so the trained net sees natural-distribution-tail states proportionally. 2x2-only proof of concept (V\* oracle exists; 3x3 doesn't have one — methodology question of how to port is part of the cycle). One DAVI cycle ≈ 1–3 hours wall on M4 Max. Run, eval against V\*, eval against M6 gate at width=256.
Surfaced: 2026-05-05

### M5-followup: curriculum scheduling for DAVI scramble depth
Alternative (or complement) to V\*-stratified resampling. K_max grows with training step rather than fixed. Methodology-portable to 3x3 (no V\* required). Trade-off: less direct attack on the data-distribution constraint than stratified resampling, but generalizes. Surfaced as a candidate cycle 4 lever in M5 cycle-3 close. **Note:** infrastructure for this already exists — `DAVIConfig.max_scramble_depth_initial` + `max_scramble_depth_ramp_steps` landed in cycle-4 (commit `025af65`) but were never exercised end-to-end. Curriculum + V\*-stratified-final-distribution is the most-promising compound version.
Surfaced: 2026-05-05

### M5-followup diagnostic: locate M6 d=14 failure states vs true-V\*
Cheap (<1 min compute) follow-up to verify intuition.md H2: regenerate the 48 failed scrambles at d=14 width=256 (seed `config.seed + 14 = 14`, deterministic), look up `V*[state]` for each, and check whether failures cluster at true-V\* = 12–14 or are uniform over true-V\*. If H2 confirmed, V\*-stratified resampling becomes the leading retraining recommendation; if falsified, the diagnosis is something else (e.g. specific symmetry classes V\_θ mis-orders) and the retraining strategy shifts. Drop into either of the two retraining backlog items above as a pre-experiment.
Surfaced: 2026-05-05
