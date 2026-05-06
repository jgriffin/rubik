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
- **M8** — 3x3 enablement: edge cubies in oracle + 3x3 move-perm + rerun M2–M7 verification on 3x3 — *not yet planned*
- **M9 (stretch)** — 3D / web frontend + solution-trace analysis — *not yet planned*

> **Sequencing note.** 2x2 carries through end-to-end (env → train → search → perf-2) before 3x3 lands. Rationale: 2x2's small state space (~3.6M reachable) makes BFS-optimal V\* a per-state ground truth — strongest possible signal for training and search correctness — and iterates 100× faster than 3x3. The `CubeSpec` abstraction is the bet that the 3x3 swap stays cheap; M8 is where we cash that bet.

## Backlog

<!--
When an idea surfaces that isn't the current block's goal, append here:

### <short title>
<1-3 lines of context>
Surfaced: YYYY-MM-DD
-->

### M5-followup: V*-stratified scramble resampling (close M6 gap)
M6 acceptance gate at d=10..14 missed by 0.3pp (d=10) up to 4.7pp (d=14). M5 cycle-2 sampler audit located the structural cause: random walks of length K_max=20 under-represent natural-state-distribution-mode true depths (mode is true-d=12 in 2x2's 3.6M states; walks land mostly on true-d=8..11). One concrete lever: at training time, draw scramble batches *with depth resampled against the natural V\* distribution* — index into `data/v_star_2x2.npz` to weight per-state batch composition so the trained net sees natural-distribution-tail states proportionally. 2x2-only proof of concept (V\* oracle exists; 3x3 doesn't have one — methodology question of how to port is part of the cycle). One DAVI cycle ≈ 1–3 hours wall on M4 Max. Run, eval against V\*, eval against M6 gate at width=256.
Surfaced: 2026-05-05

### M5-followup: curriculum scheduling for DAVI scramble depth
Alternative (or complement) to V\*-stratified resampling. K_max grows with training step rather than fixed. Methodology-portable to 3x3 (no V\* required). Trade-off: less direct attack on the data-distribution constraint than stratified resampling, but generalizes. Surfaced as a candidate cycle 4 lever in M5 cycle-3 close.
Surfaced: 2026-05-05

### M5-followup diagnostic: locate M6 d=14 failure states vs true-V\*
Cheap (<1 min compute) follow-up to verify intuition.md H2: regenerate the 48 failed scrambles at d=14 width=256 (seed `config.seed + 14 = 14`, deterministic), look up `V*[state]` for each, and check whether failures cluster at true-V\* = 12–14 or are uniform over true-V\*. If H2 confirmed, V\*-stratified resampling becomes the leading retraining recommendation; if falsified, the diagnosis is something else (e.g. specific symmetry classes V\_θ mis-orders) and the retraining strategy shifts. Drop into either of the two retraining backlog items above as a pre-experiment.
Surfaced: 2026-05-05

### W&B integration for DAVI training observability
Flip the W&B passthrough in `MetricLogger` (added in M5 commit 5) at the tier 1 → tier 2 boundary, when ~10 runs accumulate and the analysis question shifts from "is this run healthy?" to "which configs win along which axis?" Cost: ~30 min — `uv add wandb`, `wandb login`, flip the flag, smoke-test the dashboard. JSONL stays the source-of-truth; W&B is for visualization + cross-run comparison. Reverses the bootstrap stack-purity decision to drop tensorboard/wandb/matplotlib — make consciously, when payoff is real. **MPS gotcha:** W&B's GPU telemetry tab is empty on Apple Silicon (NVIDIA-only); macmon stays our GPU-utilization source for M8 perf work. Skip W&B Sweeps / Hyperband — orchestration bypasses intuition formation and the early-stopping heuristic collides with the cube DAVI proxy-gap risk. Hold for M7 (systematic hparam characterization is the actual milestone goal).
Surfaced: 2026-05-02
