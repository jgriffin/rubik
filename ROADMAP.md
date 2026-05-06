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
- **M5** — Scramble pipeline + DAVI training (2x2, smoke-tested against BFS V\*) — *not yet planned*
- **M6** — Beam search (2x2 vs BFS-optimal) — *not yet planned*
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

### W&B integration for DAVI training observability
Flip the W&B passthrough in `MetricLogger` (added in M5 commit 5) at the tier 1 → tier 2 boundary, when ~10 runs accumulate and the analysis question shifts from "is this run healthy?" to "which configs win along which axis?" Cost: ~30 min — `uv add wandb`, `wandb login`, flip the flag, smoke-test the dashboard. JSONL stays the source-of-truth; W&B is for visualization + cross-run comparison. Reverses the bootstrap stack-purity decision to drop tensorboard/wandb/matplotlib — make consciously, when payoff is real. **MPS gotcha:** W&B's GPU telemetry tab is empty on Apple Silicon (NVIDIA-only); macmon stays our GPU-utilization source for M8 perf work. Skip W&B Sweeps / Hyperband — orchestration bypasses intuition formation and the early-stopping heuristic collides with the cube DAVI proxy-gap risk. Hold for M7 (systematic hparam characterization is the actual milestone goal).
Surfaced: 2026-05-02
