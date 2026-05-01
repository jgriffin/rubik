# ROADMAP

Forward-looking. See `LOG.md` for what's been done, `SPEC.md` for the
full project spec and per-milestone acceptance criteria. Per-milestone
plans get written to `plans/m<N>-<name>.md` when each block opens (plan
mode produces them; we don't pre-create stubs).

## Milestones

- **M0** — Repo skeleton, `CubeSpec` abstraction, notation hub scaffolding — ✅ done ([plan](plans/m0-skeleton.md))
- **M1** — Slow cubie oracle (2x2), hand-rolled, generic enough for 3x3 later — ✅ done ([plan](plans/m1-cubie-oracle.md))
- **M2** — Fast tensor cube (2x2), correctness only (oracle equivalence + identities) — *not yet planned*
- **M3** — Visualization stack: ASCII + HTML/SVG, first human-verifiable checkpoint — *not yet planned*
- **M4** — Perf-1: MPS measurement methodology + batch sensitivity + experiment-loop pattern — *not yet planned*
- **M5** — Scale to 3x3 via `CubeSpec` swap; rerun M1–M4 verification — *not yet planned*
- **M6** — Scramble pipeline + DAVI training (2x2 smoke test against BFS V\*, then 3x3) — *not yet planned*
- **M7** — Beam search (2x2 vs BFS-optimal, then 3x3) — *not yet planned*
- **M8** — Perf-2 / hyperparam experiment loop on 3x3 training — *not yet planned*
- **M9 (stretch)** — 3D / web frontend + solution-trace analysis — *not yet planned*

## Backlog

<!--
When an idea surfaces that isn't the current block's goal, append here:

### <short title>
<1-3 lines of context>
Surfaced: YYYY-MM-DD
-->
