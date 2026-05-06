# davi-3x3

DAVI training experiments on the 3x3 cube. Mirrors the `davi-2x2` layout —
same `run.py` / `eval.py` / `calibrate_step_time_3x3.py` /
`build_eval_set_3x3.py` / `analysis/` shape — with `CUBE_2X2` swapped for
`CUBE_3X3` end-to-end. The shared training/search/eval stack already
parameterizes on `CubeSpec`, so the production code path is identical to
the 2x2 path.

**Active scope:** phases 1 + 2 of M8 (eval scaffold + smoke training).
Phase 3+ (axis sweeps, champion training, beam search at SPEC's d=20+,
10M tx/sec verification) gets replanned after phase 2 lands. See
[`plans/m8-3x3-davi.md`](../../plans/m8-3x3-davi.md) for full scope.

## Why this dir is mostly skeletons in the P1a scaffold

3x3 has 4.3 × 10¹⁹ states; a full BFS V* oracle is impossible. Phase 1
accepts that as a permanent constraint and scopes early M8 training as
**smoke-testing the stack** — does loss decrease, does eval produce
charts, what does the cost surface look like vs 2x2. Acceptance-grade
evaluation against SPEC.md's M8 criteria (greedy ≥ 95% on d ≤ 15, beam
4096 → 100% at d=20, ≤24 mean moves, < 10s/scramble, 10M tx/sec) is
**deferred to a later plan extension**.

So this dir lands empty-shaped first:

| File | State at P1a end | Earned in |
|---|---|---|
| `run.py` | full | P1a (scaffold) |
| `eval.py` | full | P1a (scaffold) |
| `calibrate_step_time_3x3.py` | shell — empty grid, raises until filled | P2a |
| `build_eval_set_3x3.py` | not yet authored | P1c |
| `analysis/analyze.py` | full — empty `RUNS`, writes stub `results.md` | P1a (scaffold), `RUNS` populated as cycles land |
| `analysis/capture_solve_histograms.py` | full — empty `RUNS`, architecture constants TBD, stubs out cleanly | P2b populates RUNS + arch constants |
| `analysis/render_error_trajectories.py` | shell — overlay charts; full chart suite ports forward as cycles land | P2b/P3+ |
| `analysis/audit_sampler.py` | full — short-circuits if bounded oracle absent | P1b enables (with bounded oracle) |
| `configs/` | empty (`.gitkeep`) | P2b authors first config |
| `eval_set_3x3.npz` | not yet built | P1c |

## Bounded V* sanity oracle (P1b)

A small reverse-BFS V* table covering depths 0..6 is built at
`data/v_star_bounded_3x3_k6.npz` (~1M states, ~60–80MB packed). Used as
**sanity-check ground truth at d ≤ 6** (e.g. for the
`v_star_stratified` strategy in `capture_solve_histograms.py`, and for
the audit_sampler's true-V* histogram). For d > 6 there is no oracle —
the eval set's `v_star` field carries a `-1` sentinel for those bins,
and analysis collapses to walk-depth.

The cycle-eval methodology (CLAUDE.md "Evaluating training cycles
correctly") carries forward with one caveat: "both sampling strategies"
(random_walk_depth + v_star_stratified) collapses to **only**
random_walk_depth at d > 6. At d ≤ 6 we can run both — and the bounded
oracle exists partly to surface walk-redundancy hazards on 3x3 the way
the V*-stratified eval did on 2x2 (cycle-4 lesson).

## Earn every hyperparameter — do not borrow

The 2x2 cycle-3 winning config (`body_widths=[4096,1024]`,
`n_residual_blocks=4`, `batch_size=4096`, `target_sync_interval=500`,
`max_scramble_depth=20`, etc.) is a 2x2 result; **its values do not
transfer to 3x3**. P2a's calibration produces the 3x3 starting config
the same way M5's tier 0 produced the 2x2 one. Reaching for the 2x2
winners as starting points is exactly the cheat the methodology exists
to avoid (CLAUDE.md "Earn every hyperparameter").

So: `configs/` ships empty. `calibrate_step_time_3x3.py` ships with an
empty grid. `capture_solve_histograms.py` ships with `BODY_WIDTHS=None`.
P2a and P2b fill these in based on **3x3 measurements**.

## Pipeline shape (mirrors davi-2x2)

`analysis/` has the canonical analyze/capture/render layer:

- `analyze.py` reads each run's `metrics.jsonl` and writes
  `results/results.md` with per-run sections.
- `capture_solve_histograms.py` runs post-hoc N=200 solve-length
  captures on each run's terminal checkpoint, across
  `random_walk_depth × v_star_stratified` × `greedy × beam(256)`
  (where the oracle exists), into `results/solve_histograms.json`.
- `render_error_trajectories.py` consumes both and produces
  `results/error_trajectories.html` with all runs overlaid.

Each script has a `RUNS:` table at the top — a new cycle is "append a
tuple, regenerate." Per CLAUDE.md "Cycle reporting pipeline (don't
parallel-build)": new sampling strategies or metric families extend
these scripts with parameters, never a parallel script or HTML under
`scripts/` or `results/cycle-N-*.html`.

## Pointers

- `plans/m8-3x3-davi.md` — full plan with phase 1 + phase 2 scope.
- `experiments/davi-2x2/README.md` — methodology blueprint.
- `experiments/davi-2x2/results/` — reference for the canonical
  `error_trajectories.html` chart suite that this dir's renderer ports
  forward as cycles land.
- CLAUDE.md sections: "Earn every hyperparameter — do not borrow",
  "Cycle reporting pipeline (don't parallel-build)", "Evaluating
  training cycles correctly", "Cube / cubie naming".
