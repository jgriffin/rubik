@~/.claude/cc-process.md

# rubik — Claude project notes

Deep RL solver for the Rubik's Cube targeting Apple Silicon (M4 Max, MPS
backend). Lineage is DeepCubeA / EfficientCube: train a value network via
DAVI (approximate value iteration) on backward-generated random scrambles,
then solve new scrambles via batched GPU beam search. We develop the full
pipeline (env → train → solve) on the **2x2** cube first, then enable 3x3
by swapping a `CubeSpec` — same code path, parameterized.

## Conventions

- **Package layout.** Code lives under `src/rubik/`. Tests under `tests/`.
  Per-milestone plans under `plans/m<N>-<name>.md`. Perf/hyperparam
  experiments under `experiments/<name>/` with a `results.md` per dir.
- **Python tooling.** `uv` for everything — `uv add`, `uv run`, `uv venv`.
  Never `pip` / `pip3` / `python3` directly. Invoke Python as `python`
  inside `uv run`. Pinned in `.python-version` (3.12).
- **Tests.** `pytest`. Run with `uv run pytest`. Use `pytest-snapshot` for
  visualization regression. Property-based tests for cube identities.
- **Lint + format.** `ruff` does both — `uv run ruff check` and
  `uv run ruff format`. No black/flake8/isort.
- **Markdown line wrapping.** Don't manually wrap lines in markdown files (CLAUDE.md, LOG.md, ROADMAP.md, results.md, intuition.md, plans/, etc.). Let the editor soft-wrap. Manual ~80-col breaks fragment readability for users on wider editors. Code line length is a separate rule, set by the formatter config.
- **Earn every hyperparameter — do not borrow.** Specific values for things like `body_widths`, `n_residual_blocks`, `learning_rate`, `batch_size`, `target_sync_interval`, `n_steps`, `max_scramble_depth` are an empirical question for this hardware and this problem. Pick them by running tier 0+ experiments. Do **not** reach for values from prior published work (DeepCubeA, EfficientCube, etc.) as starting points, "reference runs," "baselines for comparison," or fallback defaults — that shortcut is what the methodology exists to avoid. The general algorithmic lineage (DAVI / value iteration with target net + curriculum scrambles + MLP value head) is fine to reference; specific numeric choices are not. If you find yourself thinking "I'll just start with what they used," stop — that's the cheat.
- **Move set.** QTM only at first — 12 quarter-turn moves (6 faces × 2
  directions). No double moves (R2). Indexed 0..11 with the mapping
  documented in `src/rubik/notation/`.
- **Parameterized cube.** Single source of truth: a `CubeSpec` describes
  cube size, sticker count, faces, move tables, color count. The whole
  pipeline (env, oracle, training, search, viz) consumes `CubeSpec`. **No
  side-by-side 2x2 / 3x3 branches.** DRY.
- **Cube / cubie naming.** `cube` and `cubie` are *generic* — they apply
  to both 2x2 and 3x3. Anything specific to one size MUST carry `_2x2`
  or `_3x3` in its name, **placed at the end** so the purpose reads
  first and the size variant trails: constants like `CUBE_2X2` /
  `CUBE_3X3`, files like `oracle_rotations_2x2.html`, scripts like
  `render_oracle_rotations_2x2.py`, test functions like
  `test_cube_basic_fields_2x2`, etc. The goal: `grep "2x2"` (or `3x3`)
  finds every size-specific identifier in the repo. `2x2` is verbose
  but unambiguous and search-friendly; the alternative `2`/`3` suffix
  collides with version numbers, dimension counts, and other unrelated
  numerics. Strings inside data (e.g. `CubeSpec(name="2x2")`) are not
  separate — they match the same grep, which is fine.
- **Visuals + scripts.** Human-eyeballable HTML/SVG artifacts live in
  `visuals/` at the repo root with their generators in
  `visuals/scripts/render_<what>_<size>.py` producing
  `visuals/<what>_<size>.html`. Pure-code dev scripts (codegen, snapshot
  derivation, etc.) — anything that doesn't produce a visual artifact —
  live in top-level `scripts/`. Both are distinct from `src/rubik/viz/`
  which holds production renderers shipped with the package (M3+).
- **Dual witnesses for correctness.** Two genuinely independent cube
  implementations:
  1. A slow hand-rolled **cubie oracle** (corners as position+orientation,
     moves applied as physical rotations) — readable, obviously correct,
     extends from 2x2 to 3x3 by adding edges.
  2. A fast **tensor cube** (precomputed sticker permutations, batched on
     MPS) — the production path.
  Equivalence-tested on a corpus of random move sequences. We dropped
  pycuber as a third witness — 3x3-only and unmaintained since 2015.
- **Visualization.** ASCII renderer for tests / REPL / CI. HTML/SVG render
  components written to file and inspected via `open <file>` — the global
  "static HTML preview" pattern. **Avoid matplotlib** unless we hit a
  specific need it solves. 3D / web frontend is M9 stretch.
- **Make it work, then make it fast.** Correctness milestones precede perf
  milestones. M2 is "correct tensor impl" (no perf bar); M4 is the perf
  milestone, where we also produce the methodology for measuring MPS
  utilization on this machine.
- **Experiment-loop pattern.** Each perf-sensitive milestone gets an
  `experiments/<name>/` directory with reproducible scripts and a
  `results.md`. Established in M4, reused in M7 for hyperparams.
- **Experiment results.md format.** Every experiment dir's `results.md`
  ends with an `## Intuition` section in epistemically honest format:
  **Observations** (mechanical, from this run's data) → **Hypotheses**
  (interpretive claims with confidence level + supporting evidence +
  explicit verification plan) → **Open questions** (well-defined next
  experiments) → optionally **What we haven't verified** (caveats on
  the most speculative claims). Datestamp + run conditions at the top so
  stale interpretations are obvious. Hand-written in
  `experiments/<name>/intuition.md`; `analyze.py` reads it and appends to
  `results.md` so the section persists across regenerations. Goal:
  capture conceptual learning per experiment so a human can read just
  the intuition section to internalize what we learned, **without**
  anchoring future agents — the format invites investigation rather than
  asserting conclusions. Distinguish observation from inference from
  hypothesis; hypotheses get verification plans, not authoritative tone.
- **Cycle reporting pipeline (don't parallel-build).** Each
  `experiments/<name>/` has an `analysis/` layer with three roles:
  `analyze.py` reads per-run `metrics.jsonl` and writes
  `results/results.md`; `capture_*.py` runs post-hoc captures (e.g.
  N=200 solve-length histograms) on each run's final checkpoint into raw
  JSON under `results/`; `render_*.py` consumes both and produces ONE
  canonical comparison HTML at `results/<name>.html` with **all runs
  overlaid**. Each script has a `RUNS:` table at the top — a new cycle
  is "append a tuple, regenerate." New sampling strategies or metric
  families extend the existing capture + renderer with parameters,
  **never a parallel script or HTML** under `scripts/` or
  `results/cycle-N-*.html`. The 5/5/4 banded small-multiples convention
  (Shallow d=1–5 / Mid d=6–10 / Deep d=11–14) documented in
  `render_error_trajectories.py` applies to every per-depth chart.
- **Evaluating training cycles correctly.** Four things must hold:
  (1) **Right baseline**: a warm-start cycle is evaluated against its
  immediate predecessor (the checkpoint it warm-started from), NOT
  cycle-1. Cycle N+1 vs cycle N tells you the marginal effect of cycle
  N+1; vs cycle-1 measures cumulative everything-to-date and can hide a
  regression inside a long-run improvement curve.
  (2) **Both sampling strategies**: `random_walk_depth` (length-d walks,
  V*≤d, walk-distribution-biased) and `v_star_stratified` (states with
  true V*=d, sampled from the BFS oracle). Walk redundancy dilutes
  deep-V* mass — a length-14 walk lands mostly on V*<14 states — so
  random_walk_depth can mask regressions concentrated at the deepest
  true V* that v_star_stratified surfaces directly.
  (3) **Both solve methods**: greedy AND beam(256). Greedy at deep V*
  is ~0% on every checkpoint trained so far, so greedy-only eval flatlines
  and hides ordering-quality changes. Beam consumes ordering; that's
  where cycle-deltas live for production search.
  (4) **macro_mae alone is not the verdict**: macro_mae is a calibration
  metric. Search capability is a different one. They can diverge —
  cycle-4 improved macro_mae 3.13→2.87 while LOSING 24-30pp of beam
  capability at v_star_stratified d=11..13. Always cross-check
  capability metrics on v_star_stratified × beam before accepting a
  cycle as an improvement.

## Pointers

- **`SPEC.md`** — project goal, key decisions with rationale, full M0–M9
  milestone plan with acceptance criteria, package layout, references.
- **`ROADMAP.md`** — current milestone status and backlog.
- **`LOG.md`** — work blocks, newest on top (cc-process).
- **`plans/m<N>-<name>.md`** — written when each milestone block opens.
