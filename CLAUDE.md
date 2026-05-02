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
- **Move set.** QTM only at first — 12 quarter-turn moves (6 faces × 2
  directions). No double moves (R2). Indexed 0..11 with the mapping
  documented in `src/rubik/notation/`.
- **Parameterized cube.** Single source of truth: a `CubeSpec` describes
  cube size, sticker count, faces, move tables, color count. The whole
  pipeline (env, oracle, training, search, viz) consumes `CubeSpec`. **No
  side-by-side 2x2 / 3x3 branches.** DRY.
- **Cube / cubie naming.** `cube` and `cubie` are *generic* — they apply
  to both 2x2 and 3x3. Anything specific to one size MUST carry `_2x2`
  or `_3x3` in its name: constants like `CUBE_2X2` / `CUBE_3X3`, files
  like `oracle_rotations_2x2.html`, scripts like
  `render_oracle_rotations_2x2.py`, test functions like
  `test_cube_2x2_basic_fields`, etc. The goal: `grep "2x2"` (or `3x3`)
  finds every size-specific identifier in the repo. `2x2` is verbose
  but unambiguous and search-friendly; the alternative `2`/`3` suffix
  collides with version numbers, dimension counts, and other unrelated
  numerics. Strings inside data (e.g. `CubeSpec(name="2x2")`) are not
  separate — they match the same grep, which is fine.
- **Visuals.** Out-of-tree, human-eyeballable HTML/SVG artifacts live
  in `visuals/` at the repo root, with their generators in
  `visuals/scripts/generate_<thing>.py`. Both the artifact and its
  generator are checked in. This is distinct from `src/rubik/viz/` which
  holds production renderers shipped with the package (M3+).
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
  `results.md`. Established in M4, reused in M8 for hyperparams.

## Pointers

- **`SPEC.md`** — project goal, key decisions with rationale, full M0–M9
  milestone plan with acceptance criteria, package layout, references.
- **`ROADMAP.md`** — current milestone status and backlog.
- **`LOG.md`** — work blocks, newest on top (cc-process).
- **`plans/m<N>-<name>.md`** — written when each milestone block opens.
- **`plans/archive/llm-draft-spec.md`** — original draft, retained for
  technical details (sticker indexing, network architecture, hyperparams).
  Do **not** treat as authoritative — superseded by `SPEC.md`.
