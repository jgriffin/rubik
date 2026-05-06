# SPEC — Rubik's Cube Deep RL

Long-form project spec. Companion to `ROADMAP.md` (status) and `LOG.md`
(history). For per-milestone plans see `plans/m<N>-<name>.md` (written when
each milestone block opens). The original draft is preserved at
`plans/archive/llm-draft-spec.md` for technical details only — this file
supersedes it.

## Project goal

Build a deep RL solver for the Rubik's Cube that runs end-to-end on Apple
Silicon (M4 Max, MPS backend). The solver should reliably solve scrambled
cubes via batched GPU beam search, guided by a value network trained with
**DAVI** (approximate value iteration) on backward-generated random
scrambles. Targets at convergence: high solve rate at God's-Number-class
scramble depths, near-optimal solution lengths on 2x2 (where BFS-optimal is
computable), and competitive lengths on 3x3 (the DeepCubeA / EfficientCube
benchmark range).

The work is also a vehicle for two engineering-side questions: (a) what
PyTorch patterns push MPS hard and how do we measure that reliably on this
machine; (b) does an off-the-shelf DAVI + beam-search recipe converge to
recognizably human (CFOP-like) solving behavior, or to something alien.

## Approach

DeepCubeA / EfficientCube lineage:

1. **Vectorized cube environment.** Cube state is a small int tensor;
   moves are precomputed sticker permutations; a step is a `gather`, not a
   matmul. The env runs batched on MPS — no Python loops in the hot path.
2. **DAVI training.** Generate random scrambles from the solved state.
   For each state, look at all children, set the bootstrap target as
   `min_a (1 + V_target(child))` (with terminal child clamped to 0).
   Regress `V_θ` toward the target; periodically sync `V_target ← V_θ`.
3. **Batched beam search.** At inference, expand the beam, score children
   via `-V`, dedup, take top-B. Solution is the action sequence reaching
   the solved state. Search lives on GPU.
4. **2x2 first, parameterized.** Develop the full pipeline on 2x2, then
   enable 3x3 by swapping a `CubeSpec`. 2x2 has BFS-optimal `V*` available
   (state space ~3.7M), giving us a strong correctness signal that 3x3
   alone can't provide.

## Key decisions

### 1. 2x2 first, parameterized via `CubeSpec`

Rationale: 2x2 is small enough that we can compute BFS-optimal `V*` for
every reachable state, which gives us a per-state ground-truth signal for
any value network we train. That makes the *training pipeline* itself
testable in a way it isn't on 3x3. The cost is a `CubeSpec` abstraction —
size, sticker count, face / move tables, color count — that everything
downstream consumes. The benefit is a single code path to maintain. We
explicitly reject side-by-side 2x2 / 3x3 forks: same code, different spec.

### 2. Two witnesses for cube correctness

Rationale: a single tensor implementation can be self-consistently wrong
(e.g., a permutation that satisfies `M⁴ = I` but encodes the wrong rotation
sense). We pair the fast tensor cube with a hand-rolled **cubie oracle** —
corners stored as position + orientation (and edges, when we extend to
3x3), moves applied as physical 90° rotations. The two implementations
share no code. Equivalence is asserted on a large corpus of random move
sequences, plus the standard identities (`M⁴ = I`, `(R U R' U')⁶ = I`,
`Sune⁶ = I`, color-multiset preservation). We considered a third witness
via `pycuber` and dropped it: 3x3-only, unmaintained since 2015. We can
revisit if the two-witness setup leaves us uncertain.

### 3. QTM only

Rationale: 12 moves (6 faces × 2 directions), action indices 0..11. No
double moves (`R2`) at first. QTM is the cleanest move set for pruning
(same-face-twice is always redundant) and for building the move tables.
HTM (with double moves) is a deferred backlog item — easy to add later
once QTM is solid, costs nothing to defer.

### 4. Notation hub

Rationale: a cube state has many useful representations — flat tensor,
cubie struct, unfolded face dict, ASCII string, SVG layout. Moves
likewise — index, string ("R", "U'"), face+direction tuple. Renderers,
validators, tests, and humans all need different forms. The right answer
is a first-class `src/rubik/notation/` module with explicit converters
between representations, rather than ad-hoc transforms scattered across
viz / tests / oracle. Renderers consume notations, not raw tensors.

### 5. Visualization is its own milestone (M3)

Rationale: visualization is how a human verifies the cube is correct. It's
also the user's "favorite pattern" — static HTML previews built up
alongside the artifact, opened with `open <file>`, no dev server. We give
viz dedicated milestone time so the ASCII + HTML/SVG renderers exist
**before** we start trusting any large test suite. Three layers:

- **ASCII** — for `pytest -s`, REPL, CI logs, doctests.
- **HTML/SVG render components** — sequence grids, scramble traces,
  side-by-side comparisons, snapshot tests. Static files under
  `assets/` or per-test fixtures. **Avoid matplotlib** unless we hit a
  specific need it solves better than HTML/SVG (we don't expect to).
- **Web / 3D frontend** — deferred to M9 stretch.

### 6. "Make it work, then make it fast"

Rationale: a fast wrong cube is worse than a slow right cube — every
downstream bug becomes ambiguous. Correctness milestones (M1, M2) ship
before perf milestones (M4). M2's acceptance criterion is *correctness
only* — equivalence to the oracle on a 10k-sequence corpus plus all
identities. We move the throughput target to M4. The MPS-measurement
methodology (how to read GPU utilization reliably on macOS — `macmon`,
`mactop`, torch profiler trace exports, hyperfine for end-to-end) is
itself a deliverable of M4, not just an output.

### 7. Experiment-loop pattern

Rationale: perf and hyperparam work is a search problem with many wrong
answers. We treat each perf-sensitive milestone as an experiment dir
under `experiments/<name>/` with reproducible scripts (`run.py`,
`config.yaml`), an output dir (`runs/`, gitignored), and a `results.md`
that accumulates findings. M4 established the pattern; M7 reuses it for
training hyperparams (2x2), and M8 reuses it again for the 3x3
throughput rerun.

### 8. pycuber dropped

Rationale: considered as a third witness alongside the cubie oracle and
tensor cube. Two strikes: 3x3-only (so it can't witness 2x2, where most
of our development happens) and last commit in 2015. Hand-rolled oracle
only. Revisit if we hit ambiguity that two witnesses can't resolve.

## Tech stack

- **Python 3.12**, pinned in `.python-version`.
- **PyTorch** (MPS backend). 2.x. Author runs on M4 Max.
- **numpy** for ergonomics in tests / experiments. Not in hot paths.
- **uv** for everything — package management, venvs, script invocation.
- **pytest** + **pytest-snapshot** for tests and viz regression.
- **ruff** for lint + format.

Deferred / not yet added: tensorboard, wandb, matplotlib, hydra. Add when
a milestone genuinely needs them, not before.

## Package layout

Scaffolded at M0 (some dirs created empty with `__init__.py`, others grow
in their owning milestone):

```
src/rubik/
  cube/                # tensor cube + CubeSpec — M0 scaffold, M2 fills
    __init__.py
    spec.py            # CubeSpec dataclass: size, faces, move tables, ...
    state.py           # encoding constants, solved state
    env.py             # batched env: apply_moves, is_solved, scrambles
  oracle/              # slow cubie reference impl — M1
    __init__.py
    cubie.py           # corners (and edges for 3x3) as pos+orientation
  notation/            # representations + converters — M0 scaffold
    __init__.py
    moves.py           # move index ↔ string ↔ (face, dir) tuple
    state.py           # tensor ↔ unfolded face-dict ↔ cubie struct
  viz/                 # renderers, consume notations — M3
    __init__.py
    ascii.py
    svg.py
  model/               # value network — M5
    __init__.py
    network.py
  training/            # DAVI loop + scramble pipeline — M5
    __init__.py
    davi.py
    scrambles.py
    config.py
  search/              # batched beam search — M6
    __init__.py
    beam.py
    pack.py            # state packing for dedup
  __init__.py

tests/                 # mirrors src/rubik/ structure
plans/
  archive/llm-draft-spec.md
  m<N>-<name>.md       # written per milestone
experiments/           # appears at M4
  <name>/
    run.py
    config.yaml
    results.md
```

## Milestones

Acceptance criteria are derived from the draft spec where applicable but
adjusted for our 2x2-first sequencing. The draft's "10M transitions/sec
on 3x3" target lives at **M8** (the 3x3-enablement milestone), not M2 or
M4 — both of those are 2x2-only by design.

### M0 — Repo skeleton, `CubeSpec`, notation hub scaffolding

- Package skeleton under `src/rubik/` with `cube/`, `oracle/`,
  `notation/`, `viz/`, `model/`, `training/`, `search/` directories
  (most are empty `__init__.py` placeholders at this stage).
- `CubeSpec` dataclass defined: `size`, `n_stickers`, `faces`,
  `n_moves`, `n_colors`, plus the slot-permutation table type.
  2x2 spec instantiated as `CUBE_2X2`; 3x3 left as a `# TODO M8`.
- Notation hub stubs: `moves.py` (index ↔ string ↔ tuple) and
  `state.py` (tensor ↔ unfolded face dict) with TDD'd round-trip tests.
- `tests/` runs green (even if it's just sanity tests).
- `pyproject.toml`, `ruff` config, lint + format clean.

**Acceptance:** `uv run pytest` green; `uv run ruff check` clean;
`CubeSpec(CUBE_2X2)` round-trips through every notation converter.

### M1 — Slow cubie oracle (2x2)

- `src/rubik/oracle/cubie.py`: corner state as
  `[position_idx, orientation]` per corner slot. Moves applied as
  physical rotations of the affected corners (slots and twists explicit
  in code). Generic enough to extend to 3x3 by adding edges in M8.
- Tests: identities (`M⁴ = I`, `(R U R' U')⁶ = I`, `Sune⁶ = I`),
  inverse relation `M' = M³`, color-multiset preservation when
  rendered into sticker form.

**Acceptance:** all 2x2 corner identities hold; oracle handles a 1000-step
random walk without divergence (no NaN/invalid states).

### M2 — Fast tensor cube (2x2), correctness only

- `src/rubik/cube/env.py`: batched `apply_moves`, `apply_move_sequence`,
  `is_solved`, `random_scrambles`, `valid_next_moves_mask`.
- Move tables derived from oracle (run oracle once, snapshot the
  permutations, embed as constants — *not* a runtime dependency on the
  oracle).
- Equivalence tests: random corpus of 10k move sequences (depth 1..30),
  oracle output matches tensor output for every sequence.
- Identities replicated against tensor implementation.
- **No throughput target at this milestone** — that lives in M4.

**Acceptance:** all 2x2 identities; tensor ≡ oracle on 10k random
sequences; `Cube.SOLVED` solves under any prefix that's the inverse of a
random scramble.

### M3 — Visualization stack — first human-verifiable checkpoint

- `src/rubik/viz/ascii.py`: unfolded-cross ASCII rendering.
- `src/rubik/viz/svg.py`: SVG render components — single state, sequence
  grid, scramble + solution side-by-side.
- All renderers consume **notations**, not raw tensors. Adding a new
  representation only requires writing a converter.
- Snapshot tests via `pytest-snapshot`.
- A static HTML preview file (`assets/cube_preview.html` or similar)
  rendering solved + a few canonical scrambles, opened via `open` for
  visual sanity-check. The "favorite pattern" applied here.

**Acceptance:** snapshot tests green; the user can `open` an HTML preview
showing solved, single-move, and depth-20 scramble states for 2x2 and
visually confirm correctness.

### M4 — Perf-1: MPS measurement + batch sensitivity (2x2)

- Establish the **MPS measurement methodology** for this machine —
  written up as `experiments/mps-methodology/results.md`. Covers
  `macmon` / `mactop` for power + GPU utilization, `torch.profiler`
  trace export, `hyperfine` for end-to-end timing, the gotchas (e.g.,
  warmup, MPS sync, when CPU round-trips are masking truth).
- Batch-size sensitivity sweep on `apply_moves` and `random_scrambles`.
- First throughput targets — ported from the draft (>10M transitions/sec
  on 3x3 was the original target; on 2x2 we expect higher because each
  step is cheaper, but we set the actual number empirically here).
- Establishes the **experiment-loop pattern**: a perf-tuner subagent
  iterates hypotheses, scripts under `experiments/`, results in
  `results.md`.

**Acceptance:** measurement methodology doc exists and is reproducible
from a clean shell; throughput numbers logged with confidence intervals;
no CPU round-trips in the hot path (verified via profiler trace).

### M5 — Scramble pipeline + DAVI training (2x2)

- `training/scrambles.py`: backward random-scramble generator with
  non-trivial-move pruning. Returns `(states, depths, last_faces)`.
- `model/network.py`: MLP value network parameterized on `CubeSpec`
  (input dim derives from sticker count and color count). Body shape
  (`body_widths=(h1, h2)`, `n_residual_blocks=n`) is a required kwarg —
  no committed default. The right values for 2x2 are an open empirical
  question, picked by tier 1+ experimentation. Same class drops in for
  3x3 at M8.
- `training/davi.py`: DAVI training loop — target is
  `min_a (1 + V_target(child))` with terminal-child clamp; periodic
  `V_target ← V_θ` sync.
- **2x2 only at this milestone.** State space is small enough that BFS
  gives optimal `V*` for every reachable state. Score `V_θ` against `V*`
  per-state — strong signal beyond just loss curves. 3x3 happens at M8.

**Acceptance (2x2):** loss decreases monotonically over 100k steps; mean
absolute error vs BFS-optimal `V*` < 1.0 across all reachable states;
greedy solve rate ≥ 99% on depth ≤ 14 (2x2 QTM diameter is 14, verified
empirically by the M5 V\* enumerator). Note: "11" is the 2x2 *HTM* God's
Number — the often-quoted figure — but our move set is QTM-only, so the
relevant diameter for this acceptance gate is 14.

### M6 — Beam search (2x2)

- `search/beam.py`: batched beam search using `V` as scorer. Within-beam
  dedup via state packing. Parametric on `CubeSpec` from day one — no
  2x2-specific branches.
- Verify beam search returns BFS-optimal solution lengths for a sample
  of 1000 random scrambles — 2x2 gives us this ground truth. 3x3
  validation happens at M8.

**Acceptance (2x2):** 100% solve rate on 1000 depth ≤ 14 scrambles at
beam_width=256 (full 2x2 QTM coverage); mean solution length within 1
move of BFS-optimal.

### M7 — Perf-2 / hyperparam experiment loop (2x2 training)

- Reuse the M4 experiment-loop infrastructure for hyperparam sweeps on
  2x2 training: batch size, LR, target update frequency, network width.
- Parallel sweeps if MPS has headroom (one process per config; the
  perf-tuner subagent dispatches and aggregates).
- **Why 2x2 at this milestone, not 3x3.** Iterating on training
  hyperparams against 2x2's BFS-optimal `V*` gives faster, sharper
  signal than 3x3's "loss-curve only" view. Patterns surfaced here
  inform the 3x3 retrain at M8.

**Acceptance:** training reaches M5 acceptance numbers in ≤ 50% of the
baseline wall-clock from M5.

### M8 — 3x3 enablement (CubeSpec swap + rerun)

The 2x2 pipeline shipped end-to-end through M5/M6/M7. M8 is where the
`CubeSpec` abstraction earns its keep — adding 3x3 should be **adding a
spec, not adding code paths**.

- Add `CUBE_3X3` spec.
- Extend cubie oracle (M1) with edge cubies (position + orientation).
- Snapshot 3x3 move-perm from the extended oracle into `env.py`
  alongside `MOVE_PERM_2X2`; tensor env picks up the new spec via
  `_MOVE_PERM[spec.name]` dispatch (already in place from M2).
- Re-run M2 / M3 / M4 / M5 / M6 verification matrices on 3x3:
  - All standard 3x3 identities (`(R U R' U')⁶`, `Sune⁶`, T-perm² etc.)
  - 10k random-sequence equivalence vs. oracle
  - HTML preview shows correct 3x3 unfolded cross
  - Throughput re-measured (the draft's "10M transitions/sec on 3x3"
    target is judged here — not at M2 or M4)
  - DAVI retrain on 3x3
  - Beam search on 3x3 against trained `V_θ`
- Optional: rerun M7 hyperparam sweep to retune for 3x3 if M5's 2x2
  recipe transfers poorly.

**Acceptance:**
- Every M2–M4 acceptance criterion repeats on 3x3 without any
  2x2-specific code path.
- DAVI loss decreases on 3x3; greedy solve rate ≥ 95% on depth ≤ 15
  (matches the draft's target).
- Beam search: 100% solve rate on 1000 depth-20 scrambles at
  beam_width=4096; mean solution length ≤ 24 moves; per-scramble solve
  time < 10s wall on M4 MPS.

### M9 — Stretch: 3D / web frontend, solution-trace analysis

- 3D web viewer for solves (deferred from M3). Likely a small static-site
  frontend that reads a JSON solve trace.
- Analysis: extract 3- and 5-move subsequences from 1000 random solves;
  frequency analysis vs. uniform baseline; check for `aba⁻¹` conjugate
  patterns; compare to CFOP step boundaries ("does it ever pass through
  a white-cross-solved state?"). Writeup in `reports/`.

## Cross-cutting concerns

### Notation hub as single source of truth

Every representation conversion lives in `src/rubik/notation/`. Anything
that needs a string form, a tensor form, an SVG-friendly form, or a
cubie struct goes through the hub. Tests assert round-trip equivalence
between every pair of representations exhaustively at M0 and again on
each new representation added. Renderers (M3, M9) take notations as
input — never raw tensors — so adding a representation never touches
the renderers.

### Experiment-loop pattern

`experiments/<name>/` directories are first-class. Each has:
- `run.py` — entry point, takes a config path
- `config.yaml` — hyperparams / sweep axes
- `runs/` — outputs (gitignored)
- `results.md` — running log of hypotheses, runs, and conclusions

Started in M4, reused in M8. The pattern matters more than any single
experiment's contents.

### MPS measurement methodology

Apple Silicon GPU utilization is not as cleanly observable as CUDA. The
M4 deliverable codifies what we use (`macmon` for power + GPU%,
`torch.profiler` traces, `hyperfine` for end-to-end with warmup, manual
MPS-sync `torch.mps.synchronize()` discipline) and the gotchas (warmup,
when CPU round-trips silently dominate, when MPS-side allocator caching
inflates batch-2 runs vs. cold). Living doc — every subsequent perf
investigation that uncovers a new gotcha appends to it.

### 2x2 ↔ 3x3 single source of truth

Anywhere we'd be tempted to write `if cube_size == 2: ... else: ...` is
a sign we need to push the variation into `CubeSpec` instead. The whole
M8 milestone is essentially "did we succeed?" — adding 3x3 should be
adding a spec, not adding code paths. M2's `_MOVE_PERM[spec.name]` and
M3's `spec.size`-parametric renderers are the working examples; M5–M7
must keep that discipline so M8 stays cheap.

## References

Primary papers:

- Agostinelli et al., **DeepCubeA** (2019).
  https://www.nature.com/articles/s42256-019-0070-z
- McAleer et al., **Autodidactic Iteration / ADI** (2018).
  https://arxiv.org/abs/1805.07470
- Takano, **EfficientCube** (2023). https://arxiv.org/abs/2106.03157

Reference implementations:

- DeepCubeA: https://github.com/forestagostinelli/DeepCubeA — reference
  for cube-group conventions and ADI loop shape. **Hyperparameter values
  are deliberately not borrowed** — see CLAUDE.md "Earn every
  hyperparameter."
- EfficientCube: https://github.com/kyo-takano/efficientcube — beam
  search with policy. Same caveat.

Background:

- Kociemba, https://kociemba.org/cube.htm — cube group theory and move
  pruning conventions.
