# M2 — Fast tensor cube (2x2), correctness only

## Context

M2 builds the **fast tensor cube** for the 2x2 — the production code path
that env / training / search will all consume. Per `SPEC.md` decision 6
("make it work, then make it fast"), M2 is **correctness only**: no
throughput target, no MPS tuning, no batch-size sweep. That work lives in
M4. M2's job is to land an obviously-correct vectorized implementation
that the M1 oracle proves equivalent to on a 10k-sequence corpus, and to
fill out the env API surface that M6/M7 need (batched `apply_moves`,
`random_scrambles`, `is_solved`, etc.).

`SPEC.md` decision 1 ("2x2 first, parameterized via `CubeSpec`") and the
"no 2x2/3x3 forks" rule mean every function consumes a `CubeSpec` and
indexes into `spec`-derived constants. M5 will re-run the M2 acceptance
gate by swapping in `CUBE_3X3` — same code path.

The tensor cube **does not import the oracle at runtime** (`SPEC.md`
line 230). Move tables are snapshotted as static constants. A single
verification test imports both modules and asserts the snapshot still
matches what the oracle produces — drift catcher in CI without coupling
production code to the slow path.

## Acceptance (from `SPEC.md` lines 237–239)

> **Acceptance:** all 2x2 identities; tensor ≡ oracle on 10k random
> sequences; `Cube.SOLVED` solves under any prefix that's the inverse of a
> random scramble.

Plus the deliverables from `SPEC.md` lines 227–235:

> - `src/rubik/cube/env.py`: batched `apply_moves`, `apply_move_sequence`,
>   `is_solved`, `random_scrambles`, `valid_next_moves_mask`.
> - Move tables derived from oracle (run oracle once, snapshot the
>   permutations, embed as constants — *not* a runtime dependency on the
>   oracle).
> - Equivalence tests: random corpus of 10k move sequences (depth 1..30),
>   oracle output matches tensor output for every sequence.
> - Identities replicated against tensor implementation.
> - **No throughput target at this milestone** — that lives in M4.

## Design

### State representation

A "state" is a flat sticker tensor of shape `(spec.n_stickers,)` with
dtype `int8`, exactly matching `CubeSpec.solved_state`. A "batched state"
has shape `(B, spec.n_stickers)` with the same dtype. No new struct, no
class wrapper — keep parity with M0/M1's "tensors are the wire format"
convention. `CUBE_2X2.solved_state` is the canonical solved state.

Device-agnostic: every function preserves the device of its input tensor.
M2 doesn't pin MPS vs CPU — that's M4. Tests run on CPU.

### Move tables — sticker permutations, snapshotted from oracle

A move applied to state `s` produces `s_new[i] = s[perm[move_idx, i]]`.
We need a `(12, 24)` int permutation table:

```python
MOVE_PERM_2X2: torch.Tensor  # shape (12, 24), dtype int64, device cpu
```

`int64` because PyTorch's advanced indexing wants `long`. (Storing as
`int8` and casting on each call would be a perf hit and the table is
tiny — 12×24×8B = 2.3 KB.)

**Generation method.** The 12 permutations are derived once by:

1. Render `cubie_to_tensor(SOLVED, CUBE_2X2)` → solved sticker tensor `t0`.
2. For each `move_idx` in `0..11`: render the resulting state via
   `cubie_to_tensor(apply_move(SOLVED, move_idx), CUBE_2X2)` → `t1`.
3. The permutation `perm` satisfies `t1[i] = t0[perm[i]]`. Because
   `t0[k]` is just `k // 4` and the move-result tensor preserves the
   color multiset, the permutation must be derived by tracking sticker
   *positions*, not colors. Method: build a "fingerprint" version of
   `cubie_to_tensor` where each cubie carries its sticker-position
   index per facet (24 distinct values), apply the move, read off the
   permutation from the after-move rendering.

Generator script: `visuals/scripts/generate_move_perm_2x2.py` (one-shot,
prints the tuple-of-tuples; the printed output gets pasted into `env.py`
as a literal). Both the script and its rendered constants are checked in,
following the `visuals/` precedent of "artifact + generator both live in
the repo." Despite the name, the script lives under `visuals/` because
it's a dev-time generator producing an out-of-tree artifact (the snapshot
table); it shouldn't pollute `src/rubik/`.

A test (`test_move_perm_matches_oracle`) reproduces step 1–3 against the
embedded constant and asserts equality. If anyone changes the oracle's
sticker-geometry decisions later, the test will fail loudly and the
generator gets re-run.

### API surface

All functions live in `src/rubik/cube/env.py`. Module-level functions
(no `Cube` class) — matches `oracle/cubie.py` and `notation/state.py`
style.

```python
# Batched core: state shape (B, S) or (S,) — both work via broadcasting.
def apply_moves(states: torch.Tensor, move_idxs: torch.Tensor,
                spec: CubeSpec) -> torch.Tensor: ...

# Sequential application of a fixed move sequence to a single state.
def apply_move_sequence(state: torch.Tensor, moves: Sequence[int],
                        spec: CubeSpec) -> torch.Tensor: ...

# Batched solved check.
def is_solved(states: torch.Tensor, spec: CubeSpec) -> torch.Tensor: ...

# Generate B random scrambles of given depth from solved.
def random_scrambles(spec: CubeSpec, batch_size: int, depth: int,
                     generator: torch.Generator | None = None,
                     prune_same_face: bool = True
                     ) -> tuple[torch.Tensor, torch.Tensor]: ...
    # returns (states (B, S), move_seqs (B, depth))

# Per-position legal-move mask. Excludes same-face repeats given prev move.
def valid_next_moves_mask(prev_move_idxs: torch.Tensor | None,
                          spec: CubeSpec) -> torch.Tensor: ...
    # if prev is None: all True (n_moves,)
    # else: mask[..., m] = (m >> 1) != (prev >> 1)
```

`apply_moves` is the workhorse:

```python
def apply_moves(states, move_idxs, spec):
    perm = _MOVE_PERM[spec.name]            # (n_moves, n_stickers)
    perm_for_each = perm[move_idxs].to(states.device)  # (..., n_stickers)
    return torch.gather(states, -1, perm_for_each)
```

The `_MOVE_PERM` dict is indexed by `spec.name` so the same code path
serves 2x2 today and 3x3 in M5. Only `MOVE_PERM_2X2` exists at M2; the
3x3 entry gets added in M5 via the same generator script (renamed
`generate_move_perm_3x3.py` per the `_2x2`/`_3x3` naming convention).

`random_scrambles` uses `valid_next_moves_mask` to optionally exclude
same-face repeats — toggled by a `prune_same_face` flag (default `True`,
since DAVI / scramble generation usually wants this). M2 just needs the
function to exist and produce valid scrambles; the prune behavior is
exercised by an explicit test.

### Why `valid_next_moves_mask`

For QTM with no double moves, all 12 moves are always legal in the strict
sense. The mask exists to support same-face pruning, which both scramble
generation and beam search want. Signature takes an optional
`prev_move_idxs` because the "validity" depends on what came before, not
on the current state. When `prev_move_idxs is None`, mask is all-True.

### Re-exports

`src/rubik/cube/__init__.py` re-exports `apply_moves`,
`apply_move_sequence`, `is_solved`, `random_scrambles`,
`valid_next_moves_mask`. `src/rubik/__init__.py` extends `__all__` with
the same names so `from rubik import apply_moves, CUBE_2X2` works.

## Files

**Create:**
- `src/rubik/cube/env.py` — module-level functions above + the embedded
  `MOVE_PERM_2X2` constant + the `_MOVE_PERM` dispatch dict.
- `tests/cube/test_env.py` — equivalence sweep, identities, snapshot
  drift test, API smoke tests (see Tests section).
- `visuals/scripts/generate_move_perm_2x2.py` — one-shot generator.
  Prints the snapshot tuple-of-tuples to stdout; the human pastes into
  `env.py`. Imports the oracle (allowed; it's a dev-time tool, not
  production code).

**Modify:**
- `src/rubik/cube/__init__.py` — re-export the 5 env functions.
- `src/rubik/__init__.py` — extend `__all__`.

**Do not touch:**
- `src/rubik/cube/spec.py` — `CubeSpec` stays lean per M0 decision; no
  `move_table` field. M2 keeps move tables module-local in `env.py`.
- `src/rubik/oracle/cubie.py` — oracle is locked. M2 verifies against it,
  doesn't change it.
- `src/rubik/notation/{moves,state}.py` — already stable from M0.
- `pyproject.toml` — no new deps. `torch` already pulled at bootstrap.

## Tests (`tests/cube/test_env.py`)

Single test file, sectioned by blank lines. Mirrors M1's identity tests
applied to the tensor path.

1. **`test_solved_is_solved`** — `is_solved(CUBE_2X2.solved_state, CUBE_2X2)`
   is `True`; arbitrary scrambled state is `False`.
2. **`test_move_perm_matches_oracle`** — re-derives the 12 permutations
   via the oracle and asserts equality with the embedded
   `MOVE_PERM_2X2` constant. Drift catcher.
3. **`test_apply_moves_quartic_identity`** — parametrized over all 12
   moves: 4 applications return to solved.
4. **`test_apply_moves_inverse_relation`** — for each face, CW-applied
   thrice equals CCW-applied once.
5. **`test_apply_moves_sexy_identity`** — `(R U R' U')⁶ = I`.
6. **`test_apply_moves_sune_identity`** — `(R U R' U R U U R')⁶ = I`
   (QTM expansion of Sune).
7. **`test_cw_direction_tensor`** — replicates M1's `EXPECTED_CW_POSITIONS`
   table, but on the rendered sticker tensor: applies each face's CW
   move to solved and asserts the result equals what the oracle renders
   for the same state. Catches R↔R' inversions (the M1-follow-up bug
   class).
8. **`test_oracle_equivalence_10k`** — the M2 acceptance sweep.
   Seeded RNG (`torch.Generator().manual_seed(0)`); for `i in range(10_000)`:
   draw a random depth `d ∈ [1, 30]`, draw `d` random move indices, apply
   to solved via both oracle and tensor cube, assert tensor result equals
   `cubie_to_tensor(oracle_state, CUBE_2X2)`. Should run in well under a
   minute on CPU; if it doesn't, lower the count to 5k and note it.
9. **`test_round_trip_via_inverse_sequence`** — generates a 30-move
   scramble (seeded), builds inverse sequence (`m ^ 1` reversed), asserts
   `is_solved(apply_move_sequence(scramble_state, inverse_seq, ...))`.
   Implements the SPEC's "`Cube.SOLVED` solves under any prefix that's the
   inverse of a random scramble" acceptance bullet.
10. **`test_color_multiset_preservation_tensor`** — apply a fixed
    50-move sequence; assert the resulting state has 4 of each color.
11. **`test_apply_moves_batched`** — apply different moves to a batch of
    states in one call; spot-check by comparing to per-state results.
12. **`test_random_scrambles_shape_and_legality`** —
    `random_scrambles(CUBE_2X2, batch_size=64, depth=20, generator=…)`
    returns shapes `(64, 24)` and `(64, 20)`; states are valid (right
    color multiset); same-face prune is honored when `prune_same_face=True`.
13. **`test_valid_next_moves_mask`** — `prev=None` → all True; `prev=R`
    (idx 6) → `mask[..., 6]` and `mask[..., 7]` are False, others True.
14. **`test_device_preservation`** — call `apply_moves` on a CPU tensor;
    result is on CPU. (No MPS test at M2 — that's M4.)
15. **`test_dtype_preservation`** — input `int8` produces `int8`.

Tests use `pytest.mark.parametrize` over moves and faces where it
collapses repetition. RNG is seeded everywhere (`manual_seed(0)`) for
determinism.

## Branch & commits

Branch: `m2-tensor-cube` off `main`. Atomic commits roughly aligned
to the file groups (cc-process: small, atomic, with the LOG block
amended as work lands):

1. `M2: open block, plan, ROADMAP toggle`
2. `M2: generator script for MOVE_PERM_2X2 snapshot`
3. `M2: env.py — MOVE_PERM_2X2 + apply_moves + is_solved`
4. `M2: env.py — apply_move_sequence + random_scrambles + valid_next_moves_mask`
5. `M2: tests/cube/test_env.py — identities + drift + 10k equivalence`
6. `M2: re-exports from cube/__init__.py + rubik/__init__.py`

## Verification

Run from repo root:

```sh
uv run pytest tests/cube -v
uv run pytest                # full suite still green (M0+M1+M2 ≈ 105+ tests)
uv run ruff check
uv run ruff format --check
```

Acceptance gate: all of the above pass, including the 10k oracle
equivalence sweep. Sweep budget: under 60s on CPU. If it busts that, drop
to 5k and note the deviation in the LOG outcome.

Sanity check after the work lands:

```python
from rubik import CUBE_2X2, apply_moves, is_solved
import torch
state = CUBE_2X2.solved_state
state = apply_moves(state, torch.tensor(6), CUBE_2X2)   # R
assert not is_solved(state, CUBE_2X2)
```

## Out of scope

- **Throughput / MPS tuning** — M4. M2 explicitly has no perf bar.
- **3x3 entries in `_MOVE_PERM`** — M5, when `CUBE_3X3` lands. The
  dispatch dict is keyed by `spec.name` so adding `MOVE_PERM_3X3` is
  drop-in.
- **`Cube` class wrapper** — module-level functions are sufficient and
  match the rest of the package. Re-evaluate if downstream phases want
  method-style syntax.
- **Heuristic / value-network calls** — M6+.
- **Same-face pruning during search** — M7. M2 just provides the mask
  function so search can use it.
- **Non-QTM moves (`R2`, slice moves)** — out of scope for the whole
  project per `SPEC.md` decision 3.
