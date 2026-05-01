# M1 — Slow cubie oracle (2x2)

## Context

M1 builds the **slow, hand-rolled cubie oracle** for the 2x2 cube — one of
the two correctness witnesses defined in `SPEC.md` (decision: two-witness
correctness, oracle vs tensor cube). The oracle's job is to be
*unmistakably correct*: human-readable, traceable, and obviously right by
inspection. It will become the ground truth that M2's fast tensor cube is
verified against on a 10k random-sequence equivalence sweep.

It also pins the **within-face geometric sticker ordering** that M0
deliberately deferred. Every downstream consumer (M2 tensor cube, M3
renderer, M5 3x3 extension) inherits this convention from the oracle's
sticker rendering function.

The oracle is *not* on the perf path. M2 is correctness-only too. M4 is
the first perf milestone. So everything in M1 is optimized for clarity:
frozen dataclass + tuples, hand-written CW move tables, eager validation
on import, and a single-file test suite that exercises the algebraic
identities.

## Acceptance (from `SPEC.md`)

> `src/rubik/oracle/cubie.py`: corner state as `[position_idx, orientation]`
> per corner slot. Moves applied as physical rotations of the affected
> corners (slots and twists explicit in code). Generic enough to extend
> to 3x3 by adding edges in M5.
>
> Tests: identities (`M⁴ = I`, `(R U R' U')⁶ = I`, `Sune⁶ = I`), inverse
> relation `M' = M³`, color-multiset preservation when rendered into
> sticker form.
>
> **Acceptance:** all 2x2 corner identities hold; oracle handles a
> 1000-step random walk without divergence (no NaN/invalid states).

## Design

### Cubie state representation

Frozen dataclass with two tuples — value semantics, immutable, REPL-friendly:

```python
@dataclass(frozen=True)
class CornerState:
    positions: tuple[int, ...]      # length 8: slot -> cubie identity at that slot
    orientations: tuple[int, ...]   # length 8: slot -> twist of cubie at that slot (0,1,2)
```

Each `apply_move` returns a new `CornerState`. Equality and hash are free
— the random-walk test asserts `state == solved` after applying inverse
sequences via direct `==`. Mirrors the `frozen=True` precedent from
`CubeSpec` (M0).

3x3 extension (M5): add `edges` and `edge_orientations` as optional fields,
or wrap into a `CubeState` superset. Either way, M1 corner code is
unchanged.

### Slot numbering — bit-packed `(U/D, L/R, F/B)`

Slot index 0..7 packs three axis bits: bit 2 = U/D (0=U, 1=D), bit 1 = L/R
(0=L, 1=R), bit 0 = F/B (0=F, 1=B).

| slot | bits | corner |
|------|------|--------|
| 0    | 000  | ULF    |
| 1    | 001  | ULB    |
| 2    | 010  | URF    |
| 3    | 011  | URB    |
| 4    | 100  | DLF    |
| 5    | 101  | DLB    |
| 6    | 110  | DRF    |
| 7    | 111  | DRB    |

Why bit-packed: U/D moves operate on `slot < 4` vs `slot >= 4` (one mask).
L/R, F/B masks fall out the same way. Pays off when hand-writing move
tables and verifying invariants. Internal-only — never exposed in user
API. Singmaster ordering is more familiar from cube-theory literature; we
swap if the user prefers it (purely cosmetic).

A corner's **identity** is the slot it started in. Solved state has
`positions = (0, 1, 2, 3, 4, 5, 6, 7)`.

### Orientation convention — U/D-axis reference

- 0 = U or D facet faces the U/D axis
- 1 = corner twisted 120° CW viewed from outside, along the U/D axis
- 2 = corner twisted 120° CCW

Move twist deltas:
- **U, D moves:** all corners on the moved face get +0
- **F, B, L, R moves:** the 4 affected corners pick up alternating
  +1/+2 (mod 3). Specific assignment follows DeepCubeA / standard
  speedcubing convention. The Sune⁶ = I and (R U R' U')⁶ = I tests will
  fail loudly if any delta is wrong — no other validation needed.

Corner orientation parity invariant: `sum(orientations) ≡ 0 (mod 3)` always.
Asserted in the random-walk test.

### Within-face sticker ordering (THE M1 DECISION)

For each face, view the face from **outside** the cube with the per-face
"up" axis defined below. Sticker indices 0,1,2,3 are top-left, top-right,
bottom-left, bottom-right (row-major reading order).

| face | "up" axis | sticker 0 (TL) | sticker 1 (TR) | sticker 2 (BL) | sticker 3 (BR) |
|------|-----------|----------------|----------------|----------------|----------------|
| U    | toward B  | ULB (1)        | URB (3)        | ULF (0)        | URF (2)        |
| L    | toward U  | ULB (1)        | ULF (0)        | DLB (5)        | DLF (4)        |
| F    | toward U  | ULF (0)        | URF (2)        | DLF (4)        | DRF (6)        |
| R    | toward U  | URF (2)        | URB (3)        | DRF (6)        | DRB (7)        |
| B    | toward U  | URB (3)        | ULB (1)        | DRB (7)        | DLB (5)        |
| D    | toward F  | DLF (4)        | DRF (6)        | DLB (5)        | DRB (7)        |

Rationale: U, L, F, R, B, D arrayed left-to-right in standard "unfolded
cross" form all read consistently with this convention. M3's renderer
consumes the same layout without per-face rotation logic.

For each corner, three stickers exist on three different faces. The
table above implicitly defines which sticker-position-within-corner
(0,1,2 = U/D, F/B, L/R order) lives on which face. `cubie_to_tensor`
uses this mapping plus orientation to compute each face sticker's color.

### Move tables — hand-write 6 CW, derive 6 CCW

Build CW move tables for U, L, F, R, B, D as a dict literal:

```python
# (slot_before, slot_after, orientation_delta) for the 4 affected corners
_CW_MOVES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "U": ((0, 2, 0), (2, 3, 0), (3, 1, 0), (1, 0, 0)),  # ULF→URF→URB→ULB→ULF
    "F": ((0, 2, 1), (2, 6, 2), (6, 4, 1), (4, 0, 2)),  # alternating +1/+2
    ...
}
```

CCW moves are derived as `apply CW thrice`. The `M' = M³` acceptance test
becomes a *real* test (vs derivation tautology) only if CW and CCW are
genuinely independent — but here we deliberately couple them at
construction so the test reduces to "CW table is internally consistent",
which is fine because CW correctness is also covered by M⁴ = I,
(R U R' U')⁶ = I, and Sune⁶ = I.

**Module-import validation** (cheap, catches typos):
- Each CW move's 4 `slot_before` values are distinct and form a permutation
  with the 4 `slot_after` values (i.e. every slot_after is also a
  slot_before — the move cycles 4 corners among themselves).
- Each CW move's orientation deltas sum to 0 mod 3 (parity invariant).

### Sticker rendering — `cubie_to_tensor(state, spec) -> torch.Tensor[24]`

Lives in `oracle/cubie.py`. The cubie struct is the oracle's *internal*
representation; it is not promoted to a user-facing notation. M2's tensor
cube will call `cubie_to_tensor` to derive its sticker-permutation tables
(apply each CW move to solved, render, diff against solved).

`tensor_to_cubie` is **not** implemented — sticker→cubie inversion isn't
needed for M1 acceptance, and the inversion is non-trivial. Defer until a
concrete need surfaces.

## Files

**Create:**
- `src/rubik/oracle/cubie.py` — `CornerState` dataclass, `SOLVED`,
  `_CW_MOVES` dict, `apply_move(state, move_idx) -> CornerState`,
  `cubie_to_tensor(state, spec) -> torch.Tensor`, module-import
  validation block
- `tests/oracle/__init__.py` — empty (mirrors `tests/cube/__init__.py`)
- `tests/oracle/test_cubie.py` — full test suite (see Tests section)

**Modify:**
- `src/rubik/oracle/__init__.py` — re-export `CornerState`, `SOLVED`,
  `apply_move`, `cubie_to_tensor`

**Do not touch:**
- `src/rubik/cube/spec.py` — no corner bookkeeping on `CubeSpec` (M0
  decision: `CubeSpec` stays lean, fields earn their way in)
- `src/rubik/notation/state.py` — cubie struct is internal, not a notation
- `pyproject.toml` — no new deps; `random.seed(0)` is sufficient for the
  random walk, no hypothesis needed

## Tests (`tests/oracle/test_cubie.py`)

Single file, sectioned by blank lines:

1. **`test_solved_state`** — `SOLVED.positions == tuple(range(8))`,
   `SOLVED.orientations == (0,) * 8`.
2. **`test_move_quartic_identity`** — `@pytest.mark.parametrize("m", range(12))`,
   apply move 4 times, assert `state == SOLVED`.
3. **`test_inverse_relation`** — for each face f, applying CW move thrice
   equals applying CCW move once (this is how CCW is *derived*, but
   verifying as an explicit test documents the invariant).
4. **`test_sexy_move_identity`** — apply `R U R' U'` six times (24 moves),
   assert solved.
5. **`test_sune_identity`** — apply `R U R' U R U U R'` six times (48 QTM
   moves), assert solved. (QTM-expansion of the standard HTM Sune
   `R U R' U R U2 R'`, since QTM has no double moves.)
6. **`test_color_multiset_preservation`** — apply a fixed 50-move
   sequence, render to tensor, assert
   `Counter(tensor.tolist()) == {c: 4 for c in range(6)}`.
7. **`test_random_walk_no_divergence`** — `random.seed(0)`, 1000 random
   moves. At every step assert: `set(positions) == set(range(8))`,
   `all(o in {0,1,2} for o in orientations)`, `sum(orientations) % 3 == 0`.
8. **`test_round_trip_via_inverse_sequence`** — generate a 30-move
   scramble (seeded), apply inverse sequence (reverse + flip each
   move's direction bit), assert `state == SOLVED`.
9. **`test_solved_renders_to_spec_solved_state`** — `cubie_to_tensor(SOLVED,
   CUBE_2X2)` equals `CUBE_2X2.solved_state`. Anchors the within-face
   ordering against M0's `solved_state` property (which is just
   `[0]*4 + [1]*4 + ... + [5]*4`, so this always passes — but documents
   that the rendering convention is consistent with CubeSpec).
10. **`test_after_one_move_color_distribution`** — apply each of the 12
    moves to `SOLVED`, assert exactly 8 stickers changed face (not 0,
    not 24) — sanity check that single moves move things.

Tests use `@pytest.mark.parametrize` over moves and faces where
applicable. No hypothesis dep.

## Branch & commits

Branch: `m1-cubie-oracle` off `main`. Atomic commits aligned to the
file groups:

1. `M1: oracle skeleton — CornerState + SOLVED`
2. `M1: hand-written CW move tables + apply_move`
3. `M1: cubie_to_tensor sticker rendering`
4. `M1: oracle test suite`
5. `M1: re-export from oracle/__init__.py`

Plan and LOG block updates are amended into the relevant commits as
work lands (per cc-process).

## Verification

Run from repo root:

```sh
uv run pytest tests/oracle -v
uv run pytest                # full suite still green
uv run ruff check
uv run ruff format --check
```

Acceptance gate: all of the above pass. The 1000-step random walk test
runs in well under a second; full suite stays under a few seconds.

## Out of scope

- `tensor_to_cubie` (sticker → cubie inversion) — defer until a need
  surfaces.
- Snapshotting the move table as a CubeSpec field — M2 might do this,
  but M1 holds it inside the oracle module.
- Edge cubies — M5 (3x3 scaling milestone).
- Performance — oracle runs in pure Python; perf is M4's concern.
- Visualization — M3.
