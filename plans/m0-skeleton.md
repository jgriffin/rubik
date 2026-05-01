# M0 — Repo skeleton, `CubeSpec`, notation hub scaffolding

## Context

The bootstrap commit landed yesterday (9522ce0): `pyproject.toml`, deps, `SPEC.md`, `ROADMAP.md`, empty `src/rubik/__init__.py`. Nothing executable yet, no tests, no ruff config, no `tests/` directory.

M0 lays the foundation that every later milestone consumes: a parameterized `CubeSpec` so 2x2 and 3x3 share one code path (Decision 1), and a notation hub (Decision 4) so renderers / tests / oracle / training all transit through explicit converters instead of ad-hoc transforms. Acceptance from `SPEC.md`:

> `uv run pytest` green; `uv run ruff check` clean; `CubeSpec(CUBE_2X2)` round-trips through every notation converter.

We are **not** building any cube semantics yet — no move tables, no oracle, no `apply_moves`. Move tables come from the oracle (M1) and are snapshotted into `cube/env.py` at M2. M0 is pure scaffolding + the abstractions that downstream code will lean on.

## Design decisions

### Face order: `("U", "L", "F", "R", "B", "D")`

Inherited from `plans/archive/llm-draft-spec.md` (lines 50–55). Used both as the sticker-block order in the flat state tensor and as the face-indexing order for moves. Sticking with the archive's convention keeps the encoding doc reusable later.

### Sticker layout for `CUBE_2X2`: 24 stickers, 4 per face, face-block order matches `faces`

- Indices 0–3: U, 4–7: L, 8–11: F, 12–15: R, 16–19: B, 20–23: D.
- Solved state: stickers `[0..3]` are color 0, `[4..7]` are color 1, …, `[20..23]` are color 5. So `solved == (state == SOLVED).all(dim=-1)` works (matches archive convention).
- **Within-face geometric ordering (which sticker is top-left vs top-right) is deliberately NOT pinned at M0.** It only matters when (a) the cubie oracle (M1) maps physical rotations to slot permutations, and (b) the renderer (M3) lays out a 2D grid. M0 just commits to face-block order and stickers-per-face being contiguous; M1 will pin the within-face mapping when it generates the move tables.

### Move index mapping: `move_idx = face_idx * 2 + direction`

- `direction = 0` → CW (no prime), `direction = 1` → CCW (prime).
- Indices: `0:U, 1:U', 2:L, 3:L', 4:F, 5:F', 6:R, 7:R', 8:B, 9:B', 10:D, 11:D'`.
- Same-face pruning at M2/M6/M7 becomes `move >> 1`, which is cheap on tensors.
- String form: `"U"`, `"U'"`, etc. Tuple form: `(face_idx, direction)`.

### `CubeSpec` fields (M0 scope)

```python
@dataclass(frozen=True)
class CubeSpec:
    name: str                       # "2x2", "3x3"
    size: int                       # 2 or 3
    faces: tuple[str, ...]          # ("U", "L", "F", "R", "B", "D")
    stickers_per_face: int          # 4 (2x2), 8 (3x3 — centers dropped)
    n_colors: int                   # 6
    n_moves: int                    # 12 (QTM)

    @property
    def n_stickers(self) -> int: return len(self.faces) * self.stickers_per_face

    @property
    def n_faces(self) -> int: return len(self.faces)

    @property
    def solved_state(self) -> torch.Tensor:
        # [n_stickers] int8; stickers_per_face copies of each color in face order.
```

`frozen=True` so a `CubeSpec` is hashable and can sit as a module-level constant.

**Deliberately omitted from CubeSpec at M0:** `move_table` (filled in M2 from oracle), `face_to_color_map` (the canonical "color i ↔ face i" identity is implicit in `solved_state`), and any 3x3-specific bookkeeping like `n_corners` / `n_edges` (added in M5 if oracle architecture demands it). YAGNI guard — fields earn their way in by being needed.

`CUBE_2X2` instantiated at module level. `CUBE_3X3` left as a `# TODO M5` comment, no stub object.

### Notation hub: two converter modules, both parameterized on `CubeSpec`

`notation/moves.py` is independent of CubeSpec (the 12-move QTM mapping is identical for any cube size as long as it's QTM). `notation/state.py` is parameterized on CubeSpec because face order, sticker count, and stickers-per-face all flow through it.

State dict shape: `dict[str, list[int]]` where each value is a flat list of `stickers_per_face` ints in the same order as the tensor block. Round-trips trivially. The "unfolded face dict" name in `SPEC.md` is a forward-looking term — at M0 it's just a face-keyed flat-list dict; the renderer (M3) is what reshapes per-face lists into 2D grids.

### Ruff config

Add `[tool.ruff]` section with `target-version = "py312"`, `line-length = 88`, lint ruleset including `E`, `F`, `I`, `UP`, `B`, `SIM`. Format defaults. No per-file ignores yet.

## Files

**Created (skeleton — empty `__init__.py` only):**
- `src/rubik/cube/__init__.py`
- `src/rubik/oracle/__init__.py`
- `src/rubik/notation/__init__.py`
- `src/rubik/viz/__init__.py`
- `src/rubik/model/__init__.py`
- `src/rubik/training/__init__.py`
- `src/rubik/search/__init__.py`

**Created (with content):**
- `src/rubik/cube/spec.py` — `CubeSpec` dataclass, `CUBE_2X2` constant, `# TODO M5` for 3x3.
- `src/rubik/notation/moves.py` — `move_to_str`, `str_to_move`, `move_to_tuple`, `tuple_to_move`, plus module-level `MOVE_STRINGS` and `FACE_NAMES` constants. Independent of CubeSpec.
- `src/rubik/notation/state.py` — `state_to_dict(state, spec)`, `dict_to_state(d, spec)`. Pulls face names + chunk size from spec.
- `tests/__init__.py` — empty.
- `tests/cube/__init__.py` — empty.
- `tests/cube/test_spec.py` — sanity: `CUBE_2X2.n_stickers == 24`, `len(CUBE_2X2.faces) == 6`, `CUBE_2X2.solved_state` has the expected per-face color blocks, `frozen=True` enforced.
- `tests/notation/__init__.py` — empty.
- `tests/notation/test_moves.py` — round-trip index→str→index and index→tuple→index for all 12 indices; bad input raises `ValueError`.
- `tests/notation/test_state.py` — round-trip `solved_state → dict → state` exact equality; round-trip on a randomly-permuted state with valid color counts; dict with wrong face names or wrong list lengths raises `ValueError`.

**Modified:**
- `pyproject.toml` — add `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.format]` sections. Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `pythonpath = ["src"]`.
- `src/rubik/__init__.py` — re-export `CubeSpec`, `CUBE_2X2` for ergonomic imports (`from rubik import CUBE_2X2`).

## Verification

Run from a clean shell at repo root:

```bash
uv sync                         # ensure dev deps installed
uv run ruff format --check      # format clean
uv run ruff check               # lint clean
uv run pytest -v                # all tests green
```

Acceptance gate (from `SPEC.md`):
- `pytest` green.
- `ruff check` clean.
- The state-round-trip test exercises `CubeSpec(CUBE_2X2)` going `tensor → dict → tensor` and asserts exact tensor equality on the solved state plus at least one randomly-shuffled state.

## Out of scope (deferred)

- Move tables / sticker permutations — derived from oracle in M1, snapshotted in M2.
- `apply_moves`, `is_solved`, `random_scrambles` — M2.
- Cubie struct in `notation/state.py` — M1 introduces it; the converter slot is reserved by leaving the module name plural-friendly.
- ASCII / SVG renderers — M3.
- `CUBE_3X3` — M5 (just a TODO marker at M0).
