# M3 — Visualization stack (2x2)

## Context

M0–M2 landed the engine half: `CubeSpec`, the cubie oracle, and the fast
tensor cube (`apply_moves`, `is_solved`, `random_scrambles`, ...). Algebraic
identities cover *consistency*, but they don't catch geometry errors that
swap two faces or mirror a sticker layout — exactly what bit us on R-direction
in the M1 follow-up. Visual checking is what surfaces those.

M3 builds the human-verifiable layer **before** any larger test corpus comes
to depend on the engine being correct. Three deliverables per SPEC §M3:

1. ASCII renderer (REPL / CI logs / doctests).
2. SVG render components (single state, sequence grid, side-by-side compare),
   snapshot-tested.
3. A static HTML preview (`visuals/cube_preview_2x2.html`) showing solved,
   single-move, and deep-scramble states — opened via `open` for the
   "favorite pattern" eyeball check.

**Acceptance** (from SPEC §M3): snapshot tests green; user can `open` the
HTML preview and visually confirm solved, single-move, and depth-20 2x2
scramble states render correctly.

## Approach

### Code lives under `src/rubik/viz/`

```
src/rubik/viz/
  __init__.py     # re-export public API
  colors.py       # FACE_COLORS palette, single source of truth
  ascii.py        # render_ascii(face_dict, spec) -> str
  svg.py          # render_svg_state / _sequence / _compare -> str
```

**Renderers consume notations, not tensors** (SPEC §4 + §M3). The core
signatures take `face_dict: dict[str, list[int]]` as produced by
`rubik.notation.state.state_to_dict`. Callers with a raw tensor do
`render_ascii(state_to_dict(state, spec), spec)` — one line, no convenience
wrapper duplicates needed.

### Within-face geometry — pinned by M1

Each face's 4 stickers are stored row-major **TL, TR, BL, BR** when viewed
from outside the cube with the M1 up-axis convention:

- U: up toward B
- L, F, R, B: up toward U
- D: up toward F

This is `oracle/cubie.py:141–154` (`FACE_SLOTS`). The renderer must reshape
each face's 4-sticker block as `[[TL, TR], [BL, BR]]` exactly — no further
rotation. Verified by snapshot equivalence to the existing
`visuals/cubie_2x2_rotations.html` layout.

### Unfolded-cross layout (2x2)

Standard cross, faces in `("U","L","F","R","B","D")` order placed at:

```
              U
        L  F  R  B
              D
```

Each face is a 2-row × 2-col block of stickers. Total grid: 8 columns × 6
rows of stickers. Empty cells are blank (ASCII: spaces; SVG: just no rect).
The grid coords mirror `visuals/scripts/generate_cubie_2x2_rotations.py`'s
`FACE_TOP_LEFT` dict (lines 41–48) so the M3 SVG aligns visually with the
existing M1 rotations preview.

### colors.py — single source of truth

Lift the palette out of `visuals/scripts/generate_cubie_2x2_rotations.py`
(currently the only definition) into `src/rubik/viz/colors.py`:

```python
FACE_COLORS: dict[int, str] = {
    0: "#f5f5f5",  # U  white
    1: "#ff8c00",  # L  orange
    2: "#1aa64a",  # F  green
    3: "#dd2222",  # R  red
    4: "#1144cc",  # B  blue
    5: "#ffdd00",  # D  yellow
}
```

Keyed by face index (matches `spec.faces` ordering). Drive-by: refactor
`generate_cubie_2x2_rotations.py` to import from here, removing the
duplicate. Test asserts `len(FACE_COLORS) == 6` and every value is a 7-char
`#rrggbb` string.

### ASCII renderer (`viz/ascii.py`)

```python
def render_ascii(face_dict: dict[str, list[int]], spec: CubeSpec) -> str
```

Returns the unfolded cross with each cell as a single character — the face
letter (`U`/`L`/`F`/`R`/`B`/`D`) corresponding to that sticker's color
index. Cells separated by a single space within a row; empty cross cells
are 2 spaces (one per sticker position) plus the inter-sticker space. No
trailing newline. Example for solved 2x2:

```
    U U
    U U
L L F F R R B B
L L F F R R B B
    D D
    D D
```

Tests (exact string match, no snapshot needed for ASCII — short and stable):
- `test_render_ascii_solved` — solved 2x2 matches the literal above.
- `test_render_ascii_after_R` — apply R via `cube.env.apply_moves`, render,
  match expected string (hand-derived from `EXPECTED_CW_POSITIONS["R"]` in
  `tests/oracle/test_cubie.py`).
- `test_render_ascii_after_U` — same, for U.
- `test_render_ascii_sticker_count` — count of face letters in the output
  is exactly 24 for a 2x2.

### SVG renderer (`viz/svg.py`)

Three functions, all returning `str` (an SVG fragment, no `<?xml ...?>`
header — embeddable in HTML):

```python
def render_svg_state(face_dict, spec, *, sticker_size=20, gap=1) -> str
def render_svg_sequence(face_dicts, spec, *, labels=None, ...) -> str
def render_svg_compare(top_dicts, bottom_dicts, spec, *, top_label=None,
                       bottom_label=None, ...) -> str
```

- **`render_svg_state`** outputs a single `<svg width=W height=H>` with
  one `<rect>` per sticker, fill from `FACE_COLORS`. Stroke `#0a0a0a`,
  `rx=1.5` to match the M1 preview's CSS aesthetics. Coordinates derived
  from the face grid offsets above.
- **`render_svg_sequence`** lays out a horizontal row of states with
  optional text labels above each (e.g. "R", "U'"). Uses `<g transform="translate(...)">`
  to compose `render_svg_state` outputs.
- **`render_svg_compare`** stacks two sequences vertically with optional
  row labels at left.

**Snapshot tests** (pytest-snapshot, fixture dir `tests/viz/snapshots/`):
- `test_svg_state_solved`
- `test_svg_state_after_R`
- `test_svg_sequence_solved_R_U` — 3-frame sequence
- `test_svg_compare_scramble_vs_solved` — 2-row compare

First run materializes snapshots under version control; subsequent runs
verify byte-equality. Same pattern as DeepCubeA-style viz harnesses.

### Static HTML preview

`visuals/scripts/generate_cube_preview_2x2.py` writes
`visuals/cube_preview_2x2.html`. Sections:

1. **Solved** — single state.
2. **Single-move** — solved → R, U, F as a 3-frame sequence.
3. **Depth-10 canonical** — solved → known sequence (e.g. "R U R' U' R U R' U'"),
   shown as a sequence grid.
4. **Depth-20 scramble** — `random_scrambles(spec, 1, depth=20, seed=42)`,
   side-by-side with its inverse-applied state (must equal solved).

Imports `rubik.viz.svg`, `rubik.viz.colors`, `rubik.cube.env`. Reuses the
dark-theme CSS (`#1a1a1a` bg, `'SF Mono'` labels) from
`generate_cubie_2x2_rotations.py` for visual consistency.

User runs `uv run python visuals/scripts/generate_cube_preview_2x2.py` then
`open visuals/cube_preview_2x2.html`. This is the M3 acceptance check.

## Files

**Add:**
- `src/rubik/viz/colors.py`
- `src/rubik/viz/ascii.py`
- `src/rubik/viz/svg.py`
- `tests/viz/__init__.py`
- `tests/viz/test_colors.py`
- `tests/viz/test_ascii.py`
- `tests/viz/test_svg.py`
- `tests/viz/snapshots/` (created by first snapshot run, then committed)
- `visuals/scripts/generate_cube_preview_2x2.py`
- `visuals/cube_preview_2x2.html`
- `plans/m3-viz.md`

**Modify:**
- `src/rubik/viz/__init__.py` (currently empty) — re-export public API
- `src/rubik/__init__.py` — extend `__all__`
- `visuals/scripts/generate_cubie_2x2_rotations.py` — drive-by, import
  `FACE_COLORS` from `rubik.viz.colors` instead of defining locally
- `ROADMAP.md`, `LOG.md` — milestone close

**Branch:** `m3-viz` (per cc-process branch convention).

## Atomic commits

Per the M2 cadence — small, scannable commits with green tests at each
boundary. Approximate sequence (deviations OK if a fixup separates):

1. **plan + LOG open** — write `plans/m3-viz.md`, open M3 LOG block.
2. **`viz/colors.py` + tests** — palette + invariant tests.
3. **`viz/ascii.py` + tests** — ASCII renderer + 4 tests.
4. **`viz/svg.py` + tests** — SVG renderer (3 functions) + 4 snapshot tests.
   First snapshot run creates `tests/viz/snapshots/`; commit includes the
   generated snapshot files.
5. **`viz/__init__.py` re-exports + `rubik/__init__.py` `__all__`.**
6. **HTML preview generator + artifact** — `generate_cube_preview_2x2.py`
   + checked-in `cube_preview_2x2.html`.
7. **Drive-by colors dedup** — refactor `generate_cubie_2x2_rotations.py`
   to import from `rubik.viz.colors`.
8. **LOG outcome + ROADMAP ✅** — close block.

## Reuse

- `state_to_dict` (`src/rubik/notation/state.py:14`) — feeds renderers.
- `apply_moves`, `random_scrambles` (`src/rubik/cube/env.py`) — generate
  states for tests and the HTML preview.
- `FACE_NAMES`, `MOVE_STRINGS` (`src/rubik/notation/moves.py:9,11`) — face
  ordering and move labels in sequence frames.
- M1 preview CSS theme (`generate_cubie_2x2_rotations.py:79–96`) — copied
  into the M3 preview generator for visual consistency.
- M1 within-face geometry (`oracle/cubie.py:141–154`, `FACE_SLOTS`) — the
  reshape contract; not imported (different layer), but matched.

## Verification

End-to-end gate (mirrors prior milestones):

```sh
uv run pytest                 # all prior tests + new viz tests green
uv run ruff check             # clean
uv run ruff format --check    # clean
uv run python visuals/scripts/generate_cube_preview_2x2.py
open visuals/cube_preview_2x2.html
```

Manual eyeball check on the HTML preview:
- Solved cube: each face is a uniform 2x2 of one color, palette matches
  speedcubing convention.
- After R: U-face top row swaps with F-face top row swaps with D-face top
  row swaps with B-face bottom row (the M1 cycle for R).
- Depth-20 scramble: looks scrambled; its inverse-applied counterpart in
  the side-by-side equals solved.
- Visual consistency with `visuals/cubie_2x2_rotations.html` (same dark
  theme, same sticker shape).

## Out of scope

- 3D / web frontend (M9 stretch).
- 3x3 rendering — happens automatically once `CUBE_3X3` lands at M5
  (renderer is already parametric on `spec`), but the preview generator
  stays 2x2-named per the `_2x2` naming convention. M5 will add a
  `generate_cube_preview_3x3.py` companion.
- matplotlib-based renderers (avoided per CLAUDE.md unless we hit a need).
- Animated transitions / move-by-move morphs.
