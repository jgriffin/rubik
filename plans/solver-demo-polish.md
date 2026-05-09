# Solver demo polish

Drive-by polish on top of M9.1 ([`plans/m9.1-solver-demo.md`](m9.1-solver-demo.md)).
Branch `solver-demo-polish` (off `361d29a`). LOG block opened in `7f1bd64`.
Pulls most of M9.3's "alt input + polish" forward, plus a new strip-view
visualization that wasn't in any prior plan.

## Context

The M9.1 demo works end-to-end (scramble → solve → step-through animation),
but it's a **demonstration** rather than an **inspection tool**. Three gaps:

1. **Length is locked.** The Scramble button hardcodes length 14 — we can't
   eyeball depth-1 trivial cases, depth-30 hard ones, or anything in between
   without editing source.
2. **The state and solution are opaque.** No way to copy out the current
   scramble state to share or save, no way to paste in a state (or a known
   move sequence) to inspect a particular position.
3. **Step-through plays one frame at a time.** To verify the network's
   solution is actually correct, you have to manually click through 14
   states, holding each in your head — no way to see the entire trajectory
   at once.

This block addresses all three. Mobile, share-URL, manual color-grid entry
stay in M9.3.

## Locked decisions

- **Move parsing:** hand-roll over the existing `MoveStr` union (training
  is QTM-only; cubing.js's broader `Alg` parser would have to reject
  `R2`/`Rw`/`M`/`x` anyway, so the broader surface is overkill).
- **Strip layout:** 6-column CSS grid wrap with small cubes (compact at
  length 30; `[move-label, cube, step#]` per cell; clicking a cell jumps
  the secondary animation player to that step).
- **Length picker:** `<input type="range">` 1–30 + numeric readout +
  "Surprise Me" button (random length in [1, 30]).
- **Move-paste semantics:** pasting moves replaces `solution` only; cube
  state is independent (pasted via a separate state field). Setting state
  clears `solution` and resets `stepIdx`; setting moves resets `stepIdx`
  but leaves `scrambleState` alone.

State model after the polish:

- `scrambleState: Facelet` — input to the solver. Set by Scramble or by
  pasting into the state field. Setting it clears `solution` and resets
  `stepIdx`.
- `solution: MoveStr[]` — move sequence being stepped through. Set by
  Solve or by pasting into the moves field. Setting it resets `stepIdx`.
- `stepIdx: number` — playback cursor (0..solution.length).
- `displayedState = applyMoves(scrambleState, solution.slice(0, stepIdx))`
  — derived. Drives both the secondary big cube and the strip-view
  current-step highlight.

## Commit breakdown

Six atomic commits, each via a single `general-purpose` subagent dispatch
(matching M9.1's cadence). This file lands as commit 1.

### Commit 1 — plan

This file. No code.

### Commit 2 — move parser + formatter

**New files**
- `web/src/state/parseMoves.ts` — exports
  - `parseMoves(s: string): MoveStr[]` — split on whitespace (collapse
    runs), trim, validate each token against the `MoveStr` union; throws
    `ParseMovesError` with the offending token + position on failure;
    accepts the empty string (returns `[]`).
  - `formatMoves(moves: MoveStr[]): string` — joins with single spaces.
  - `class ParseMovesError extends Error` carrying
    `{ token: string, index: number }`.
- `web/src/state/parseMoves.test.ts` — vitest, covers:
  - Valid sequences (single move, length-14 random, length-30, empty).
  - Whitespace tolerance: leading/trailing whitespace, double spaces,
    tabs, newlines.
  - Round-trip: `parseMoves(formatMoves(xs)) === xs` over a sampled set.
  - Rejection: `R2`, `Rw`, `M`, `x`, `r`, `RR`, `R '`, garbage; assert
    `ParseMovesError` with correct token + index.

No UI integration; pure module.

**Commit message** — `web: move-string parser + formatter`

### Commit 3 — length picker

**New files**
- `web/src/components/ScrambleControls.tsx` — replaces
  `ScrambleButton.tsx`:
  - `<input type="range" min=1 max=30 step=1>` slider + adjacent numeric
    readout.
  - **Scramble** button (calls `onScramble(length)`).
  - **Surprise Me** button — picks a random integer in [1, 30] and fires
    `onScramble(randomLength)` immediately, also updating the slider so
    the user sees what was rolled.
  - Test IDs: `scramble-button`, `surprise-button`, `length-slider`,
    `length-readout`.

**Deleted files**
- `web/src/components/ScrambleButton.tsx`

**Modified files**
- `web/src/App.tsx` — owns `length: number` state (initial 14); passes
  to `ScrambleControls`; existing `handleScramble(length)` already
  accepts a length param so the call signature is compatible.
- `web/e2e/scramble-solve.spec.ts` — extend stub-mode test:
  - Drag slider to length 6
    (`page.locator('[data-testid=length-slider]').fill('6')`),
    click Scramble, assert move list count = 6 after Solve.
  - Click Surprise Me, assert slider readout changed to a value in
    [1, 30].

Backend already accepts `length: int = 20 (ge=0, le=100)` — no server
change.

**Commit message** — `web: scramble length picker (slider + Surprise Me)`

### Commit 4 — state + moves text fields (paste/copy)

**New files**
- `web/src/state/validateFacelet.ts` (+ test) — pure validator. 54 chars,
  charset `URFDLB`, exactly 9 of each. Returns
  `{ ok: true; state: Facelet } | { ok: false; error: string }` with
  descriptive errors.
- `web/src/components/StateField.tsx` — `<textarea>` showing the current
  `scrambleState` (54 chars, monospace, fixed rows=2). Below:
  - **Copy** button → `navigator.clipboard.writeText(scrambleState)`.
  - **Set state** button → validates input via `validateFacelet`; on
    success calls `onSetState(parsed)`; on failure renders inline error.
  - Test IDs: `state-field`, `state-copy`, `state-set`, `state-error`.
- `web/src/components/MovesField.tsx` — `<textarea>` showing
  `formatMoves(solution)`. Below:
  - **Copy** button → clipboard.
  - **Set moves** button → calls `parseMoves`; on success calls
    `onSetMoves(parsed)`; on `ParseMovesError` renders inline error
    (e.g., `"Invalid move 'R2' at position 5"`).
  - Test IDs: `moves-field`, `moves-copy`, `moves-set`, `moves-error`.
- `web/e2e/state-moves-paste.spec.ts` (stub-mode) — flows:
  1. Scramble length 8 → click state Copy → verify clipboard text =
     scrambleState.
  2. Paste a known facelet (compute test fixture client-side via
     `applyMoves(SOLVED_3X3, ['R', 'U', "R'"])`) → click Set →
     assert renderer reflects the new state via shifted `data-color`
     attributes.
  3. Paste `"R U R' U' F"` into moves field → click Set → assert
     `data-testid="move-item"` count = 5 with the right labels.
  4. Paste `"R2 U"` → click Set → assert error message visible and
     solution unchanged.

**Modified files**
- `web/src/App.tsx` — wire `handleSetState` (sets `scrambleState`,
  clears `solution`, resets `stepIdx`) and `handleSetMoves` (sets
  `solution`, resets `stepIdx`, leaves `scrambleState` alone).

**Commit message** — `web: paste/copy text fields for cube state + moves`

### Commit 5 — strip view + animation-player demote

**New files**
- `web/src/components/MoveStripView.tsx` — renders `N+1` cells in a
  6-column CSS grid where `N = solution.length`:
  - Cell 0: label `"Start"` + cube of `scrambleState` + step number `0`.
  - Cell `i` (1..N): label = move that produced this state (e.g., `"R"`)
    + cube of `applyMoves(scrambleState, solution.slice(0, i))` +
    step number `i`.
  - Cubes use `FlatCubeRenderer` with `sizePx={80}`.
  - Active cell (where `stepIdx === i`) has a visual border highlight
    (2px solid accent color).
  - Click a cell → `onJumpTo(i)` — sets `stepIdx = i`.
  - Memoize cell states with `useMemo` so the whole grid only recomputes
    when `scrambleState` or `solution` change.
  - Test IDs: `strip-view`, per-cell `strip-cell-${i}`,
    `data-active="true|false"`.
- `web/preview/strip-view.html` (**the user's "favorite pattern"**) —
  static HTML mirroring the React grid:
  - Three demos side-by-side: length 0 (just Start), length 14 (typical),
    length 30 (deep). Light + dark themes.
  - Pure-static SVG generation duplicating the renderer logic (mirrors
    the existing `flat-cube.html`).
  - Active-cell highlight shown on one demo so the visual contract is
    visible without running the app.
  - Open with `open web/preview/strip-view.html` for human eyeball.
- `web/e2e/strip-view.spec.ts` (**real-backend** spec, gated by
  `PLAYWRIGHT_REAL_BACKEND=1` like `step-through.spec.ts`):
  - Scramble length 8 → Solve → wait for solution → assert
    `[data-testid^=strip-cell-]` count = 9 (states 0..8) → click cell 4
    → assert step counter shows `"4 / 8"` → assert cell 4 has
    `data-active="true"`.

**Modified files**
- `web/src/App.tsx` — layout reflow:
  - **Top:** state field + moves field (side-by-side at desktop width;
    inline error space).
  - **Below:** scramble controls (slider + Scramble + Surprise) + Solve
    button.
  - **Main:** `MoveStripView` (primary).
  - **Secondary:** the existing big `FlatCubeRenderer` (sizePx=240) +
    `StepControls`, demoted to a smaller right-side column or below the
    strip with reduced visual weight.
  - Health JSON debug stays at the bottom.

**Commit message** — `web: per-move strip view as primary visualization`

### Commit 6 — close

Full verification sweep (matching M9.1's pattern):

- `uv run pytest` (default, fast; expect 707 passed unchanged).
- `uv run pytest -m slow` (real-model integration; unchanged).
- `pnpm --prefix web test` (vitest — should grow by parseMoves +
  validateFacelet cases).
- `pnpm --prefix web test:e2e` (Playwright stub-net).
- `PLAYWRIGHT_REAL_BACKEND=1 pnpm --prefix web test:e2e` (Playwright real).
- `uv run ruff check src/ tests/ scripts/codegen/` (unchanged scope).
- `pnpm --prefix web lint` (ESLint).

If anything fails, fix in a small pre-close commit (mirror M9.1's
`287a2af` test-isolation fix). Then close the LOG.md block:

- Append **Outcome** with what landed, decisions worth recording, and
  followups (mobile / share-URL / color-grid entry stay in M9.3).
- Fill **Commits:** with the SHA list.
- Flip 🟡 → ✅.

**Commit message** — `log: close solver demo polish`

## File map summary

**New (web frontend):**
- `web/src/state/parseMoves.ts` + test
- `web/src/state/validateFacelet.ts` + test
- `web/src/components/ScrambleControls.tsx`
- `web/src/components/StateField.tsx`
- `web/src/components/MovesField.tsx`
- `web/src/components/MoveStripView.tsx`
- `web/preview/strip-view.html`
- `web/e2e/state-moves-paste.spec.ts`
- `web/e2e/strip-view.spec.ts`

**Modified:**
- `web/src/App.tsx` — layout reflow + new handlers + length state
- `web/e2e/scramble-solve.spec.ts` — extend with slider drag + Surprise

**Deleted:**
- `web/src/components/ScrambleButton.tsx` — superseded by
  `ScrambleControls`

**No changes:** Python backend, `src/rubik/`, `tests/`, `scripts/`, plans
under `plans/m9*`. Backend already accepts `length` 0–100.

## Existing utilities being reused

| Utility | Path | Used by |
|---|---|---|
| `applyMoves(facelet, moves)` | `web/src/state/applyMove.ts` | StripView cell states |
| `MoveStr` union (12 entries) | `web/src/state/faceletMoves.ts` | Parser validation |
| `FACELET_MOVES` permutation table | `web/src/state/faceletMoves.ts` | (via applyMoves) |
| `SOLVED_3X3` const | `web/src/App.tsx:11` | StateField placeholder + tests |
| `FlatCubeRenderer` | `web/src/components/FlatCubeRenderer.tsx` | StripView cells (sizePx=80) |
| `apiScramble({ length })` | `web/src/api/client.ts` | Length picker (already supports param) |
| `data-testid` pattern | existing components | New components mirror it |

## Verification (end-to-end after commit 5)

1. **Visual eyeball — favorite pattern.**
   `open web/preview/strip-view.html`. Verify the 6-column grid reads
   cleanly at length 14 and length 30 across light + dark themes.
   Active-cell highlight is distinguishable. If the layout is cramped,
   iterate on the static HTML before committing the React component.
2. **Live demo end-to-end (real backend).**
   - `uv run rubik-serve` and `pnpm --prefix web dev`.
   - Drag length slider to 6 → Scramble → cube renders → Solve → moves
     appear in the strip → click cell 3 → big cube updates to step 3 →
     click Play → animation plays + strip highlights moving cell.
   - Surprise Me → fires a random-length scramble.
   - Copy state → paste into a fresh tab's state field → Set → cube
     renders correctly.
   - Paste `"R U R' U' F"` into moves field → Set → strip rebuilds with
     5 cells + start.
   - Paste `"R2 X"` → error message appears, solution unchanged.
3. **Test sweep** (commit 6).

## Out of scope (this block)

- Mobile/responsive layout (M9.3).
- URL-encoded share state `/#state=...` (M9.3).
- Manual color-grid entry (M9.3).
- Speed slider for animation player (M9.3 polish; current 500ms hardcode
  stays).
- 3D `<twisty-player>` (M9.2).
- Surfacing `final_value` from `BeamSearchResult` (parked for M9.2 or
  later).
- The `experiments/davi-2x2/analysis/` ruff failures (separate drive-by
  block).
