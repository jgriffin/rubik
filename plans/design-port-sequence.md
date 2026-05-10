# Plan — Design port: Sequence layout

Port the Claude Designer "Sequence" handoff into the existing React/Vite web app at `web/`. v1 covers full visual chrome on the 2D net renderer; 3D iso, split view, the 1-column row variant, and 2×2-toggle behavior are out of scope.

## Inputs

- **Reference HTML.** `web/preview/sequence-reference.html` — the single-file standalone bundle from the handoff. Open with `open web/preview/sequence-reference.html`. This is the pixel target for every phase's acceptance check.
- **Design tokens, type scale, per-section specs.** `/tmp/rubik-handoff/design_handoff_rubik_solver/README.md` (copy worth keeping locally; do not bundle).
- **Portable cube logic.** `/tmp/rubik-handoff/design_handoff_rubik_solver/source/cube.js` — `RubikLib.svgNet`, `svgIso`, `rotY2`, demo data. v1 only needs to match `svgNet`'s output style; iso + split helpers can stay in the handoff bundle until v2.

## Out of scope (v1)

| Feature | Why deferred | Where stubbed |
| --- | --- | --- |
| 3D iso renderer (`svgIso`) | Net-only is the user-confirmed v1 scope | Render toggle disabled past `2D` |
| Split view (Y²-rotated dual iso) | Same as above | Same as above |
| 1-column row layout with prose descriptions | Materially different layout; treat as v2 variant | Column toggle excludes `1`, or routes `1` to a single-column-of-cards (no prose) |
| 2×2 cube logic toggle | Backend has 2×2 already, but wiring touches `apiHealth` model selection, env reload, etc. — separate concern | Header toggle is local UI state only; defaults to and stays on 3×3 |
| Paste-moves into section ii | Design treats `ii.` as display-only; the existing `MovesField` textarea has no analog in the reference | `MovesField` deleted; paste-state into section i remains the supported paste path |

## Salvage from `solver-demo-polish`

Keep:
- `web/src/state/parseMoves.ts` + tests — needed to render section ii cells and validate any future move input.
- `web/src/state/validateFacelet.ts` + tests — the new section-i grid wants per-face validation.
- `web/src/components/FlatCubeRenderer.tsx` — the cube-drawing logic (sticker positions, color map, facelet indexing). Restyle, don't rewrite. The V2 in-SVG label overlay (`Overlay[]` prop) is dropped — labels live in card chrome now.
- `web/src/api/client.ts` — unchanged.
- `web/src/state/applyMove.ts` + `faceletMoves.ts` — unchanged.

Replace:
- `App.tsx` — full restructure into 920px column with three roman sections.
- `MoveStripView` → new `SolutionGrid` + `SolutionCard`.
- `StateField` (textarea) → `StateGrid` (6 face inputs with auto-advance + cascading paste).
- `MovesField` (textarea) → deleted; `ii.` is rendered directly from current scramble.
- `ScrambleControls` → folded into the section-i header's "length pack" chip.
- `StepControls` + `MoveList` → deleted; card selection in `iii.` replaces them.

## Design tokens

Land in `web/src/styles/tokens.css` as CSS custom properties on `:root`:

```css
--bg:     #f3efe6;   /* page, warm bone */
--paper:  #fbf8f1;   /* cards + inputs */
--ink:    #131210;   /* primary text */
--dim:    #6f6a60;   /* captions, meta */
--rule:   #d9d2c2;   /* 1px borders */
--accent: #c64a23;   /* italic glyphs, active states */
```

Fonts via Google `<link>` in `index.html`: Fraunces (opsz 9..144, weights 300/400/500) + JetBrains Mono (400/500/600). No bundled font files.

## Phases

Each phase is one atomic commit on this branch unless noted. Each phase keeps `pnpm dev` green and leaves the app navigable.

### Phase 0 — Foundations
**Touches:** `web/index.html`, `web/src/index.css`, new `web/src/styles/tokens.css`.
**Does:** Add Google Fonts links. Drop tokens into `:root`. Replace the body reset to use `--bg`, `--ink`, Fraunces as the default serif, JetBrains Mono available via class. No component changes.
**Acceptance:** Page background is cream, body text reads in Fraunces, no visual regressions in existing components beyond colour shift.

### Phase 1 — Layout shell + header + footer
**Touches:** `App.tsx`, new `web/src/components/Wordmark.tsx`, new `web/src/components/CubeSizeSwitch.tsx`, new `web/src/components/SolvedFooter.tsx`.
**Does:** Center the page in a 920px column (`padding: 56px 28px 96px`). Drop the existing H1 + the `<h2>animation player</h2>` block. Add the "rubik *solver*" wordmark + 2×2/3×3 segmented control (toggle is local state, no behavior). Add the italic *Solved.* footer wired from real `SolveStats` (`time_ms` → `240 ms`, model id from `apiHealth`).
**Acceptance:** Reference HTML's header bar and footer match side-by-side at the same column width. Existing strip + controls still render below the new chrome (will be replaced in later phases).

### Phase 2 — Section i — starting state
**Touches:** new `web/src/components/StateGrid.tsx`, new `web/src/components/LengthPack.tsx`, new `web/src/components/SectionHeader.tsx`. Delete `StateField.tsx`, `ScrambleControls.tsx`.
**Does:** Six-face input grid (U L F R B D) with per-face `<input>`. Auto-advance on the 9th char, backspace-at-empty jumps back, paste cascades into all subsequent inputs (strip non-letters, uppercase). Per-face caption: italic accent letter + dim mono face name. Right-side controls: `clear` text-button + 1px hairline + length-pack chip (SCRAMBLE button | 90px slider | italic value). Section header reuses the new `SectionHeader` (`i.` italic accent + "STARTING STATE" mono uppercase).
**Acceptance:** Manual paste of a 54-char facelet string fills all six faces. `validateFacelet` drives an accent-color underline on invalid faces. Scramble button still calls `apiScramble` and updates the grid.

### Phase 3 — Section ii — moves to apply
**Touches:** new `web/src/components/MovesGrid.tsx`. Delete `MovesField.tsx`, `MoveList.tsx` (or whatever is showing the move list in `App.tsx` today).
**Does:** 10-column ruled grid. Each cell renders `R` / `U` / etc. in Fraunces 18px; modifier (`'` → `′`, `2` stays) renders italic-accent immediately right of the letter. Empty trailing cells = dashed 55%-opacity border + `·` glyph. Right-side header meta: "N moves" in dim mono.
**Acceptance:** Section ii reflects the current scramble exactly; cell count and "N moves" stay in sync with `apiScramble` results.

### Phase 4 — Section iii — solution
**Touches:** new `web/src/components/SolutionGrid.tsx`, new `web/src/components/SolutionCard.tsx`, new `web/src/components/RenderModeSwitch.tsx`, new `web/src/components/ColumnsSwitch.tsx`. Restyle `FlatCubeRenderer.tsx` (drop the `Overlay[]` prop, drop in-SVG step/label injection; labels are now external chrome). Delete `MoveStripView.tsx`, `StepControls.tsx`.
**Does:** Section header with render-mode segmented control (`2D` active, `3D`/`split` disabled in v1) and columns segmented control (`2`/`3`/`4`/`6`; `1` deferred — disabled or routed to a degenerate column-of-cards). CSS `--cols` drives the grid. Per-card: step number `00` mono dim top-left, move glyph Fraunces top-right (sized per column count: 26 at 3-col, 32 at 2-col, 16 at 5/6-col), `FlatCubeRenderer` below at column-dependent px size (220/200/130/90). First card is `step 00` with dashed border + italic dim "start" label. Selected card = 1px accent border + 1px accent box-shadow ring; clicking a card sets `activeStepIndex` (replaces the old stepper).
**Acceptance:** Render at each supported column count matches the reference HTML. Card click selection works. Solver run end-to-end populates the grid in real time as solve completes.

### Phase 5 — Cleanup + e2e refresh
**Touches:** `web/e2e/*.spec.ts`, dead-code sweep.
**Does:** Update `state-moves-paste.spec.ts` → `state-grid-paste.spec.ts` (cascading paste assertions). Replace `strip-view.spec.ts` → `solution-grid.spec.ts` (column toggle, card selection, step-00 start card). `scramble-solve.spec.ts` likely survives unchanged. Delete any orphaned imports and the `web/preview/strip-view.html` if it no longer renders meaningfully.
**Acceptance:** `pnpm lint`, `pnpm typecheck`, `pnpm test`, and Playwright e2e all green. `web/preview/sequence-reference.html` opened side-by-side with `pnpm dev` shows no daylight on header / footer / sections i / ii / iii at 3-column 2D.

### Close
- Update LOG.md Outcome with files touched, decisions worth remembering, anything that surfaced as backlog.
- Merge `design-port-sequence` → `main` with `--no-ff`, delete the branch.

## Open questions

None blocking. Decided going in:
- **v1 scope** = 2D net only, full chrome (user-confirmed).
- **Branch** = new `design-port-sequence` off main (post-merge of `solver-demo-polish`).
- **Paste-moves UX** = dropped in v1 (no analog in the reference; revisit if missed).
