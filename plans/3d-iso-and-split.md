# Plan — 3D iso renderer + split view

Wire the v2 stubs left behind by `plans/design-port-sequence.md`: enable
the **iso** (3D) and **dual** (split) modes in `RenderModeSwitch`, render
the chrome-palette iso cube via a port of the handoff bundle's `svgIso`.
The 2D net mode and all section i / ii layout stay exactly as shipped in
the design-port branch — this is purely additive at the section iii
render layer.

## Inputs

- **Iso math source.** `/tmp/rubik-handoff/design_handoff_rubik_solver/source/cube.js`,
  `svgIso(facelet, sizePx, opts)` at lines 81–154. ~73 lines of pure SVG
  math — 30° iso projection (`cos π/6`, `sin π/6`), three visible faces
  (U, F, R), gap-inset stickers, viewBox auto-fitted from the projected 8
  cube corners. Zero deps. Translate to TSX.
- **Style anchor.** `web/src/components/FlatCubeRenderer.tsx` — chrome
  palette (`#f5f3ee` / `#d24244` / `#5cb27a` / `#f0c64a` / `#e08a4a` /
  `#3a6ee0`), ink stroke `#131210`, strokeMul `0.025`, `testId` prop
  contract (default `"flat-cube"`, pass `null` to suppress). Iso renderer
  mirrors all of these.
- **Toggle surface.** `web/src/components/RenderModeSwitch.tsx`
  (currently disables iso + dual with title-attr reasons),
  `web/src/components/SolutionCard.tsx` (where the renderer mounts),
  `web/src/components/SolutionGrid.tsx` (already accepts a `renderMode`
  prop but doesn't consume it).
- **Reference layout.** `web/preview/sequence-reference.html` shows the
  intended visual styling for 3D + split modes — open and inspect for
  static-preview parity.

## Out of scope

- 1-column row layout (`ColumnsSwitch` still has `cols=1` disabled — the
  prose-described row variant is a separate v2 strand).
- 2×2 cube logic toggle (still local-UI-state-only as in the design
  port).
- Animation / interpolation between iso states.
- Iso renderer in section i's state grid — those are edit surfaces, not
  display surfaces; they stay 2D.

## Salvage / unchanged

- `FlatCubeRenderer` stays exactly as-is. No prop changes.
- `App.tsx`'s `renderMode` state and `setRenderMode` plumbing already
  exist and already flow to `SolutionGrid`.
- E2E specs `state-grid-paste.spec.ts` / `scramble-solve.spec.ts` /
  `health.spec.ts` are unaffected.

## Style decisions (locked)

- **Chrome palette, both modes.** `IsoCubeRenderer` reuses
  `FlatCubeRenderer`'s `COLOR_FOR_LETTER` map verbatim. Iso applies
  designer shading on top: `shadeFront=0.96`, `shadeRight=0.88`,
  `shadeTop=1.0` (top stays full chroma, right + front are tinted by
  multiplying RGB channels — preserved from `cube.js:100-107`).
- **Ink stroke `#131210` at strokeMul 0.025**, same formula as Flat:
  `Math.max(0.012, sizePx * 0.025 / 9)` adapted for the iso unit-space
  viewBox (~5 units across).
- **Visible-only.** Iso renders 27 stickers (9×3 faces), not 54 — the
  hidden D, L, B faces are dropped. This is a known visual contract: iso
  views show the front-top-right of the cube. Hidden-face state is
  irrelevant to the render.
- **TestId contract.** `data-testid="iso-cube"` by default. `testId` prop
  override matches Flat's signature so cards can suppress the duplicate
  testid in `dual` mode.

## Atomic commits

```
1. plan: 3d iso and split           (this file)
2. log: open 3d-iso-and-split       (LOG.md entry)
3. web: phase 1 — iso cube renderer + static preview
4. web: phase 2 — wire 3D render mode (enable iso button)
5. web: phase 3 — wire split mode    (flat + iso side-by-side)
6. log: close 3d-iso-and-split       (LOG.md amend)
```

App must stay runnable at every commit; per-phase acceptance below.

## Phase 1 — IsoCubeRenderer + static preview

**Files added:**
- `web/src/components/IsoCubeRenderer.tsx` — port of `svgIso`. Same
  prop-shape as `FlatCubeRenderer` (`facelet: string`, `sizePx?: number`,
  `testId?: string | null`). Reuses the chrome `COLOR_FOR_LETTER` map.
  Tints right/front faces via a small `tint(hex, mul)` helper matching
  `cube.js:100-107`. Auto-fits viewBox from the 8 projected cube
  corners + a small stroke-width pad. SVG is rendered at the requested
  pixel size with `display: block` so parents control flow.
- `web/preview/iso-cube.html` — static preview, favorite pattern. Three
  states (solved + length-7 scramble + length-20 scramble) × three sizes
  (90 / 130 / 220 px) × light + dark theme. Inlines the iso math + chrome
  palette directly (intentional duplication, flagged in a comment header).
  Opens with `open web/preview/iso-cube.html` for visual eyeball — no
  React, no build step. Dark/light side-by-side via a `prefers-color-scheme`-
  agnostic `body[data-theme]` toggle pair on the page.
- `web/src/components/IsoCubeRenderer.test.ts` — vitest unit. Asserts:
  (a) renders 27 polygons (3 faces × 9 stickers, hidden faces dropped),
  (b) each visible-face center sticker carries the expected
  `data-color` attr (`U`, `F`, `R`) for a SOLVED facelet, (c) palette
  values for tinted faces match `tint(COLOR_FOR_LETTER[F], 0.96)` /
  `tint(COLOR_FOR_LETTER[R], 0.88)`. No vitest snapshot — explicit
  attribute checks instead, since the SVG path coordinates are a
  derived value.

**Files unchanged.** No App.tsx wiring this commit; `IsoCubeRenderer` is
not yet referenced from any production component. The component is
import-ready but inert.

**Acceptance.** `pnpm --prefix web test` passes (43 vitest now, was 42).
`pnpm --prefix web build` clean. `open web/preview/iso-cube.html` shows
all 9 cube views (3 states × 3 sizes) in light + dark, with chrome
palette consistent with the FlatCubeRenderer side-by-side check from the
design-port reference. No e2e changes.

## Phase 2 — Wire 3D render mode

**Files changed:**
- `web/src/components/SolutionCard.tsx` — accept new prop
  `renderMode: RenderMode`. Switch on `renderMode === "iso"` to render
  `IsoCubeRenderer` instead of `FlatCubeRenderer`. Both renderers receive
  the same `sizePx` (220 / 200 / 130 / 130 / 90 by column, unchanged from
  Flat — iso's slightly taller bounding box centers cleanly inside the
  existing `.render` slot).
- `web/src/components/SolutionGrid.tsx` — destructure and forward
  `renderMode` to each `<SolutionCard>` (currently it accepts the prop
  but doesn't consume it).
- `web/src/components/RenderModeSwitch.tsx` — flip
  `MODES[1]` (`iso`) entry to `disabled: false`, drop the `reason`
  title attr. Leave `dual` disabled until Phase 3.
- `web/e2e/solution-grid.spec.ts` — extend the existing render-mode
  test. Drop `await expect(page.getByTestId("render-mode-iso")).toBeDisabled()`;
  add a new step that clicks the 3D button, asserts it gains `.on`, asserts
  the rendered SVG inside `sol-card-1` contains `polygon` elements (iso) and
  no `rect[data-pos]` (Flat). Keep the dual-disabled assertion.

**Files unchanged.** Component CSS, App.tsx state, all other specs.

**Acceptance.** All vitest specs pass. Playwright stub: 12/13 → 12/13
(same count; the new assertion replaces a disabled-check). Switching
between 2D and 3D flips the renderer for every card; column toggle still
works in 3D mode. Visual eyeball: open `pnpm dev` on :5173, scramble +
solve, click 3D — all cards (including step 0 / start) show the iso view
in chrome palette.

## Phase 3 — Wire split (dual) mode

**Files changed:**
- `web/src/components/SolutionCard.tsx` — add `dual` branch. Wrap both
  renderers in a `<div class="render-pair">` that uses
  `display: flex; gap: var(--gap)` to lay them out side-by-side. Pass
  `testId={null}` to the inner FlatCubeRenderer in this mode (so the
  `flat-cube` testid stays unique to net-only mode), but keep
  `iso-cube` on the iso side (iso testid is already only used in dual or
  iso mode and doesn't collide with anything). Or — cleaner — pass
  testIds `flat-cube-pair` / `iso-cube-pair` so dual-mode tests can
  target them without ambiguity. Decide at write time based on which
  reads more clearly.
- `web/src/styles/components.css` — add `.render-pair` styles (flex,
  gap-tuned to leave breathing room without crushing the cubes; align
  vertically on the middle row of the iso since flat is taller).
- Add `DUAL_SIZE_BY_COLS` to SolutionGrid to halve sizes for split mode:

  ```ts
  const DUAL_SIZE_BY_COLS: Record<Cols, number> = {
    1: 200, 2: 160, 3: 100, 4: 90, 6: 60,
  };
  ```

  Forwarded as `sizePx` to SolutionCard when `renderMode === "dual"`.
  Net mode keeps `CUBE_SIZE_BY_COLS` (unchanged).
- `web/src/components/RenderModeSwitch.tsx` — flip `MODES[2]` (`dual`)
  to `disabled: false`, drop the `reason`.
- `web/e2e/solution-grid.spec.ts` — drop the `dual-disabled` assertion;
  add: click the split button, assert it gains `.on`, assert the rendered
  SVG inside `sol-card-1` contains BOTH `polygon` (iso) AND `rect[data-pos]`
  (Flat).

**Files unchanged.** App.tsx, all other components / specs.

**Acceptance.** All three render modes work end-to-end. Scramble +
solve, click net → cards show flat-only; click 3D → iso-only; click
split → both side-by-side. At cols=6, dual mode lands ~60 px each
(120 px total per card render-slot) — visual eyeball that this
remains legible. If too cramped, raise the cols=6 dual size to 70 and
re-eyeball.

## Phase 4 — Close

Final verification:
- `uv run pytest` (default) — should be 707 passed (no Python touched).
- `pnpm --prefix web test` — 42 → 43 (+1 from the iso component test).
- `pnpm --prefix web test:e2e` (stub) — 12/13 still passing.
- `PLAYWRIGHT_REAL_BACKEND=1 pnpm --prefix web test:e2e` —
  13/13 (1 previously-skipped real-backend spec now runs; expect green).
- `pnpm --prefix web build` + `pnpm --prefix web lint` — clean.
- `uv run ruff check src/ tests/ scripts/codegen/` — clean (none touched).

Close the LOG block; flip the design-port block's "follow-on backlog"
line to note iso + split are no longer pending.

## Risks / open questions

- **Iso polygon stroke aliasing at 60 px.** At dual-mode cols=6, the iso
  cube projects to a ~60 px footprint. The default 0.025 strokeMul might
  read as 1.5 px in unit space — visible but possibly too heavy at this
  size. If it looks chunky in the static preview, drop strokeMul to
  0.018 for sizes ≤ 90 (parameterize). Will know at static-preview
  inspection time, before any TSX wiring.
- **viewBox padding mismatch.** `cube.js`'s `svgIso` pads viewBox by
  `sw * 4`. Our chrome stroke is unit-size 0.012-clamped — different
  from `cube.js`'s default. Ensure the pad uses the same effective
  stroke-width formula or strokes will clip on the cube edges.
- **Accessibility: iso cube has no flat-cube fallback for the visible
  faces.** All current e2e renderer assertions target `data-pos` /
  `data-color` attributes that only exist on FlatCubeRenderer rects.
  Iso polygons need their own `data-color` (and optionally `data-face`)
  attrs so future tests can introspect. Land in Phase 1.
