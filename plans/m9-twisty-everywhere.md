# M9.2 Block 1A — Twisty everywhere (replace iso)

## Context

The `design-port-sequence` block established a chrome-palette editorial UI
with three section iii render modes (`net` / `iso` / `dual`). The
follow-on `3d-iso-and-split` block ported a homegrown SVG iso renderer
(`IsoCubeRenderer.tsx`, 73 lines, 30° iso projection of U+F+R faces) into
that switch. M9.2's original goal was to swap our hand-rolled 3D for
cubing.js's `<twisty-player>` — a real WebGL puzzle viewer.

After research (`.planning/research/m9-twisty-player.md`) and a scope
discussion, the M9.2 work expands into three sub-blocks. **This plan
covers Block 1A only**: get cubing.js's twisty-player rendering in every
3D context the app currently has, plus a new section iv "watch the solve"
animated player. Follow-on:

- **Block 1B** — per-card move animations on 2D + 3D synchronized clock.
- **Block 1C** — sections ii/iii/iv share `activeStepIdx` for cross-surface
  navigation.

**Key decision baked in.** cubing.js exposes no public sticker-color
override API ([issue #424](https://github.com/cubing/cubing.js/issues/424)
filed 2026-05-07; maintainer estimate ≥1 year). We accept palette
divergence: 2D net keeps the chrome palette (`#f5f3ee` warm cream,
designer-tuned greens/reds); all 3D contexts ship cubing.js stock colors
(true white, saturated WCA). This is a known visible regression vs. the
chrome unification the iso block achieved — documented as a design
trade-off, not a bug.

**Scope boundary.** This block is *infrastructure*: replace one renderer
with another, add one new section. **No animation behavior changes** —
cards still show a static post-move state, just rendered by a different
component. **No cross-surface state sharing** — sections ii/iii/iv stay
independent. The user vision of synchronized animations across surfaces is
1B/1C's domain; building that on top of this block's clean
twisty-everywhere baseline is materially easier than weaving it through
the iso/twisty mix.

Branch `m9-twisty-player` already exists off `main` HEAD `e0a063b`. LOG
block `2026-05-10 — Block 1A: Twisty everywhere (replace iso) 🟡` already
opened.

## Approach

`cubing@0.63.3` is already in `web/package.json` — no install. cubing.js
ships with three.js bundled and ESM exports; Vite 8 handles it natively.
The static import graph from `cubing/twisty` is small (~41 KB gz entry
shell); the 3D scene + three.js are dynamically imported inside the player
only when a `<twisty-player>` mounts. Lazy-loading the wrapper via
`React.lazy(() => import("./TwistyPlayerWrapper"))` keeps app first-paint
unaffected on flows that never touch 3D.

Static-mode recipe (per the research artifact):

```
experimental-setup-alg = <scramble moves joined>
experimental-setup-anchor = "end"
alg = <solution[:N] joined w/ space>
control-panel = "none"
background = "none"
back-view = "none"
tempo-scale = 0
hint-facelets = "none"
+ CSS: pointer-events: none           (disable drag-rotate)
```

For a card showing post-step-N state, the cleanest formulation is
`experimental-setup-alg=<scramble-moves-joined> + alg=<solution[:N]
joined>` with `anchor=end` — the player displays exactly
`applyMove(scramble, solution[:N])`, matching FlatCubeRenderer's ground
truth. The scramble is available as `scrambleMoves` in App state
(string[] of QTM moves).

The app's `scrambleState` is a 54-char facelet, but `scrambleMoves` (the
move list that produced it) is the cleaner setup-alg input for
twisty-player. They're both already in App state and stay in sync via
`handleScramble` / `handleClear` / `handleSetState`. Use `scrambleMoves`
joined with spaces. (If a state was pasted via `handleSetState`,
`scrambleMoves` is `[]` and twisty-player just shows the solved-cube +
alg=solution[:N]. Acceptable for 1A; revisit if needed for state-paste
roundtrips in a later block.)

For section iv, the same component with `mode="animated"` and a larger
canvas, no zero-tempo, default `control-panel="bottom-row"` —
twisty-player's own play/pause/scrubber UI is fine for v1. We layer
interactivity on top in 1C.

## Phases

Six atomic commits. Each phase ends with the app runnable; tests green
where applicable; nothing half-wired.

### P0 — Scaffold + static HTML preview

**Touch.**
- New `web/src/components/TwistyPlayerWrapper.tsx` (imperative, ref-based,
  `new TwistyPlayer(config)` inside `useEffect`; modes `"static"` and
  `"animated"`; props `{ scrambleAlg, solutionAlg, sizePx, mode, testId? }`).
- New `web/src/types/twisty-player.d.ts` (JSX runtime augmentation for
  `react/jsx-runtime` so `<twisty-player>` is typed if anyone uses JSX
  later — wrapper itself uses imperative path, so this is belt-and-braces).
- New `web/preview/twisty-player.html` (static preview via cubing CDN
  `https://cdn.cubing.net/v0/js/cubing/twisty`). Variants: solved, length-7
  scramble, length-20 scramble × sizes 90/130/220 × static + animated.
  Mirror the `iso-cube.html` structure (light + dark theme panes).
- New `web/src/components/TwistyPlayerWrapper.test.tsx` (vitest contract,
  modeled on `IsoCubeRenderer.test.ts`). Cases: mode=static hides
  control-panel + sets pointer-events: none; mode=animated keeps
  control-panel; solutionAlg prop change re-mounts player; testId honored
  with default + override + null. Note: cubing.js renders to a real
  `<twisty-player>` custom element under jsdom; assertions target the
  React-managed wrapper attrs and the imperative config snapshot, not the
  shadow DOM.

**Verify.** `pnpm --prefix web test` passes (existing 59 + new ~6).
`open web/preview/twisty-player.html` shows all variants rendering
without console errors. `pnpm --prefix web lint` clean. `pnpm --prefix web
build` succeeds. **No app changes** — the wrapper is import-ready but
unused.

### P1 — Cards: iso branch → twisty-static

**Touch.**
- `web/src/components/SolutionCard.tsx`: replace `<IsoCubeRenderer>` in
  the `renderMode === "iso"` branch with `<TwistyPlayerWrapper
  mode="static" scrambleAlg={scrambleAlg} solutionAlg={solutionAlg.slice(0,
  stepNum).join(" ")} sizePx={sizePx} testId="twisty-cube" />`. Need new
  props on SolutionCard for `scrambleAlg` and `solutionAlg` (the prefix to
  apply); plumbing comes from SolutionGrid.
- `web/src/components/SolutionGrid.tsx`: pass `scrambleMoves` (joined to
  string) as `scrambleAlg` and the full `solution` array down to each
  card. Each card slices `solution.slice(0, stepNum)`.

**Verify.** Manual: scramble + solve → toggle to 3D → cards render twisty
players showing the same per-step state the flat net shows in 2D mode.
The dual branch still uses `IsoCubeRenderer` (untouched until P2). Lint
clean; vitest still 59+ green; e2e `solution-grid.spec.ts` test 1 will
break (asserts `polygon[data-face]`); that's expected — fix in P5.

### P2 — Cards: dual (split) branch → twisty-static

**Touch.**
- `web/src/components/SolutionCard.tsx`: in the `renderMode === "dual"`
  branch, swap the right-side `IsoCubeRenderer` for
  `<TwistyPlayerWrapper mode="static" testId="twisty-cube-pair" ...>`.
  Left side stays `<FlatCubeRenderer testId="flat-cube-pair" />`.
- `.render-pair` CSS unchanged (`web/src/styles/components.css`).
- Sizing: keep `DUAL_SIZE_BY_COLS` halving logic in SolutionGrid.

**Verify.** Manual: toggle to split → flat (chrome) + twisty (cubing.js
stock) render side-by-side; visible palette divergence per the locked
decision. Lint + vitest green; e2e split test will break — fix in P5.

### P3 — Section iv "watch the solve" animated player

**Touch.**
- New `web/src/components/SectionFour.tsx`: SectionHeader (`roman="iv."
  name="watch the solve"`) + `<TwistyPlayerWrapper mode="animated" />`
  centered in a fixed-aspect container. Larger size (e.g. 360px square,
  full app-column width is 920px so plenty of room). Renders only when
  `solution !== null`; otherwise omitted (or shown empty with dim
  placeholder text — TBD visual eyeball during build).
- `web/src/App.tsx`: render `<SectionFour scrambleAlg={...}
  solutionAlg={...} />` between `<SolutionGrid>` and the error pre /
  `<SolvedFooter>`.
- New CSS in `web/src/styles/components.css`: `.section-four-stage`
  centering wrapper; minimal — twisty-player provides its own chrome via
  `control-panel="bottom-row"`.

**Verify.** Manual: scramble + solve → section iv appears below the cards
with a Play button → click play → cube animates from scrambled to solved;
scrubber works. Reset (Clear) → section iv disappears (or shows
placeholder). Lint + vitest green; e2e: add a smoke test asserting
`section-four` testid appears post-solve and twisty-player attaches.

### P4 — Retire IsoCubeRenderer + colocate cubePalette

**Touch.**
- Delete `web/src/components/IsoCubeRenderer.tsx`.
- Delete `web/src/components/IsoCubeRenderer.test.ts` (188 lines, 17
  cases — none of which other tests depend on per the explore agent).
- Delete `web/preview/iso-cube.html`.
- Move `web/src/styles/cubePalette.ts` →
  `web/src/components/cubePalette.ts` (it was a shared module for two
  consumers; FlatCubeRenderer is now the sole consumer, colocate).
- Update `web/src/components/FlatCubeRenderer.tsx` import:
  `"../styles/cubePalette"` → `"./cubePalette"`.
- Remove the iso import from `SolutionCard.tsx` (already orphaned by P1
  + P2 but the import line lingers).
- The CSS `.render-pair` stays (still used by P2's dual branch with
  flat + twisty).

**Verify.** `grep -r "IsoCubeRenderer" web/src web/e2e web/preview` →
zero hits except this commit's deletion record. `pnpm --prefix web
build` clean. `pnpm --prefix web lint` clean. `pnpm --prefix web test`
goes 59 → 53 (drop 17 iso cases, add ~6 twisty cases per P0; final
~48 — exact count to confirm in the commit).

### P5 — E2E refresh + verification sweep

**Touch.**
- `web/e2e/solution-grid.spec.ts` lines 26–79: rewrite the two iso-aware
  tests.
  - Test 1 (3D toggle): replace `polygon[data-face]` selectors with
    twisty-player-rooted selectors. Cubing.js's `<twisty-player>` exposes
    a shadow DOM containing a canvas; the simplest stable selector is
    `[data-testid="twisty-cube"]` → assert it's present (1 match) when
    3D is active; assert it's absent when 2D is active. Drop the
    polygon-count contract entirely — that was iso-specific. Keep
    rect[data-pos]=54 for 2D mode.
  - Test 2 (split toggle): replace `iso-cube-pair` polygon assertions
    with `twisty-cube-pair` presence. Keep `flat-cube-pair` rect
    assertions (FlatCubeRenderer untouched).
- `web/e2e/scramble-solve.spec.ts` if it exists and references iso —
  per the explore agent it doesn't, but verify.
- New `web/e2e/section-four.spec.ts`: smoke test. Scramble + solve →
  assert section-four testid is visible → assert at least one
  `twisty-player` element is in its subtree → click play → assert
  `twisty-player` exposes some "playing" indicator (TBD, may need DOM
  inspection during build).

**Final verification sweep at close.**
- `uv run pytest`: 707 passed (no Python touched).
- `pnpm --prefix web test`: ~48 passed.
- `pnpm --prefix web test:e2e`: stub-mode green.
- `PLAYWRIGHT_REAL_BACKEND=1 pnpm --prefix web test:e2e`: real-backend green.
- `pnpm --prefix web build`: clean, bundle visualizer report inspected
  for three.js code-split (research recommendation; one-time check).
- `pnpm --prefix web lint`: clean.
- `uv run ruff check src/ tests/ scripts/codegen/`: clean (none touched).

## Critical files

**Modify:**
- `web/src/components/SolutionCard.tsx` (P1, P2, P4)
- `web/src/components/SolutionGrid.tsx` (P1 — pass scrambleAlg, solutionAlg)
- `web/src/components/FlatCubeRenderer.tsx` (P4 — palette import path)
- `web/src/App.tsx` (P3 — render SectionFour)
- `web/src/styles/components.css` (P3 — section-four-stage rule)
- `web/e2e/solution-grid.spec.ts` (P5)

**Create:**
- `web/src/components/TwistyPlayerWrapper.tsx` (P0)
- `web/src/components/TwistyPlayerWrapper.test.tsx` (P0)
- `web/src/types/twisty-player.d.ts` (P0)
- `web/preview/twisty-player.html` (P0)
- `web/src/components/SectionFour.tsx` (P3)
- `web/e2e/section-four.spec.ts` (P5)

**Move:**
- `web/src/styles/cubePalette.ts` → `web/src/components/cubePalette.ts` (P4)

**Delete:**
- `web/src/components/IsoCubeRenderer.tsx` (P4)
- `web/src/components/IsoCubeRenderer.test.ts` (P4)
- `web/preview/iso-cube.html` (P4)

## Reuse

- `web/src/components/SectionHeader.tsx` — used as-is for section iv.
- `web/src/components/FlatCubeRenderer.tsx` — untouched; remains the 2D
  renderer with chrome palette (consumes `cubePalette.ts` after P4 move).
- `web/preview/iso-cube.html` structure — mirrored for
  `web/preview/twisty-player.html` (light + dark panes, sizes × variants).
  Don't copy the iso math — twisty-player draws itself; we just embed it.
- Research artifact `.planning/research/m9-twisty-player.md` —
  authoritative reference for `TwistyPlayerConfig` API, static-mode
  attribute combo, lazy-loading pattern, multi-instance renderer pool.
- `IsoCubeRenderer.test.ts` structure — modeled for the
  TwistyPlayerWrapper test (testId default/override/null pattern,
  size-prop coverage).
- LOG block follows cc-process: `<plan SHA>` (this plan) +
  `<open-block SHA>` + P0–P5 SHAs + `<close SHA>` recorded in **Commits:**.

## Verification (end-to-end)

After P5 commit, with the dev runner:

```
bin/site up --real        # FastAPI :8000 + Vite :5173 with the real model
open http://localhost:5173
```

Manual flow:
1. Page loads, header + sections i/ii/iii visible, no section iv yet.
2. Click *scramble* → state grid + moves grid populate; cards in
   section iii show solved-→-... per-step (may be empty if solver hasn't
   run; depends on existing UX). Section iv still hidden.
3. Click *solve* → cards populate with the solution sequence; section iv
   appears below the grid with a play button.
4. Toggle render mode 2D → 3D → split:
   - 2D: chrome palette, FlatCubeRenderer.
   - 3D: cubing.js stock palette, twisty-player static per card.
   - Split: flat + twisty side-by-side per card.
   - Palette divergence between 2D and 3D is visibly present — accepted.
5. Section iv: click play → twisty-player animates the full solve;
   scrubber works.
6. Click *clear* → state resets; section iv disappears.

Automated:
- `pnpm --prefix web test` 48 green.
- `pnpm --prefix web test:e2e` (stub) green.
- `PLAYWRIGHT_REAL_BACKEND=1 pnpm --prefix web test:e2e` green.
- `pnpm --prefix web build` clean.
- `pnpm --prefix web lint` clean.
- `uv run pytest` 707 passed (smoke; nothing Python should change).

## Out of scope (deferred to 1B / 1C)

- **Per-card move animations** (1B): cards in 1A render the static
  post-move state via `tempo-scale=0`, not an animated single-move loop.
- **2D animation system** (1B): FlatCubeRenderer stays static; it doesn't
  learn animated face rotation or wrap-sticker sliding in this block.
- **Section iv timing-sync to 2D** (1B): twisty-player drives its own
  animation in section iv; FlatCubeRenderer in section iii does not
  follow.
- **Cross-surface `activeStepIdx` sharing** (1C): sections ii/iii/iv each
  manage their own state. MovesGrid stays display-only; clicking a card
  doesn't scrub section iv; section iv playing doesn't highlight
  MovesGrid cells.
- **Custom palette in twisty-player** (out of M9 entirely): no upstream
  API; if we ever pursue it, it's a contribution to cubing.js, not a
  consumer concern.
- **Drag-rotate on static cards**: explicitly disabled via CSS
  `pointer-events: none` per the research recommendation. If we want to
  re-enable it later (for "preview this state from any angle"), drop the
  pointer-events line.

## Risks

1. **Twisty-player one-frame flicker on mount with `tempo-scale=0`**
   (research pitfall #2). The player renders the *initial* setup-alg
   frame for one tick before settling on the alg-end state. Mitigation:
   `opacity: 0` during mount, unhide on `requestAnimationFrame` settle.
   First confirm in `web/preview/twisty-player.html` (P0) before deciding
   whether to add the opacity dance to the React wrapper.
2. **Renderer pool warm-up at column counts ≥ 4.** Cubing.js shares a
   single WebGL context after `DEFAULT_MAX_DEDICATED_RENDERERS = 2` —
   subsequent players use a shared `drawImage` pipeline. Should be fine
   at static `tempo-scale=0`; verify by spot-checking 30 cards
   (`scrambleLength=30 + solve` at `cols=6`) for visual artifacts and
   browser memory profiler in P0/P1.
3. **E2E shadow DOM piercing.** Playwright handles custom elements via
   `>>>` pierce syntax, but the simplest route is a `data-testid` on the
   wrapper `<div>` — the test asserts the wrapper presence, not the
   inner WebGL canvas. Avoids version-coupling to cubing.js shadow DOM.
4. **Bundle bloat regression.** First 3D mount drags ~250 KB gz of
   cubing.js + three.js chunks. Lazy-loading via `React.lazy` keeps this
   off the first-paint critical path for 2D-only flows. Verify in P5
   build: three.js chunks must be in their own files, not the main
   bundle.
5. **`experimental-` attribute deprecation.** `experimental-setup-alg`
   and `experimental-setup-anchor` are the load-bearing static-mode
   inputs. Pin `cubing@~0.63.3` in `package.json` (use `~` not `^`) so a
   minor-version bump can't silently change the contract. If the
   research-noted PR for stable `setup-alg` ever lands, migrate then.
6. **Bundle-size sanity check** runs once at P5; if the tsc/vite build
   ships three.js inside the main chunk (no code-split), revisit
   `React.lazy` wiring before close.
