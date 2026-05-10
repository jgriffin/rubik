# Block 1B — Per-card move animations

## Context

Block 1A ("twisty everywhere") landed all three render modes (net / iso-now-twisty / split) plus a section iv full-solve animated player. Cards currently render the **static post-move state**: net via `FlatCubeRenderer` SVG, 3D via `<twisty-player tempo-scale="0">`. The user vision pinned in 1A's close: each step card should **animate its single move once** on mount/scroll-into-view, then settle on the post-move state — both 2D and 3D — with click-to-replay. Section iv stays independent in 1B (it already animates via cubing.js's default play controls); cross-surface sync between sections ii/iii/iv is block 1C's job.

**Why now.** The static-card story is a UI dead-end — the cards render correct facelets but tell the user nothing about *which sticker moved where*. Animation is what makes the solve readable as a sequence of physical operations, not just a list of states. Block 1B builds the animation primitive; block 1C wires cross-surface scrubbing on top of it.

**Key MVP decisions confirmed with user:**
- **B-face wrap = cross-fade.** Source-side stickers fade out at the net edge while destination-side stickers fade in at B's edge. Cheapest path that doesn't read as a hard teleport. P0 preview validates before P1.
- **"Shared clock" = matched-duration independent triggers.** Both renderers fire simultaneously on mount/click; each runs its own ~600 ms animation. Sub-frame drift is invisible. True timeline coupling (single RAF driving both) is deferred to 1C where cross-surface scrubbing actually requires sample-accurate sync.

## Branch

`m9-per-card-animations` off `main` HEAD `2a01a7c` (palette-unification merge).

## Phases

Each phase = one atomic commit. App runnable throughout.

### P0 — 2D animation primitive in static HTML preview

**The speculative work.** Validates the 2D math + cross-fade-wrap geometry in isolation before any TSX. User opens the file, eyeballs, gates P1.

- New `web/preview/flat-cube-animated.html` — standalone, no React, inline SVG + JS. Imports the existing chrome palette inline (matches `web/src/components/cubePalette.ts` for fidelity; not a build-time dep).
- Implements `animateMove(svgRoot, preFacelet, move, durationMs)`:
  - Looks up the 9 affected sticker positions from a derived "move → net-grid coordinates" table (computed inline from `FACELET_MOVES` + `FACE_OFFSETS` + `FACE_GRID` — same data used in `web/src/state/faceletMoves.ts`).
  - Splits affected stickers into **face-rotation** (the 9 stickers on the turned face — rotate as a `<g>` group around face center) and **edge-cycle** (12 stickers along adjacent-face edges — translate from source position to destination position).
  - For edge-cycle stickers whose source-or-destination crosses the net's right boundary (any move involving B as adjacent — `R`, `R'`, `L`, `L'`, `U`, `U'`, `D`, `D'` all touch B's edge in the 4×3 cross net), apply **cross-fade**: source fades 1→0 while destination fades 0→1 over the duration.
  - 600 ms duration, ease-out timing curve (`t` → `1 - (1-t)²`).
- Renders ~6 panels in a grid: `solved + R`, `solved + R'`, `solved + B`, `solved + B'` (the wrap-stress cases), `mid-scramble + L`, `mid-scramble + U`. Both light + dark theme rows.
- "Play all" button + per-panel "replay" buttons.

**Eyeball gate.** User opens `web/preview/flat-cube-animated.html`, eyeballs the cross-fade on B-class moves. If it reads as broken, fall back to ghost-sticker dual render (P0 stays open until acceptable). Gate must pass before P1.

### P1 — `FlatCubeRenderer` learns animation

Port the P0 preview math into the production component.

- Extract pure animation math into `web/src/components/cubeAnimation.ts`:
  - `getMoveStickerKinematics(move: MoveStr): { faceRotation: { stickerIndices: number[]; centerX: number; centerY: number; angleDeg: number }; edgeCycle: { sourceIdx: number; destIdx: number; isWrap: boolean }[] }`
  - Pure function over the layout tables. Easy to vitest-snapshot per move.
- Extend `FlatCubeRenderer.tsx`:
  - New optional prop `animation?: { move: MoveStr; preFacelet: string; durationMs: number; nonce: number } | null` (default `null`).
  - When `null`, current static render path (no behavior change).
  - When set, render the **pre-move** facelet, then run a `requestAnimationFrame` loop applying transforms / opacities to the affected stickers per `cubeAnimation.ts`. On animation end, settle on the post-move state (the existing `facelet` prop is the ground truth).
  - `nonce` bump triggers re-animate (key for click-to-replay; `useEffect` dep).
- Vitest contract:
  - `cubeAnimation.test.ts` — per-move snapshots of `getMoveStickerKinematics` output (12 moves × kinematics shape).
  - `FlatCubeRenderer.test.ts` — animation-prop smoke (renders without crashing); animation-null preserves the existing 8-test contract verbatim; `nonce` change triggers re-render.
- No SolutionCard wiring yet. Component animation-ready but unused in production.

### P2 — `TwistyPlayerWrapper` learns "play once + replay"

- Add a third mode `mode="play-once"` to the wrapper.
  - Config: `setupAlg = preMoveAlg` (caller computes; `scrambleAlg + solution[:N-1].join(" ")`), `alg = move`, `experimentalSetupAnchor = "start"`, `tempoScale = 1`, `controlPanel = "none"`, `pointerEvents: none`.
  - On mount, the player renders at setup-end (= pre-move state) and plays through `alg` once. Settles at post-move.
- Expose a `replay()` method via `forwardRef` + `useImperativeHandle`:
  - `replay()` resets `player.timestamp` to start-of-alg and calls `player.play()`.
- `buildTwistyPlayerConfig` in `twistyPlayerConfig.ts` extended with the new mode branch.
- Vitest:
  - `twistyPlayerConfig.test.ts` — config snapshot for `play-once` mode (5 → 6 cases).
  - `TwistyPlayerWrapper.test.tsx` — smoke test that ref exposes `replay`.
- Builds cleanly; `.lazy` chunk shape unchanged.

### P3 — `SolutionCard` wires both renderers

- `SolutionGrid` already memoizes per-step `states` (state-after-step-N). Compute `preFacelet` per step card as `states[stepNum - 1]` and pass to the card. The start card (stepNum=0) has no move and skips animation.
- `SolutionCard` props gain: `preFacelet?: string` (undefined for start card), `move?: MoveStr` (the moveLabel typed as the literal move).
- New `useEffect` in `SolutionCard`:
  - Install `IntersectionObserver` on the card's outer element. First `isIntersecting` event → fire animation on both 2D and 3D (state both renderers receive the `animation` prop / `play-once` mode + an incremented nonce).
  - Click handler increments nonce → re-fires animation.
  - For `renderMode === "dual"`, both halves of the pair animate.
  - For start card, no observer, no click-replay — pure static render as today.
- For renderers in `static`/`net` mode (no animation prop, no play-once mode), the new wiring is a no-op. Mode switch already gates this.
- Performance note: animation triggers once per card per visibility transition. 20 cards × one-shot animations = trivial; no continuous RAF.

### P4 — e2e refresh + verification sweep

- Existing visibility assertions in `web/e2e/solution-grid.spec.ts` stay green — animation doesn't change the DOM contract (wrapper testids still present; rect counts still 54 in net mode after settle).
- Add one **deterministic** assertion: after triggering scroll-into-view + waiting for animation duration + 50 ms buffer, the post-animation DOM state matches the pre-animation static contract (54 rects with correct `data-color` for the post-move facelet). Avoids wall-clock-fragile mid-animation assertions.
- Skip click-replay e2e — requires animation in-flight detection which is brittle in Playwright. Manual eyeball + vitest covers it.
- Standard verification:
  - `pnpm --prefix web test` (vitest)
  - `pnpm --prefix web lint`
  - `pnpm --prefix web build`
  - `pnpm --prefix web test:e2e` (stub) + `PLAYWRIGHT_REAL_BACKEND=1 pnpm --prefix web test:e2e`
  - `uv run pytest` (no Python touched, but confirms repo health)
  - `uv run ruff check src/ tests/ scripts/codegen/`
  - **Eyeball check** — run dev server, scramble + solve, watch the cards animate as user scrolls. Verify wrap-stress moves (R/L/U/D — anything touching B) read cleanly.

## Critical files

| Phase | Action | Path |
|---|---|---|
| P0 | new | `web/preview/flat-cube-animated.html` |
| P1 | new | `web/src/components/cubeAnimation.ts` |
| P1 | new | `web/src/components/cubeAnimation.test.ts` |
| P1 | edit | `web/src/components/FlatCubeRenderer.tsx` |
| P1 | edit | `web/src/components/FlatCubeRenderer.test.ts` |
| P2 | edit | `web/src/components/TwistyPlayerWrapper.tsx` |
| P2 | edit | `web/src/components/twistyPlayerConfig.ts` |
| P2 | edit | `web/src/components/TwistyPlayerWrapper.test.tsx` |
| P3 | edit | `web/src/components/SolutionGrid.tsx` |
| P3 | edit | `web/src/components/SolutionCard.tsx` |
| P4 | edit | `web/e2e/solution-grid.spec.ts` |

## Reused, don't rebuild

- `FACELET_MOVES` (`web/src/state/faceletMoves.ts:12-25`) — 12 QTM moves × 54-index permutation. Source of truth for "which stickers move where."
- `FACE_OFFSETS` + `FACE_GRID` (`web/src/components/cubePalette.ts:13-20` + `FlatCubeRenderer.tsx:19-26`) — net-grid layout. `cubeAnimation.ts` computes net-grid coordinates from these tables.
- `applyMove` (`web/src/state/applyMove.ts`) — already used by SolutionGrid to memoize per-step facelets. `preFacelet[N] = states[N-1]` is free.
- `TwistyPlayer` imperative API (`player.timestamp`, `player.play()`) — public on the cubing.js export. Just hadn't been surfaced in the wrapper.
- `IntersectionObserver` — browser-native; no library.

## Out of scope (block 1C)

- **Cross-surface sync.** Sections ii/iii/iv share `activeStepIdx`. Click MovesGrid cell → highlights card + scrubs section iv player. Section iv playing → sweeps active highlight through ii and iii in lockstep. MovesGrid cells become interactive (currently display-only).
- **True single-RAF shared clock.** `TwistyPlayerWrapper` exposes only `replay()` in 1B; 1C will need finer timeline control.
- **Section iv per-move-step coupling.** Section iv keeps cubing.js's default `bottom-row` controls in 1B; M9 backlog item (twisty bottom-row restyle) and 1C cross-surface sync will replace them together.
- **`RenderMode = "net" | "iso" | "dual"` rename to `"3d"`.** Still deferred — bundle with 1C.
- **`React.lazy` revisit.** No bundle-shape change in this block.

## Risks

- **2D wrap looks bad despite cross-fade.** Mitigation: P0 eyeball gate; if cross-fade fails, P0 stays open and we add ghost-sticker dual render before P1. Plan does not pre-commit P1 to the cross-fade path.
- **Animation hammers performance with 20+ cards in dual mode.** Each dual card runs 2D + 3D simultaneously; 20 cards = 40 one-shot animations. Mitigation: IntersectionObserver gates triggers (only visible cards animate); animations are single-shot, not continuous. If real measurement shows trouble, stagger by `idx * 30 ms` for a cascade effect.
- **Cubing.js imperative replay doesn't behave.** `player.timestamp = 0; player.play()` may not reliably reset on all versions. Fallback: re-instantiate the player on replay (key bump on the wrapper). Costs a flicker frame but is bulletproof.
- **`IntersectionObserver` fires too eagerly on initial render.** All cards above the fold mount with `isIntersecting: true` immediately → all animate at once. Fix in P3: gate first-fire on `useRef` flag flipped after first render commit, or accept the cascade as a feature (everything animates as solve completes).

## Verification (end-to-end)

After P4 close, the full user flow:

1. `pnpm --prefix web dev` (or just `python -m http.server` on the preview during P0)
2. Scramble (e.g., length 14)
3. Click solve
4. Cards render → each visible step card animates its move once → static state held until click
5. Scroll cards below the fold into view → they animate as they enter
6. Click any settled card → that card's animation replays (both 2D and 3D in dual mode)
7. Section iv "watch the solve" still works as before (its own internal play controls; no behavior change)
8. Switch render mode 2D / 3D / split — animations follow the active mode

Tests green: vitest 53 → ~70 (P1 + P2 add cases); e2e stub + real-backend unchanged; build clean; lint + ruff clean.

## Commits (expected sequence)

1. `<plan>` — commits this plan to `plans/m9-per-card-animations.md`.
2. `<open-block>` — opens the LOG block on the new branch.
3. `<P0>` — `web/preview/flat-cube-animated.html`.
4. `<P1>` — `cubeAnimation.ts` + `FlatCubeRenderer` extension + tests.
5. `<P2>` — `TwistyPlayerWrapper` play-once + replay ref + tests.
6. `<P3>` — `SolutionCard` IntersectionObserver + click handler + props plumbing.
7. `<P4>` — e2e refresh.
8. `<close>` — close the LOG block.
