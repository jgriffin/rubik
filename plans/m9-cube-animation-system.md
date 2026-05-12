# M9.2 — Cube animation system

## Context

Block 1A landed the three render modes (net / iso-now-twisty / split) plus a section iv full-solve player that uses cubing.js's stock playback chrome. The palette-unification drive-by aligned 2D and 3D colors. Block 1B opened to add per-card move animations, paused after four P0 preview revisions failed to converge on a 2D visual model. Block 1B' (2026-05-11) ran a discussion-first exploration and locked the visual model in `web/preview/flat-cube-animated.html` rev5: ribbon-extension slides + face/ring rotations, secondary-opposite for R/L, F/B as ring rotation, data-first direction derivation, default-clipped to the cross silhouette with debug-toggleable extensions.

This plan rescopes beyond per-card single-move animation. The new vision: **one configurable animation system** that walks any `(startState, moveSequence)` through animated playback, with the controller as single source of truth for the clock and renderers as pluggable consumers. Two production surfaces fall out of the same primitive:

1. **Per-card single-move** (the original 1B goal): each step card animates its one move on scroll-into-view, click-to-replay.
2. **Full-solve dual-cube** (the new vision): one big 2D cross + one 3D cube walk the entire solution side-by-side, driven by an external play/pause/scrub bar that *we* own. cubing.js's stock playback chrome retires from that view — we use the twisty-player only as a passive 3D renderer with `controlPanel="none"` and externally-driven `timestamp`. If cubing.js fights external clock control, the 3D path is replaceable; the controller doesn't care.

## Architecture

Three layers. The controller is the contract; renderers are interchangeable.

### 1. `cubeNetAnimations.ts` — pure math (rev5 model)

`getRenderInstructions(preFacelet: string, move: MoveStr, progress: number): RenderPlan` returns the render plan for a single move at a single timestamp. The plan describes:

- ribbon-slide groups with offset vectors (primary + secondary for R/L; secondary direction is **opposite** primary by physical wrap-around)
- face-rotation transform: angle (degrees) + pivot (face geometric center)
- F/B ring-rotation transform: angle + pivot at F-center (for F/B edge cycles, which form closed rings around F-center)
- per-group clip-path-applied flag (cross silhouette by default) + extension-visibility flag (debug toggle)
- inverse-opacity overlay state (when extensions visible)
- post-progress=1 facelet must equal `applyMove(preFacelet, move)` — property-testable identity

Slide directions derived **from data** (mode-of-motion-vector across edge stickers), not from a hand-authored table — the rev5 lesson was that hand-authored direction tables silently disagree with the physical wrap topology.

### 2. `useCubeSequence` — controller hook

```ts
function useCubeSequence(spec: {
  startFacelet: string;
  moves: MoveStr[];
  msPerMove?: number;     // default 600
  gapMs?: number;         // default 0 (back-to-back); set ~100 for readable per-move beats
  autoplay?: boolean;     // default false
}): CubeSequence;

interface CubeSequence {
  // state (re-renders on change)
  status: "idle" | "playing" | "paused" | "ended";
  timestamp: number;
  totalDurationMs: number;
  currentMoveIndex: number;     // -1 before first move starts; moves.length-1 at end
  currentMoveProgress: number;  // 0..1 within current move

  // controls
  play(): void;
  pause(): void;
  seek(ms: number): void;
  seekToMove(i: number): void;
  replay(): void;             // seek(0) + play()

  // read-only sequence info
  readonly startFacelet: string;
  readonly moves: MoveStr[];
}
```

Internally drives a `requestAnimationFrame` loop while `status === "playing"`. State machine: `idle → playing → paused | ended`; `play()` from `ended` is equivalent to `replay()`. Snapshot-testable via Vitest fake timers.

### 3. Renderers — consume the sequence handle

- **`<NetAnimatedCube sequence={cubeSeq} />`** — SVG, applies `cubeNetAnimations.ts` instructions at the controller's `currentMoveIndex` + `currentMoveProgress`. Composed of `<NetCubeRenderer/>` (the existing static facelet renderer, renamed from `FlatCubeRenderer` per 1B' close) for the non-animated layers, plus animated `<g>` groups for ribbons + face/ring rotations.
- **`<TwistyAnimatedCube sequence={cubeSeq} />`** — wraps `<twisty-player>` with `controlPanel="none"`, `tempoScale=1`, `experimentalSetupAnchor="start"`, `setupAlg = (scrambleAlg + priorMoves)`, `alg = sequence.moves.join(" ")`. Binds `player.timestamp = sequence.timestamp` in a useEffect. If `player.play()` and our rAF clock fight, we drive *only* via direct timestamp writes (no `.play()`).
- **`<DualAnimatedCube sequence={cubeSeq} />`** — composes both side-by-side, one controller.

### Why this shape

- **Single source of truth for time.** All renderers read the same `sequence.timestamp`. No internal clocks fight. Sub-frame drift between 2D and 3D becomes a function of `useEffect` timing only, not of separate animation loops.
- **External controller is the commitment, not the renderer.** We can swap cubing.js for a different 3D lib if needed — the controller contract doesn't change. The user explicitly wants this property.
- **Per-card and full-solve use the same primitive.** A card uses `moves: [card.move]`. The full-solve viewer uses `moves: solution`. The controller doesn't know the difference.
- **Cross-surface scrubber sync (the old "block 1C" goal) is free.** Hover/click on a section ii/iii card calls `globalSequence.seekToMove(card.stepNum - 1)`. Same controller, no extra plumbing.

## Branch

`m9-cube-animation-system` off main HEAD `c2eabf2` (Block 1B' merge).

## Blocks

This milestone breaks into four LOG blocks, each on its own branch sharing the parent. One coherent goal per block; eyeball gate between blocks.

### Block A — Math + controller + 2D renderer (no production wiring)

Output: `cubeNetAnimations.ts` + `useCubeSequence` + `NetAnimatedCube` + `NetCubeRenderer` (renamed), gated against a controller-driven extension of `flat-cube-animated.html`. **No SolutionCard wiring yet.** The whole system works end-to-end *in the preview*; production cards still render the old static path.

Phases (each = one atomic commit, app builds + tests pass throughout):

- **A·P0 — Preview substrate refactor.** Extend `web/preview/flat-cube-animated.html` with a "sequence widget" up top: `startFacelet` input + `moves` input (e.g. `"R U R' U'"`) + play/pause/replay/seek controls + a single big cross rendering the live sequence. Per-move-panel grid (12 QTM from solved) preserved below as a unit-test substrate. Standalone (inline JS, no React), but written to mirror the controller/math API shape so the port to TS modules is structural-only.
- **A·P1 — `web/src/components/cubeNetAnimations.ts`.** Pure functions + types. Per-move kinematics derived data-first. Vitest: 12-move kinematics snapshots + property tests asserting post-progress=1 facelet matches `applyMove`.
- **A·P2 — `web/src/hooks/useCubeSequence.ts`.** Controller implementation. rAF loop, state machine, seek logic. Vitest under fake timers: status transitions, replay-from-end correctness, seek mid-move boundary correctness.
- **A·P3 — `NetCubeRenderer.tsx` (rename from `FlatCubeRenderer.tsx`) + `NetAnimatedCube.tsx`.** Single-commit rename touches `SolutionCard.tsx`, tests, preview imports. Then add `NetAnimatedCube` consuming a `CubeSequence` handle. Vitest: animation smoke (renders without crashing across all 12 moves); static-mode behavior unchanged (existing test contract preserved verbatim).

**Eyeball gate (block-close).** User opens `web/preview/flat-cube-animated.html`, types a move sequence into the widget, hits play, watches the 2D cross walk through the sequence with the rev5 visual model. Matches the per-move panels below. Eyeball passes → Block A closes.

### Block B — 3D renderer on the same controller

Output: `TwistyAnimatedCube` that drives cubing.js's twisty-player from the controller, validated in the same preview now showing 2D + 3D side-by-side.

- **B·P4 — `web/src/components/TwistyAnimatedCube.tsx`.** Wraps `<twisty-player>` with `controlPanel="none"`, `tempoScale=1`. Binds `player.timestamp` from `sequence.timestamp`. Surfaces the cubing.js external-clock fight (if any) and resolves it: ideally pure timestamp writes work; if not, fall back to a minimal `.play()`/`.pause()` mirror of our controller's `status`; if cubing.js fights both paths, scope a pivot here.
- **B·P0' — Preview extends to dual.** The sequence widget now drives both renderers side-by-side; user eyeballs lockstep.
- **B·P4-tests — Vitest:** `TwistyAnimatedCube.test.tsx` smoke + an integration test that asserts the wrapper sets `player.timestamp` in response to controller state changes.

**Eyeball gate.** 2D and 3D walk the same sequence in visible lockstep. Sub-frame drift acceptable.

### Block C — Per-card wiring

Output: production `SolutionCard` animates each card's single move on scroll-into-view, click-replays. Both net and twisty modes wired through the new components.

- **C·P5 — `SolutionGrid` + `SolutionCard` rewire.** Each non-start card constructs `useCubeSequence({ startFacelet: states[stepNum-1], moves: [card.move], msPerMove: 600 })`. `IntersectionObserver` fires `.replay()` on first visibility; click handler also fires `.replay()`. Both net (`NetAnimatedCube`) and twisty (`TwistyAnimatedCube`) modes use the new components. Start card (stepNum=0) skips the hook entirely — pure static `NetCubeRenderer`.
- **C·P5-tests — Vitest:** `SolutionCard.test.tsx` observer wiring smoke + click handler + start-card-skips-animation.
- **C·e2e refresh.** `solution-grid.spec.ts` adds a deterministic post-settle assertion: 54 rects with correct `data-color` after `msPerMove + 50ms` buffer. Mid-animation assertions skipped (intentionally non-deterministic). Existing visibility tests stay green.

### Block D — Full-solve dual-cube view (section iv replacement)

Output: section iv shows ONE big 2D cross + ONE 3D cube + a custom playback bar, walking the entire solution from scramble to solved. Retires the cubing.js-controlled player from 1A. Cross-surface scrubber sync (was the old "block 1C") folds in here.

- **D·P6 — `FullSolveViewer.tsx` + `PlaybackBar.tsx`.** Single `useCubeSequence` over the full solution (`startFacelet: scramble`, `moves: solution`). `DualAnimatedCube` renders both surfaces. Custom playback bar: play/pause button + scrubber + move-counter "12 / 42" + per-move tick marks. Styled to match the editorial-paper aesthetic. ROADMAP backlog entry "section iv twisty-player bottom-row restyle" retires when this lands.
- **D·P7 — Cross-surface sync.** App lifts the section iv `useCubeSequence` to a top-level context; section ii/iii cards subscribe. Hover/click on a card → `globalSeq.seekToMove(card.stepNum - 1)`. Section iv's playback head reflects card focus; vice versa, scrubbing in section iv highlights the active card in sections ii/iii.
- **D·e2e.** Scrubbing through the full solve renders the right state at each move boundary; cross-surface highlight sync tested.

## What carries forward from rev5

- **Timing constants** (named in the preview): `DUR_FORWARD_MS=600`, `DUR_REVERSE_MS=150`, `DUR_PAUSE_MS=250` (the auto-reset beat). Carried into `cubeNetAnimations.ts` as defaults.
- **Direction-data-first principle.** No hand-authored direction tables. Derive from the move's effect on facelet positions.
- **Production-clipped by default + debug "Show extensions" toggle.** Same default in `NetAnimatedCube`; the toggle becomes a component prop for the preview, hidden in production.
- **F/B = ring rotation, not slide.** Encoded in `cubeNetAnimations.ts`'s move-classification table (face turn / linear edges / ring edges).
- **Z-order: static + ribbons UNDER overlay; face + ring rotations ABOVE.** Encoded in the SVG group ordering inside `NetAnimatedCube`.

## Open questions resolved later

- **cubing.js external-clock fight (Block B).** Plausibly tractable via direct `player.timestamp` writes + `controlPanel="none"` + `tempoScale=0` (we don't call `.play()`; cubing.js renders whatever timestamp we set). If `.play()`/timestamp fight, we just don't call `.play()`. If the renderer doesn't update on timestamp changes without `.play()`, we have a real problem and might need to swap the 3D lib. Risk surfaces in Block B; not pre-emptively designed for.
- **Per-card hooks at scale (Block C).** 20+ cards each running their own `useCubeSequence` hook. Each hook only runs an rAF loop while playing, and per-card animations are one-shot (~600 ms). Should be cheap; profile if visible.
- **Card-vs-full-solve clock interaction (Block D).** A card animating its own move while section iv is scrubbing — independent clocks (cards keep local sequences) or unified (cards subscribe to section iv's clock and ignore their own move)? Decide at C/D boundary based on what feels right; both are tractable.

## What this plan supersedes

- The original `plans/m9-per-card-animations.md` (P0–P4 against the rev1–rev4 conveyor-source-projection model that 1B' replaced). Filename renamed to reflect expanded scope.
- ROADMAP backlog entry "M9 backlog: section iv twisty-player bottom-row restyle" (subsumed by Block D's playback bar).
- The original "block 1C" notion of cross-surface sync as a separate milestone — folded into Block D since the shared-controller architecture makes it nearly free.
