# M9.3 — Interactive moves editor

## Context

M9.2 landed the cube animation system (Block A: foundation primitive; Block C: production wiring with press-and-hold replay). Block B (3D renderer animation) was deferred 2026-05-12 in favor of this milestone pivot — animation-on-cards is now visible, and the user signaled during Block C eyeball that the higher-value next direction is making the app feel like an interactive playground rather than a one-shot solver demo.

This plan recasts the original M9.3 ("Alt input + polish: paste notation, manual color entry, share URL, mobile") around a single load-bearing change: **moves become the editable source of truth**. The original M9.3 scope items (paste notation, color entry, share URL, mobile) fall out as flavor features the new editor surface enables naturally — paste notation = pasting into the moves field; share URL = serialize scramble+moves into the URL.

## Vision

The app should feel like an REPL / sandbox for cube moves. Refresh gives you a fresh scramble to work with. Auto-solve fills in a known-good solution. You can clear the moves and type your own (`R U R' U' F R F'`), and the cards below update *live* as you edit. Hit Solve again and the moves field re-populates. Animation system (already shipped) makes hand-typing a move feel rewarding — type `R`, watch the card animate.

User's framing during the Block C eyeball gate:

> *"if I start typing moves like R R' ... We should be adding cards on the bottom as we go along. So again, Solve adds a bunch of moves and because the moves got added, the cards get edited below. But we're also supposed to have the ability to edit the moves... this stuff should be a lot more dynamic and interactive. Make it fun."*

## Architecture (proposed; refine in plan-phase before any block opens)

Three layers. Roughly mirrors the M9.2 controller-renderer split but for state/UI:

### 1. Moves state (in `App.tsx` or a dedicated hook)

`moves: MoveStr[]` becomes the editable source of truth in App state. Today's flow:

```
scramble (button)  → server returns {state, moves}  → setState(scrambleState, scrambleMoves)
solve (button)     → server returns {moves}          → setSolution(moves)  // writes to a separate cards-driving state
```

New flow:

```
scramble (button)  → setMoves([])     + setScrambleState(state)
solve (button)     → setMoves(solveResponse.moves)
edit field         → setMoves(parsedMoves)
```

`SolutionGrid` derives cards from `moves` directly (it already computes per-step facelets via `applyMoves`; it just consumes a different state slice).

### 2. Moves text input component

New `MovesEditor` component. Single-line (or multi-line?) text input. As-you-type parser that converts text like `"R U R' U' F"` into a `MoveStr[]`. Validates: each token is a legal QTM move (12 base moves + prime). Surface parse errors inline ("unknown move `X`" at column N). Debounce or stream-parse — both fine; pick whichever feels right at the eyeball.

### 3. Auto-scramble + auto-solve

Page-load effect: if no scramble in state and no URL state, call `/api/scramble`. After scramble, if auto-solve enabled (default? toggle?), call `/api/solve`. Same effect path as the existing buttons, just fired automatically on mount.

## Blocks

Original three-block plan reshaped twice mid-milestone (Block A design pivot 2026-05-12, then post-Block-A re-scoping 2026-05-12). Current sequence below — each block = one LOG block on its own branch, atomic commits, eyeball-gated at close.

### Block A — Section ii becomes the editable source of truth ✅ done

[2026-05-12, branch `m9.3-block-a-moves-editor`, 10 work commits.] Section ii ("moves to apply") rewritten as the editable surface — per-cell `<input>` grid with auto-advance, paste-spread, backspace-rewind, and a cell-mode-vs-text-mode selection model (full-selection = keyboard-navigation mode, collapsed/partial = caret edit). Section iii renamed "steps" and derives its cards from `moves` (no longer "the solution"). Solve consolidated into a bottom-anchored label inside the trailing dashed cell; matched by a "solved" label in the same slot + appended to section iii once the cube reaches solved. Render-mode reshaped from a 3-button segmented control to two independent 2D/3D toggles (both on = split). CubeSizeSwitch promoted to the wordmark's serif register. Solve semantics: from current state + append (was: from start + replace). Mid-block design pivot recorded as a feedback memory ("edit the existing surface, don't add a parallel input"). See LOG 2026-05-12 for the full file inventory and per-phase commit list.

### Block B — Move-cell polish (cell-mode focus ring + solve as primary CTA) ✅ done

[2026-05-12, branch `m9.3-block-b-move-cell-polish`, 5 work commits.] Two pillars shipped: (a) cell-mode visual differentiation via `data-cell-mode` attribute + hidden `::selection` + inset focus ring on wrapper (`:has()`); first-click reliability fixed with an `onClick`-after-mouse-chain force-select that wins the race against the browser's mouseup-drag-end. (b) Solve label promoted to a primary CTA — Fraunces 16px / weight 600 / theme-orange with `transform: scale` + `filter: brightness` press feedback (the previous orange→ink hover flip dropped because a color flip on press misrepresents the action's semantics). Codified as a `.text-action` utility class for future text-styled action surfaces. Side fix: off-by-one in the cell→card active sync — cell N corresponds to step N+1 (the state AFTER move N), now consistent with `SolutionGrid`'s step coordinates. See LOG 2026-05-12 for the full file inventory and per-phase commit list.

### Block C — Cube mode: single-cube view + 2D per-step animation + auto-play ✅ done

[2026-05-13, branch `m9.3-block-c-cube-mode`, 6 work commits.] Cube-mode rendering shipped with 2D per-step animation on forward `activeIdx` jumps + auto-play that advances `activeIdx` at a 500ms cadence riding the per-step animations. New `CubeStage` component owns cube-mode rendering; `SolutionGrid` early-returns to it for `cols===1`. Section iii's selector restructured to `2D 3D | cube | 2 3 4 5 6` (the word "columns" dropped, "1" replaced with `cube`, "5" added). Section iv (the M9.2 "watch the solve" twisty-player strip) deleted — cube mode supersedes its job. **C·P4 (3D per-step animation via twisty-player's native playback) descoped** mid-block per user direction "right now, lets just focus on the 2d step animations"; deferred to a future block, not blocking the M9.3 milestone arc. **Design pivot at C·P5 close**: C·P5's play button placement on the cube card itself was reshaped at the would-be eyeball — user wants section ii to be THE navigation/transport surface for the trajectory, not the cube card. Block D (about to open) picks up the relocation. See LOG 2026-05-13 for the full file inventory, per-phase commits, and load-bearing decisions.

### Block D — Auto-scramble + sharable URL + Block-A cleanup deferrals (planned)

The original "Block B — auto-scramble + auto-solve" plus the items that fell out of Block A's close note:
- Auto-scramble on mount (no URL state) — `App.tsx` first-paint effect, ~10 lines.
- Sharable state URL (originally "Block C: sharable state URL"; serialize scramble + moves into URL params; restore on load).
- Drive-by: delete unused `SolveButton.tsx` (~20 lines).
- Reverse active-state sync: card click → cell focus (today only cell → card works one-way).
- Card animation on cell click (imperative hook into the per-card sequence's `replayWithReverse`).
- RTL test backfill for the edit-surface behaviors (auto-advance, paste-spread, cell-mode↔text-mode, Escape).

> Note 2026-05-13: a new block-D will be inserted ahead of this one when the next block opens — "section ii as navigation/transport: start cell + play migration + cube-card press-and-hold" — pushing this block to E (and the existing Block E to F). The renumbering happens at the next block's D·P0 opener, not here.

### Block E — Mobile + keyboard shortcuts (planned, M9.3-closing)

The original "Block C — polish + make it fun" residual. Mobile responsiveness checkpoint (the per-cell input grid + 3D view need a mobile pass). Keyboard shortcuts (Cmd-K to focus the first empty cell? Enter-to-solve? Shift-Enter for scramble?). Final UAT pass before milestone close.

## Open questions

### Where does the `MovesEditor` text input live in the layout?

Three plausible spots from the close-of-Block-C conversation:

1. **Above section iii (the cards section).** "Here are the moves; below are the cards that visualize each." Cleanest mental model — moves are the contract, cards are the visualization.
2. **Between section i (scramble) and section iii (solution).** Chronological flow: scramble → moves → cards.
3. **Replace the Solve button area entirely.** Solve button becomes a small icon next to the moves field. Moves field is always visible and always editable; clicking Solve just populates it.

User has not picked. **Resolve in plan-phase before opening Block A.**

### Auto-solve default: on or off?

User's words: *"once we add a scramble cube, I think we can automatically run the solve. Like a release case where we don't want to actually go solve the things or, I don't know, maybe we can have an auto-solve thing."* — they're undecided. Defaults:
- **On**: refresh → scramble + solve in one shot. Loop-of-default is what user is "tired of hitting the button" for.
- **Off** with toggle: more conservative; first-time visitors see the scramble but choose whether to solve.

Mobile gate: auto-solve on mobile if the device is slow could feel laggy. Worth a checkpoint.

### Parser scope

QTM only (matches the rest of the codebase) or include double-moves (`R2`, `U2`)? Today's solver is QTM-only; the moves editor accepting double-moves would let users hand-type things the solver can't solve. Either choice has a reasonable answer; resolve in plan-phase.

### URL state

Block C "sharable state URL" candidate — serialize what exactly? `?scramble=R+U+R%27` (just the scramble notation) → app derives state by applying to solved. `?state=<54-char-facelet>` (full state). Both have merits. Resolve when Block C scope lands.

## Dependencies / prerequisites

- M9.2 animation system shipped — required (this milestone leans on it for the "make it fun" loop).
- Existing `/api/scramble` and `/api/solve` endpoints — adequate. No backend changes anticipated for Blocks A or B.

## What this plan supersedes

- Original M9.3 line in `ROADMAP.md` ("Alt input + polish") — recast into this plan 2026-05-12.
- ROADMAP backlog entry "M9 backlog: interactive moves editor + auto-scramble + decoupled solve flow" — promoted into this milestone, backlog entry removed.

## Block B (M9.2 3D renderer animation) deferral

Decided 2026-05-12 during Block C close: pivot to M9.3 (this milestone) before completing M9.2 Block B (TwistyAnimatedCube). 2D animation is visible and valuable; 3D animation is incremental polish on a path with known unknowns (cubing.js external-clock fight, per the M9.2 plan). The M9.2 plan stays open; Block B revisitable as a standalone block whenever priorities allow. Block D (full-solve dual-cube view) likely follows whichever of {Block B, M9.3} doesn't ship first.
