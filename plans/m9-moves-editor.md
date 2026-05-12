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

Three blocks proposed; each = one LOG block on its own branch, atomic commits, eyeball-gated at close. Open in plan-phase before execution.

### Block A — State-model surgery: moves becomes source of truth

The central change. Touches `App.tsx` + `SolutionGrid.tsx` + new `MovesEditor.tsx` + tests.

- **A·P1 — Refactor state model.** `moves` becomes editable App-level state. `solve` writes to `moves` instead of a separate cards-driving state. `SolutionGrid` consumes `moves` directly. Existing scramble flow updated to reset `moves` to empty. **No new UI yet** — internal refactor that should produce the same observable behavior at this commit.
- **A·P2 — `MovesEditor` component.** New text input bound to `moves`. As-you-type parser; live cards update. Inline error surface for invalid tokens. Location TBD — see **Open questions** below.
- **A·P3 — Tests.** Vitest for parser, MovesEditor smoke; e2e for type-in-field → cards-update flow.

### Block B — Auto-scramble + auto-solve

Smaller block. Falls out cleanly after Block A's state-model surgery is in place.

- **B·P1 — Auto-scramble on mount.** App-level effect: on first mount with no URL state, call `/api/scramble`.
- **B·P2 — Auto-solve toggle.** Setting (default on? off? UI for it?). After scramble lands, if enabled, fire solve.
- **B·P3 — Tests + UX polish.** Loading states. Race-condition handling (user clicks scramble during auto-solve, etc.).

### Block C — Polish + "make it fun"

Falls out as edge cases surface from A+B.

- **C·candidates** — animation interaction with mid-edit moves (does typing a move animate the new card? probably yes, IO won't fire since the card was already in view — need to .replay() on append?). Clear-moves button. Keyboard shortcuts (Cmd-K to focus moves field? Enter-to-solve?). Copy/paste UX. Sharable state URL (serializes scramble + moves into URL params). Mobile responsiveness checkpoint.
- Pick which subset to land based on what the A+B eyeballs surface.

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
