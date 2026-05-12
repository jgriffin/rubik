// SolutionCard tests — covers the discriminated-union split between
// start and non-start cards, and the SSR-observable render contract
// the e2e suite keys on (data-pos, data-color, data-testid).
//
// Testing strategy: `renderToStaticMarkup` (SSR-only). This project
// has no @testing-library/react / jsdom (see web/package.json), so
// mounted-DOM tests (click handlers firing, IntersectionObserver
// firing, state-driven re-renders) are deferred. They're flagged
// with `it.skip` and a comment pointing at the same
// "queued for RTL upgrade" pattern documented in
// useCubeSequence.test.tsx.
//
// What SSR DOES cover for non-start cards:
//   - useCubeSequence runs inside the component body (SSR executes
//     function bodies, just not effects). The sequence is constructed
//     at currentMoveIndex=-1 (idle, ts=0), so Cube2D's animated path
//     falls through to StaticInner — emitting the same data-pos /
//     data-color attributes as the start card's static path. This is
//     the invariant the e2e selectors depend on, and it MUST hold for
//     non-start cards in their pre-animation state.
//   - useEffect never fires under SSR, so the IntersectionObserver
//     branch is never reached. Good — we don't want test envs that
//     happen to define IO globally to trigger replay() under SSR.

import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import SolutionCard from "./SolutionCard";
import { applyMove } from "../state/applyMove";
import type { MoveStr } from "../state/faceletMoves";

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

// ---------------------------------------------------------------------
// Start card — static path
// ---------------------------------------------------------------------

describe("SolutionCard — start card (isStart=true)", () => {
  it("renders Cube2D static path in net mode (data-pos + data-color attributes)", () => {
    const html = renderToStaticMarkup(
      createElement(SolutionCard, {
        stepNum: 0,
        moveLabel: null,
        facelet: SOLVED,
        scrambleAlg: "",
        moves: [],
        sizePx: 130,
        renderMode: "net",
        isStart: true,
        isActive: false,
        onClick: () => {},
      }),
    );
    // E2e contract — all 54 sticker positions + their colors emitted.
    for (let i = 0; i < 54; i++) {
      expect(html).toContain(`data-pos="${i}"`);
    }
    // Static SVG has viewBox "0 0 W H" — not the animated envelope's
    // negative origin. Confirms we're on the static path, not the
    // animated-fallthrough path.
    expect(html).toMatch(/viewBox="0 0 [\d.]+ [\d.]+"/);
    expect(html).not.toContain("data-layer=");
  });

  it("data-testid is sol-card-0 and shows 'start' label", () => {
    const html = renderToStaticMarkup(
      createElement(SolutionCard, {
        stepNum: 0,
        moveLabel: null,
        facelet: SOLVED,
        scrambleAlg: "",
        moves: [],
        sizePx: 130,
        renderMode: "net",
        isStart: true,
        isActive: false,
        onClick: () => {},
      }),
    );
    expect(html).toContain('data-testid="sol-card-0"');
    expect(html).toContain("start");
  });
});

// ---------------------------------------------------------------------
// Non-start card — animated path falling through to StaticInner
// ---------------------------------------------------------------------

describe("SolutionCard — non-start card (isStart=false)", () => {
  it("renders Cube2D animated path that falls through to StaticInner at idx=-1", () => {
    // preFacelet = SOLVED so the pre-animation render emits SOLVED's
    // letters. The animated branch hits AnimatedInner, sees
    // currentMoveIndex=-1 (idle, ts=0), and renders StaticInner with
    // the start facelet — same data-pos/data-color contract.
    const html = renderToStaticMarkup(
      createElement(SolutionCard, {
        stepNum: 1,
        moveLabel: "R",
        facelet: applyMove(SOLVED, "R"),
        preFacelet: SOLVED,
        move: "R" as MoveStr,
        scrambleAlg: "",
        moves: ["R"],
        sizePx: 130,
        renderMode: "net",
        isStart: false,
        isActive: false,
        onClick: () => {},
      }),
    );
    // All 54 stickers present (StaticInner contract held by fallthrough).
    for (let i = 0; i < 54; i++) {
      expect(html).toContain(`data-pos="${i}"`);
    }
    // Pre-animation: colors match preFacelet (SOLVED), not the
    // post-move facelet. U-center idx 4 is 'U'; R-center idx 13 is
    // 'R'; B-center idx 49 is 'B'.
    expect(html).toMatch(/data-pos="4"[^>]*data-color="U"/);
    expect(html).toMatch(/data-pos="13"[^>]*data-color="R"/);
    expect(html).toMatch(/data-pos="49"[^>]*data-color="B"/);
    // Verify we're on the fallthrough (no animated overlay layers).
    expect(html).not.toContain("data-layer=");
  });

  it("data-testid carries the step number", () => {
    const html = renderToStaticMarkup(
      createElement(SolutionCard, {
        stepNum: 3,
        moveLabel: "U'",
        facelet: SOLVED,
        preFacelet: SOLVED,
        move: "U'" as MoveStr,
        scrambleAlg: "",
        moves: ["R", "U", "U'"],
        sizePx: 130,
        renderMode: "net",
        isStart: false,
        isActive: false,
        onClick: () => {},
      }),
    );
    expect(html).toContain('data-testid="sol-card-3"');
  });
});

// ---------------------------------------------------------------------
// renderMode branches
// ---------------------------------------------------------------------

describe("SolutionCard — renderMode wiring (non-start card)", () => {
  it("dual mode: 2D half + 3D half (twisty)", () => {
    const html = renderToStaticMarkup(
      createElement(SolutionCard, {
        stepNum: 1,
        moveLabel: "R",
        facelet: applyMove(SOLVED, "R"),
        preFacelet: SOLVED,
        move: "R" as MoveStr,
        scrambleAlg: "",
        moves: ["R"],
        sizePx: 100,
        renderMode: "dual",
        isStart: false,
        isActive: false,
        onClick: () => {},
      }),
    );
    // 2D half is animated-fallthrough → emits data-pos / data-color.
    expect(html).toContain('data-testid="flat-cube-pair"');
    expect(html).toContain('data-pos="0"');
    // 3D half is a twisty wrapper div. TwistyPlayer is constructed in
    // useEffect → never under SSR → the wrapper div is the only thing
    // emitted. Its testid is twisty-cube-pair.
    expect(html).toContain('data-testid="twisty-cube-pair"');
    expect(html).toContain("render-pair");
  });

  it("iso mode: only twisty, no Cube2D rendered", () => {
    const html = renderToStaticMarkup(
      createElement(SolutionCard, {
        stepNum: 1,
        moveLabel: "R",
        facelet: applyMove(SOLVED, "R"),
        preFacelet: SOLVED,
        move: "R" as MoveStr,
        scrambleAlg: "",
        moves: ["R"],
        sizePx: 130,
        renderMode: "iso",
        isStart: false,
        isActive: false,
        onClick: () => {},
      }),
    );
    // Twisty wrapper div present.
    expect(html).toContain('data-testid="twisty-cube"');
    // No Cube2D — no data-pos attributes.
    expect(html).not.toContain("data-pos=");
  });

  it("start card iso mode: only twisty, no Cube2D rendered", () => {
    const html = renderToStaticMarkup(
      createElement(SolutionCard, {
        stepNum: 0,
        moveLabel: null,
        facelet: SOLVED,
        scrambleAlg: "",
        moves: [],
        sizePx: 130,
        renderMode: "iso",
        isStart: true,
        isActive: false,
        onClick: () => {},
      }),
    );
    expect(html).toContain('data-testid="twisty-cube"');
    expect(html).not.toContain("data-pos=");
  });
});

// ---------------------------------------------------------------------
// Pointer events + IntersectionObserver wiring — mount-mode tests deferred.
//
// renderToStaticMarkup produces static HTML — event handlers don't
// attach, useEffect never fires, IntersectionObserver never observes.
// Verifying the pointer-driven press-and-hold path, the keyboard-only
// click fallback, AND the IO one-shot behaviour all require jsdom +
// @testing-library/react. Queued for the RTL upgrade, same pattern as
// documented in useCubeSequence.test.tsx.
//
// Current interaction contract (factory-level coverage in
// state/cubeSequence.test.ts already verifies the state machine; these
// mounted tests would just confirm the wiring fires the methods):
//   - pointerdown → onActiveChange() + seq.replayWithReverseHold()
//     (reverse leg + indefinite held dwell at the pre-state).
//   - pointerup / pointercancel / pointerleave → seq.releaseHold()
//     (forward play; collapses to reverse-then-forward if released
//     mid-reverse).
//   - onClick fallback (keyboard activation only — Space/Enter on a
//     focused button fires a synthetic click but no pointer events):
//     onActiveChange() + seq.replayWithReverse() — the finite-pause
//     choreography for a snappy keyboard cue. A useRef flag set in
//     pointerdown / cleared on the trailing click dedupes against the
//     pointer path so mouse-driven clicks don't double-trigger.
//   - Mount → IO fires "isIntersecting" → seq.play() called once
//     (forward only — IO is the first-visibility cue), observer
//     disconnected (subsequent intersections are no-ops).
// ---------------------------------------------------------------------

describe.skip("SolutionCard — pointer events + IntersectionObserver (queued for RTL)", () => {
  it.skip("pointerdown fires onActiveChange + seq.replayWithReverseHold()", () => {
    // Queued: needs jsdom + RTL to dispatch PointerEvent on rendered
    // buttons and spy on the hook's returned CubeSequence.
  });

  it.skip("pointerup fires seq.releaseHold()", () => {
    // Queued.
  });

  it.skip("pointercancel and pointerleave both fire seq.releaseHold() (drag-off safety net)", () => {
    // Queued.
  });

  it.skip("keyboard activation (Space/Enter → synthetic click, no pointer events) routes to replayWithReverse()", () => {
    // Queued: needs jsdom keyboard-event simulation; the useRef flag
    // should remain false so the click branch runs.
  });

  it.skip("pointer-driven click does NOT double-trigger (useRef dedupe)", () => {
    // Queued: pointerdown→pointerup→click sequence; verify
    // replayWithReverse() is NOT called (only replayWithReverseHold +
    // releaseHold from the pointer path).
  });

  it.skip("IntersectionObserver fires seq.play() once on first visibility", () => {
    // Queued: needs jsdom + a stubbed IntersectionObserver to simulate
    // the entry, then assert play() called exactly once and the
    // observer disconnected.
  });
});
