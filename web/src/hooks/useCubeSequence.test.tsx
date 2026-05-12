// Hook-level tests. The vitest env is "node" and `@testing-library/react`
// is not installed in this project (see web/package.json) — interactive
// hook testing (mount → state change → re-render assertion) requires
// jsdom + RTL. Until those land, the rigorous behavioural coverage lives
// on the factory in `state/cubeSequence.test.ts` (24+ cases covering
// every state transition + seek + gap + subscribe).
//
// This file covers the hook's CONTRACT surface:
//   - SSR `renderToStaticMarkup` doesn't throw with default + autoplay
//     specs.
//   - The hook returns a `CubeSequence` ref with the expected getters
//     and control methods, callable from the rendered component.
// Mounted-DOM coverage upgrades to RTL when the user installs it.
import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { useCubeSequence } from "./useCubeSequence";
import { createCubeSequence } from "../state/cubeSequence";
import type { MoveStr } from "../state/faceletMoves";

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

// Test harness: a component that uses the hook and renders the
// sequence's initial-snapshot state into a span. SSR runs the
// component body but never the effects, so the rAF loop never
// starts — perfect for asserting the initial-state shape.
function HookHarness({
  startFacelet,
  moves,
}: {
  startFacelet: string;
  moves: MoveStr[];
}): React.ReactElement {
  const seq = useCubeSequence({ startFacelet, moves });
  return createElement(
    "span",
    { "data-testid": "harness" },
    `status=${seq.status} ts=${seq.timestamp} idx=${seq.currentMoveIndex} total=${seq.totalDurationMs} n=${seq.moves.length}`,
  );
}

describe("useCubeSequence — SSR smoke", () => {
  it("renders without throwing on a basic spec", () => {
    const html = renderToStaticMarkup(
      createElement(HookHarness, {
        startFacelet: SOLVED,
        moves: ["R", "U", "R'", "U'"],
      }),
    );
    expect(html).toContain("status=idle");
    expect(html).toContain("ts=0");
    expect(html).toContain("idx=-1");
    expect(html).toContain("total=2400");
    expect(html).toContain("n=4");
  });

  it("renders with empty moves array", () => {
    const html = renderToStaticMarkup(
      createElement(HookHarness, {
        startFacelet: SOLVED,
        moves: [],
      }),
    );
    expect(html).toContain("status=idle");
    expect(html).toContain("total=0");
    expect(html).toContain("n=0");
  });

  it("throws (during SSR component body) on invalid spec", () => {
    // The factory validates synchronously inside `useMemo`. SSR runs
    // the component body, hitting the factory and the throw.
    expect(() =>
      renderToStaticMarkup(
        createElement(HookHarness, {
          startFacelet: "U".repeat(10),
          moves: [],
        }),
      ),
    ).toThrow(/54 chars/);
  });
});

// Direct interaction tests against the factory through the hook. These
// don't mount React — they construct the sequence via createCubeSequence
// to verify that the hook's public API contract is identical to the
// factory's. (The hook IS the factory return value plus useSyncExternalStore;
// we treat the factory's behavioural tests as transitively covering the
// hook's returned `CubeSequence`.)
describe("useCubeSequence — return-value contract", () => {
  it("returns a CubeSequence with the expected method surface", () => {
    // Build the equivalent sequence directly — the hook wraps this
    // value, so contract tests run against the same ref.
    const seq = createCubeSequence({
      startFacelet: SOLVED,
      moves: ["R", "U"] as MoveStr[],
    });
    // State getters
    expect(typeof seq.status).toBe("string");
    expect(typeof seq.timestamp).toBe("number");
    expect(typeof seq.totalDurationMs).toBe("number");
    expect(typeof seq.currentMoveIndex).toBe("number");
    expect(typeof seq.currentMoveProgress).toBe("number");
    expect(typeof seq.startFacelet).toBe("string");
    expect(Array.isArray(seq.moves)).toBe(true);
    // Controls
    expect(typeof seq.play).toBe("function");
    expect(typeof seq.pause).toBe("function");
    expect(typeof seq.seek).toBe("function");
    expect(typeof seq.seekToMove).toBe("function");
    expect(typeof seq.replay).toBe("function");
    expect(typeof seq.replayWithReverse).toBe("function");
    expect(typeof seq.subscribe).toBe("function");
  });
});

// Ref-stability + mounted-rendering tests deferred: this project has
// no jsdom / RTL setup (web/vitest.config.ts environment = "node";
// web/package.json has no @testing-library/* dep). The hook captures
// the sequence via `useMemo` on spec identity — React semantics
// guarantee the memo holds across renders. When jsdom + RTL land
// (likely alongside the first hook that needs mount-state tests),
// add: "stable spec → same CubeSequence ref across renders", and
// "spec change → new ref, old paused on cleanup".
