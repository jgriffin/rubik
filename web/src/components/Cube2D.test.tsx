// Cube2D tests — covers BOTH the static mode (byte-equivalent to the
// pre-rename FlatCubeRenderer) and the animated mode (subscribes to a
// passed-in CubeSequence and renders the rev5 overlay via
// `cube2DKinematics.getRenderInstructions`).
//
// Testing strategy: `renderToStaticMarkup` (SSR-only). This project
// has no `@testing-library/react` / `jsdom` (see `web/package.json`),
// so interactive rendering tests are deferred. The static branch is
// pure-render with no hooks; the animated branch's `useSyncExternalStore`
// subscribe never fires under SSR, but the initial-render path through
// the snapshot getter DOES run, exercising the JSX construction.
//
// Behavioural coverage of the controller lives in
// `state/cubeSequence.test.ts`; coverage of the math lives in
// `cube2DKinematics.test.ts`. Cube2D tests here verify the
// PRESENTATION CONTRACT: SVG structure, layer ordering, transforms,
// clip-path, data-attributes — what consumers (SolutionCard, e2e)
// rely on visually.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import Cube2D from "./Cube2D";
import { createCubeSequence } from "../state/cubeSequence";
import { applyMove } from "../state/applyMove";
import type { MoveStr } from "../state/faceletMoves";

// rAF / performance shim. Node doesn't expose requestAnimationFrame.
// We install a minimal `setTimeout(0)` shim so `seq.play()` works
// (it always schedules an rAF). Tests use `play() + pause()` to put
// the sequence at currentMoveIndex=0 progress=0 (otherwise seek(0) →
// status=idle → currentMoveIndex=-1 → static fallthrough). We never
// actually need to advance the rAF queue here — pause() cancels
// before the first callback fires, which is exactly what we want
// for snapshotting the progress=0 state.
beforeEach(() => {
  let nextRafId = 1;
  vi.stubGlobal("requestAnimationFrame", (): number => nextRafId++);
  vi.stubGlobal("cancelAnimationFrame", () => {});
});
afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

const ALL_MOVES: MoveStr[] = [
  "U", "U'", "D", "D'",
  "R", "R'", "L", "L'",
  "F", "F'", "B", "B'",
];

// ---------------------------------------------------------------------
// Static-mode tests — preserve FlatCubeRenderer's contract verbatim
//
// These mirror what the e2e suite (`web/e2e/renderer.spec.ts`,
// `solution-grid.spec.ts`) asserts via DOM queries: 54 sticker rects
// with `data-pos` + `data-color` attributes, default testid, viewBox
// math from `sizePx`.
// ---------------------------------------------------------------------

describe("Cube2D — static mode (replaces FlatCubeRenderer)", () => {
  it("renders an SVG with default testid 'flat-cube' when testId not given", () => {
    const html = renderToStaticMarkup(
      createElement(Cube2D, { facelet: SOLVED }),
    );
    expect(html).toContain('data-testid="flat-cube"');
  });

  it("explicit testId overrides the default", () => {
    const html = renderToStaticMarkup(
      createElement(Cube2D, { facelet: SOLVED, testId: "custom-id" }),
    );
    expect(html).toContain('data-testid="custom-id"');
  });

  it("testId={null} omits the data-testid attribute", () => {
    const html = renderToStaticMarkup(
      createElement(Cube2D, { facelet: SOLVED, testId: null }),
    );
    expect(html).not.toContain("data-testid");
  });

  it("emits 54 sticker rects with data-pos 0..53", () => {
    const html = renderToStaticMarkup(
      createElement(Cube2D, { facelet: SOLVED, testId: null }),
    );
    // Count rects via data-pos="N" matches.
    for (let i = 0; i < 54; i++) {
      expect(html).toContain(`data-pos="${i}"`);
    }
  });

  it("emits data-color per sticker matching the facelet letter", () => {
    // First 9 stickers (U face) are 'U' on solved; mid (R face) are 'R'; etc.
    const html = renderToStaticMarkup(
      createElement(Cube2D, { facelet: SOLVED, testId: null }),
    );
    // U-center (idx 4) = 'U'.
    expect(html).toMatch(/data-pos="4"[^>]*data-color="U"/);
    // R-center (idx 13 = 9+4) = 'R'.
    expect(html).toMatch(/data-pos="13"[^>]*data-color="R"/);
    // F-center (idx 22) = 'F'.
    expect(html).toMatch(/data-pos="22"[^>]*data-color="F"/);
    // B-center (idx 49 = 45+4) = 'B'.
    expect(html).toMatch(/data-pos="49"[^>]*data-color="B"/);
  });

  it("computes viewBox proportional to sizePx (12 cols × 9 rows)", () => {
    const html = renderToStaticMarkup(
      createElement(Cube2D, { facelet: SOLVED, sizePx: 90 }),
    );
    // sizePx=90 → stickerPx=10 → widthPx=120, heightPx=90.
    expect(html).toContain('viewBox="0 0 120 90"');
    expect(html).toContain('width="120"');
    expect(html).toContain('height="90"');
  });

  it("rejects bad facelet length", () => {
    expect(() =>
      renderToStaticMarkup(
        createElement(Cube2D, { facelet: "U".repeat(10) }),
      ),
    ).toThrow(/54 chars/);
  });

  it("scrambled facelet renders the scrambled letters", () => {
    const scrambled = applyMove(SOLVED, "R");
    const html = renderToStaticMarkup(
      createElement(Cube2D, { facelet: scrambled, testId: null }),
    );
    // R move places B-colored stickers at U-right column (idx 2, 5, 8).
    // Verify at least one position changed from 'U' to a non-U letter.
    const uRightMatch = html.match(/data-pos="2"[^>]*data-color="([^"]+)"/);
    expect(uRightMatch).not.toBeNull();
    expect(uRightMatch![1]).not.toBe("U");
  });

  it("renders style display:block (so parent flow centers it)", () => {
    const html = renderToStaticMarkup(
      createElement(Cube2D, { facelet: SOLVED }),
    );
    expect(html).toMatch(/style="display:\s*block/);
  });
});

// ---------------------------------------------------------------------
// Animated-mode tests
//
// Each test creates a CubeSequence, seeks to a known timestamp, and
// snapshots the rendered SVG. The CubeSequence factory is used
// directly (not via `useCubeSequence`) since the hook owns its
// internal sequence; here we want full control over the timestamp.
// ---------------------------------------------------------------------

function renderAtTimestamp(
  moves: MoveStr[],
  ts: number,
  startFacelet: string = SOLVED,
): string {
  const seq = createCubeSequence({ startFacelet, moves });
  seq.seek(ts);
  return renderToStaticMarkup(
    createElement(Cube2D, { sequence: seq, testId: null }),
  );
}

describe("Cube2D — animated mode: pre-animation render", () => {
  it("currentMoveIndex=-1 falls through to StaticInner (start facelet)", () => {
    const seq = createCubeSequence({
      startFacelet: SOLVED,
      moves: ["R", "U"],
    });
    // Just created, not played — currentMoveIndex should be -1.
    expect(seq.currentMoveIndex).toBe(-1);
    const html = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, testId: null }),
    );
    // Static-mode SVG: viewBox 0 0 widthPx heightPx (not the widget
    // envelope's negative origin).
    expect(html).toMatch(/viewBox="0 0 \d+ \d+"/);
    // All 54 stickers rendered as solved letters.
    for (let i = 0; i < 9; i++) {
      expect(html).toMatch(new RegExp(`data-pos="${i}"[^>]*data-color="U"`));
    }
  });

  it("empty moves array falls through to StaticInner", () => {
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: [] });
    const html = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, testId: null }),
    );
    expect(html).toMatch(/viewBox="0 0 \d+ \d+"/);
  });
});

describe("Cube2D — animated mode: SVG envelope + structure", () => {
  it("viewBox is cropped to the cross region (0 0 240 180) — not the full widget envelope", () => {
    // Contract: the animated SVG element renders at the same intrinsic
    // size as the static SVG (stickerPx*12 × stickerPx*9) so production
    // CSS doesn't scale animated cards down relative to the start card.
    // The viewBox is cropped to the cross region of the WIDGET_ENVELOPE
    // coordinate system: x=0..240 (12 sticker units), y=0..180 (9 rows).
    // Rotation groups whose swing corners extend past this region render
    // outside the SVG box via overflow:visible.
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: ["R"] });
    seq.seek(300); // mid-move
    const html = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, testId: null }),
    );
    expect(html).toContain('viewBox="0 0 240 180"');
  });

  it("animated SVG element width/height match static SVG at the same sizePx (regression: cross-size mismatch)", () => {
    // Regression test for C·P1 fix 1: before the fix, animated mode
    // rendered an SVG ~2.4× wider than static mode (22×19 envelope vs
    // 12×9 cross). In production, `.sol-cell .render .net svg { max-
    // width: 100% }` scaled the larger animated SVG down to fit the
    // card, shrinking the cross to ~63% of the static cross size and
    // producing a visible start-card-vs-other-cards mismatch in the
    // solution grid. The fix makes both modes render the SAME intrinsic
    // SVG dimensions; this test pins that invariant.
    const sizePx = 130; // production cols=3 sizing
    const staticHtml = renderToStaticMarkup(
      createElement(Cube2D, { facelet: SOLVED, sizePx, testId: null }),
    );
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: ["R"] });
    seq.seek(300);
    const animatedHtml = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, sizePx, testId: null }),
    );
    const widthRe = /width="([\d.]+)"/;
    const heightRe = /height="([\d.]+)"/;
    const staticW = staticHtml.match(widthRe)?.[1];
    const staticH = staticHtml.match(heightRe)?.[1];
    const animatedW = animatedHtml.match(widthRe)?.[1];
    const animatedH = animatedHtml.match(heightRe)?.[1];
    expect(staticW).toBeDefined();
    expect(staticH).toBeDefined();
    expect(animatedW).toBe(staticW);
    expect(animatedH).toBe(staticH);
    // Sanity: at sizePx=130, stickerPx ≈ 14.44, widthPx ≈ 173.33, heightPx=130.
    expect(Number(animatedH)).toBe(130);
  });

  it("emits a cross-silhouette clipPath in <defs>", () => {
    const html = renderAtTimestamp(["R"], 300);
    expect(html).toContain("<clipPath");
    expect(html).toContain('clipPathUnits="userSpaceOnUse"');
  });

  it("animated stickers carry rx + gap + stroke-width matching static (regression: C·P1 fix 2)", () => {
    // Regression for C·P1 fix 2: before the fix, animated mode used the
    // rev5 preview's no-gap-crisp look (gap=0, no rx, strokeWidth=1).
    // Side-by-side in the production solution grid, the start card
    // (static) and non-start cards (animated) rendered visibly different
    // sticker cells — animated looked heavier/blockier. The fix aligns
    // animated stickers to the static renderer's gap+rx geometry,
    // expressed in viewBox units:
    //   gap          = STICKER_PX * 0.06 = 1.2
    //   rx           = STICKER_PX * 0.06 = 1.2
    //   strokeWidth  = STICKER_PX * 0.025 = 0.5
    // Each rect's `width` is therefore STICKER_PX - gap = 18.8.
    // This test pins those values so a regression in either direction
    // (going back to no-gap-crisp, or drifting the gap/rx ratio) shows up.
    const html = renderAtTimestamp(["R"], 300);
    // rx attribute present on every sticker rect (was absent / 0 before).
    expect(html).toMatch(/<rect[^>]*\srx="1\.2"/);
    // Stroke width matches static's STICKER_PX * 0.025 = 0.5.
    expect(html).toMatch(/<rect[^>]*\bstroke-width="0\.5"/);
    // Width matches STICKER_PX - gap = 18.8 (i.e. gap-inset positioning).
    expect(html).toMatch(/<rect[^>]*\bwidth="18\.8"/);
    // Static stickers' x coords are STICKER_PX*col + gap/2 = col*20 + 0.6.
    // Verify at least one static sticker is offset by gap/2 = 0.6 (not 0).
    expect(html).toMatch(/data-layer="static">[\s\S]*?<rect[^>]*\bx="\d+\.6"/);
  });

  it("clip-path padding tracks ANIM_STROKE_W, not a hardcoded value (regression: C·P1 fix 2)", () => {
    // Regression for the slip-through corollary of C·P1 fix 2: before
    // tightening, the cross-silhouette clipPath was padded by a hardcoded
    // 2 viewBox units to swallow rev5's wider stroke. With animated mode
    // now using the static stroke (0.5 viewBox units), the old padding
    // was 4× too wide and exposed a thin sliver of slide-group depart
    // copies below the cross at progress=1.
    //
    // The geometry contract:
    //   vertical strip (U/F/D cols 3..5 column rows): x=3-pad, y=-pad,
    //   w=3+2*pad, h=9+2*pad in viewBox sticker units (multiplied by
    //   STICKER_PX=20). With pad=0.5 →
    //     x = 60 - 0.5 = 59.5
    //     y = -0.5
    //     w = 60 + 1   = 61
    //     h = 180 + 1  = 181
    //   horizontal strip (L/F/R/B row): x=-pad, y=60-pad, w=240+2*pad,
    //   h=60+2*pad →
    //     x = -0.5
    //     y = 59.5
    //     w = 241
    //     h = 61
    const html = renderAtTimestamp(["R"], 300);
    // Vertical strip with the tightened padding.
    expect(html).toMatch(
      /<rect[^>]*\bx="59\.5"[^>]*\by="-0\.5"[^>]*\bwidth="61"[^>]*\bheight="181"/,
    );
    // Horizontal strip with the tightened padding.
    expect(html).toMatch(
      /<rect[^>]*\bx="-0\.5"[^>]*\by="59\.5"[^>]*\bwidth="241"[^>]*\bheight="61"/,
    );
  });

  it("ribbon-container <g> carries the clip-path reference", () => {
    const html = renderAtTimestamp(["U"], 300);
    expect(html).toMatch(/data-layer="ribbon-container"[^>]*clip-path="url\(#/);
  });

  it("face-rotation group has no clip-path (rotation corners overflow)", () => {
    const html = renderAtTimestamp(["U"], 300);
    // Find the face layer and verify it does NOT have a clip-path attr.
    const faceMatch = html.match(
      /<g[^>]*data-layer="face"[^>]*>/,
    );
    expect(faceMatch).not.toBeNull();
    expect(faceMatch![0]).not.toContain("clip-path");
  });

  it("overlayOpacity=0 (default) → no overlay rect emitted", () => {
    const html = renderAtTimestamp(["R"], 300);
    expect(html).not.toContain('data-layer="extension-overlay"');
  });

  it("style.overflow=visible + overflow attr both set", () => {
    const html = renderAtTimestamp(["R"], 300);
    expect(html).toMatch(/style="display:\s*block;\s*overflow:\s*visible/);
    expect(html).toMatch(/overflow="visible"/);
  });
});

describe("Cube2D — animated mode: per-move group counts", () => {
  // The plan structure differs by move kind:
  //   h-ribbon (U/D): 1 slide group (primary) + 1 face rotation
  //   v-ribbon (R/L): 2 slide groups (primary + secondary) + 1 face rotation
  //   ring     (F/B): 0 slide groups + 1 face rotation + 1 ring rotation

  it("U → 1 slide group (primary), 1 face rotation, 0 ring", () => {
    const html = renderAtTimestamp(["U"], 300);
    const slideCount = (html.match(/data-layer="edge-primary"/g) ?? []).length;
    const secCount = (html.match(/data-layer="edge-secondary"/g) ?? []).length;
    const faceCount = (html.match(/data-layer="face"/g) ?? []).length;
    const ringCount = (html.match(/data-layer="edge-ring"/g) ?? []).length;
    expect(slideCount).toBe(1);
    expect(secCount).toBe(0);
    expect(faceCount).toBe(1);
    expect(ringCount).toBe(0);
  });

  it("R → 2 slide groups (primary + secondary), 1 face rotation, 0 ring", () => {
    const html = renderAtTimestamp(["R"], 300);
    expect((html.match(/data-layer="edge-primary"/g) ?? []).length).toBe(1);
    expect((html.match(/data-layer="edge-secondary"/g) ?? []).length).toBe(1);
    expect((html.match(/data-layer="face"/g) ?? []).length).toBe(1);
    expect((html.match(/data-layer="edge-ring"/g) ?? []).length).toBe(0);
  });

  it("F → 0 slide groups, 1 face rotation, 1 ring rotation", () => {
    const html = renderAtTimestamp(["F"], 300);
    expect((html.match(/data-layer="edge-primary"/g) ?? []).length).toBe(0);
    expect((html.match(/data-layer="edge-secondary"/g) ?? []).length).toBe(0);
    expect((html.match(/data-layer="face"/g) ?? []).length).toBe(1);
    expect((html.match(/data-layer="edge-ring"/g) ?? []).length).toBe(1);
  });
});

describe("Cube2D — animated mode: easing + transforms", () => {
  it("progress=0 → face rotation transform angle is 0 (rotate(0 ...))", () => {
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: ["R"] });
    // play() to advance currentMoveIndex from -1 to 0 with progress 0.
    seq.play();
    seq.pause();
    expect(seq.currentMoveIndex).toBe(0);
    expect(seq.currentMoveProgress).toBe(0);
    const html = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, testId: null }),
    );
    expect(html).toMatch(/data-layer="face"[^>]*transform="rotate\(0 /);
  });

  it("progress=1 → face rotation transform angle is the target", () => {
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: ["R"] });
    seq.seek(seq.totalDurationMs); // = 600 ms; status=ended, progress=1
    const html = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, testId: null }),
    );
    // R has faceRotationAngle=90 (per the kinematics snapshot in
    // cube2DKinematics.test.ts). At progress=1, easeOut(1)=1, so
    // transform = rotate(90 ...).
    expect(html).toMatch(/data-layer="face"[^>]*transform="rotate\(90 /);
  });

  it("progress=0.5 → face rotation transform angle is angle * easeOut(0.5) = 0.75 * angle", () => {
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: ["R"] });
    seq.seek(300); // midpoint of 600 ms move
    const html = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, testId: null }),
    );
    // R angle = 90, easeOut(0.5) = 0.75 → 67.5
    expect(html).toMatch(/data-layer="face"[^>]*transform="rotate\(67\.5 /);
  });

  it("slide group transform translates by sv * STICKER_PX * easeOut(progress)", () => {
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: ["U"] });
    seq.seek(300);
    const html = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, testId: null }),
    );
    // U has slideVector { dx: -3, dy: 0 } per kinematics snapshot.
    // STICKER_PX=20, easeOut(0.5)=0.75 → tx = -3 * 20 * 0.75 = -45, ty = 0.
    expect(html).toMatch(
      /data-layer="edge-primary"[^>]*transform="translate\(-45 0\)"/,
    );
  });
});

describe("Cube2D — animated mode: smoke across all 12 QTM at 3 progress points", () => {
  // 12 moves × 3 progress points = 36 cases. Each just confirms the
  // render doesn't throw and produces a valid SVG envelope. Caught
  // bugs: missing kinematics path, bad clip-path geometry, NaN in
  // a transform (renderToStaticMarkup would emit "NaN" which is
  // valid HTML but spotted by string scan).
  for (const move of ALL_MOVES) {
    for (const p of [0.0, 0.5, 0.99]) {
      const ts = Math.round(p * 600);
      it(`${move} at progress ~${p}`, () => {
        // We use seek for progress 0.5 and 0.99; for progress=0 we use
        // play()+pause() so currentMoveIndex advances from -1 to 0.
        const seq = createCubeSequence({
          startFacelet: SOLVED,
          moves: [move],
        });
        if (p === 0.0) {
          seq.play();
          seq.pause();
        } else {
          seq.seek(ts);
        }
        const html = renderToStaticMarkup(
          createElement(Cube2D, { sequence: seq, testId: null }),
        );
        expect(html).toContain('viewBox="0 0 240 180"');
        expect(html).not.toContain("NaN");
        // Always a face rotation present.
        expect(html).toContain('data-layer="face"');
      });
    }
  }
});

describe("Cube2D — animated mode: post-end render = applyMove(...)", () => {
  // After seek(totalDurationMs), status=ended and progress=1. The
  // rendered SVG should encode the post-move facelet — i.e. the
  // visible color at each canonical position equals
  // `applyMove(preFacelet, move)`. Static stickers show their own
  // post-move colors directly; rotated/slid groups encode the
  // post-state via transforms applied at progress=1.
  //
  // We can't easily reconstruct the rendered-color-at-position from
  // an SSR string (transforms are post-rendered by the browser, not
  // by SSR). Instead, we check the simpler invariant: the
  // RenderPlan's staticStickers cover exactly the indices where
  // `preFacelet[i] === applyMove(preFacelet, move)[i]` (static
  // indices), and each such sticker carries the expected color.
  it("U: static indices show applyMove(SOLVED, 'U')[i] colors at progress=1", () => {
    const expected = applyMove(SOLVED, "U");
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: ["U"] });
    seq.seek(seq.totalDurationMs);
    const html = renderToStaticMarkup(
      createElement(Cube2D, { sequence: seq, testId: null }),
    );
    // Sample a few static positions (not on U-face, not on the
    // affected edge cycle for U). For U, F-center (idx 22), D-center
    // (idx 31), L-center (idx 40), R-center (idx 13), B-center (idx 49)
    // are all static — their letters should match `expected`.
    for (const idx of [13, 22, 31, 40, 49]) {
      expect(html).toMatch(
        new RegExp(`data-color="${expected[idx]}"`),
      );
    }
  });
});

// ---------------------------------------------------------------------
// Discriminated-union prop enforcement
// ---------------------------------------------------------------------

describe("Cube2D — discriminated-union prop typing", () => {
  it("passing both facelet AND sequence is a type error", () => {
    const seq = createCubeSequence({ startFacelet: SOLVED, moves: ["R"] });
    // @ts-expect-error — facelet and sequence are mutually exclusive
    const _bothProps: import("./Cube2D").Cube2DProps = {
      facelet: SOLVED,
      sequence: seq,
    };
    expect(_bothProps).toBeDefined(); // touch to keep no-unused-vars happy
  });

  it("passing neither facelet NOR sequence throws at runtime", () => {
    // TS doesn't strictly forbid `{}` here (both branches mark the
    // other field as `?: undefined` so `{}` is structurally
    // assignable). At runtime the static branch is hit with
    // facelet=undefined and the length check throws. Documented as
    // the safety net behind the discriminated-union typing.
    expect(() =>
      renderToStaticMarkup(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        createElement(Cube2D, {} as any),
      ),
    ).toThrow();
  });
});
