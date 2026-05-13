import { test, expect } from "@playwright/test";
import { applyMove } from "../src/state/applyMove";
import type { MoveStr } from "../src/state/faceletMoves";

// DUR_FORWARD_MS (forward leg) is 600ms in cube2DKinematics.ts. We add
// a small safety buffer for the rAF tick + React render after settle.
// Mid-animation assertions are intentionally skipped here — animation
// progress depends on rAF timing the e2e environment doesn't control
// precisely. Per-rAF-tick coverage lives in the vitest suite
// (state/cubeSequence.test.ts + components/Cube2D.test.tsx).
const DUR_FORWARD_MS = 600;
const SETTLE_BUFFER_MS = 50;

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

// Face-offset table mirrors src/components/cubePalette.ts. Used here to
// compute the center index of each face for the post-settle data-color
// assertion (centers + 4 are static for any single QTM move EXCEPT the
// turned face's own center).
const FACE_OFFSETS: Record<string, number> = {
  U: 0,
  R: 9,
  F: 18,
  D: 27,
  L: 36,
  B: 45,
};

test.describe("solution grid — toggles + card selection", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("scramble-button")).toBeEnabled();
  });

  test("columns switch swaps active button and sets --cols var", async ({ page }) => {
    // Default is 3.
    await expect(page.getByTestId("columns-3")).toHaveClass(/on/);

    await page.getByTestId("columns-6").click();
    await expect(page.getByTestId("columns-6")).toHaveClass(/on/);
    await expect(page.getByTestId("columns-3")).not.toHaveClass(/on/);
    await expect(page.getByTestId("solution-grid")).toHaveAttribute(
      "style",
      /--cols: ?6/,
    );
  });

  test("v2 toggles still disabled (1-column)", async ({ page }) => {
    await expect(page.getByTestId("columns-1")).toBeDisabled();
  });

  test("3D toggle swaps renderer between flat (rect[data-pos]) and twisty-player wrapper", async ({
    page,
  }) => {
    // Start card always renders. Render-mode is two independent toggles
    // (2D / 3D); pressing one alone selects that renderer, both pressed
    // gives split. The constraint is that at least one stays on.
    const card = page.getByTestId("sol-card-0");
    await expect(card).toBeVisible();

    // Default: 2D on, 3D off → net mode (54 flat rects, no twisty wrapper).
    await expect(page.getByTestId("render-mode-2d")).toHaveClass(/on/);
    await expect(page.getByTestId("render-mode-3d")).not.toHaveClass(/on/);
    await expect(card.locator("rect[data-pos]")).toHaveCount(54);
    await expect(card.getByTestId("twisty-cube")).toHaveCount(0);

    // Click 3D — now both on (dual). Click 2D off — only 3D (iso mode).
    await page.getByTestId("render-mode-3d").click();
    await page.getByTestId("render-mode-2d").click();
    await expect(page.getByTestId("render-mode-2d")).not.toHaveClass(/on/);
    await expect(page.getByTestId("render-mode-3d")).toHaveClass(/on/);
    await expect(card.locator("rect[data-pos]")).toHaveCount(0);
    await expect(card.getByTestId("twisty-cube")).toBeVisible();
    await expect(card.getByTestId("twisty-cube")).toHaveCount(1);

    // Click 2D back on — both on again (dual). Click 3D off → only 2D.
    await page.getByTestId("render-mode-2d").click();
    await page.getByTestId("render-mode-3d").click();
    await expect(page.getByTestId("render-mode-2d")).toHaveClass(/on/);
    await expect(page.getByTestId("render-mode-3d")).not.toHaveClass(/on/);
    await expect(card.locator("rect[data-pos]")).toHaveCount(54);
    await expect(card.getByTestId("twisty-cube")).toHaveCount(0);
  });

  test("both 2D and 3D on renders the side-by-side split view", async ({
    page,
  }) => {
    const card = page.getByTestId("sol-card-0");
    await expect(card).toBeVisible();

    // Default 2D on; click 3D → both on (dual).
    await page.getByTestId("render-mode-3d").click();
    await expect(page.getByTestId("render-mode-2d")).toHaveClass(/on/);
    await expect(page.getByTestId("render-mode-3d")).toHaveClass(/on/);

    // Both pair-testid renderers are present.
    await expect(card.getByTestId("flat-cube-pair")).toBeVisible();
    await expect(card.getByTestId("twisty-cube-pair")).toBeVisible();

    // Cube2D's static-mode contract still holds: 54 stickers as <rect data-pos>.
    // Wrapper presence is the 3D contract; shadow-DOM pixels are out of scope.
    await expect(card.locator("rect[data-pos]")).toHaveCount(54);

    // Click 3D off → back to 2D only; pair testids gone.
    await page.getByTestId("render-mode-3d").click();
    await expect(page.getByTestId("render-mode-2d")).toHaveClass(/on/);
    await expect(page.getByTestId("render-mode-3d")).not.toHaveClass(/on/);
    await expect(card.getByTestId("flat-cube-pair")).toHaveCount(0);
    await expect(card.getByTestId("twisty-cube-pair")).toHaveCount(0);
    await expect(card.locator("rect[data-pos]")).toHaveCount(54);
  });

  test("toggling off the only-on side is rejected (at least one stays on)", async ({
    page,
  }) => {
    // Default 2D on, 3D off. Try to click 2D off — should be a no-op.
    await expect(page.getByTestId("render-mode-2d")).toHaveClass(/on/);
    await page.getByTestId("render-mode-2d").click();
    await expect(page.getByTestId("render-mode-2d")).toHaveClass(/on/);
    await expect(page.getByTestId("render-mode-3d")).not.toHaveClass(/on/);
  });

  test("clicking the start card sets it active", async ({ page }) => {
    const start = page.getByTestId("sol-card-0");
    await start.click();
    await expect(start).toHaveAttribute("data-active", "true");
  });

  test("cube-size header toggle moves the active state", async ({ page }) => {
    await expect(page.getByTestId("cube-size-3")).toHaveClass(/on/);
    await page.getByTestId("cube-size-2").click();
    await expect(page.getByTestId("cube-size-2")).toHaveClass(/on/);
    await expect(page.getByTestId("cube-size-3")).not.toHaveClass(/on/);
  });
});

// ---------------------------------------------------------------------
// Per-card animation — deterministic post-settle assertion.
//
// Block C wired the cube animation primitive into production
// SolutionCard: non-start cards animate their move on first scroll-
// into-view via IntersectionObserver → seq.play(). Forward leg runs
// for DUR_FORWARD_MS=600ms; on settle status="ended" and progress=1.
//
// The settle assertion contract (per Cube2D.test.tsx precedent — see
// "post-end render = applyMove(...)" describe block):
//   - After DUR_FORWARD_MS + SETTLE_BUFFER_MS, the animated SVG
//     contains data-layer="face" — proves AnimatedInner is the render
//     path (no fallthrough to StaticInner, no idle freeze).
//   - The static-stickers layer carries each non-rotating-face center's
//     post-move color. For any single QTM move M, the 5 non-M centers
//     are guaranteed static (perm[center]=center), and their pre-move
//     letter equals their post-move letter — the canonical face letter.
//   - The full 6-letter color set still appears on the card — a move
//     doesn't drop any color from the visible cross.
//
// Why not 54 data-pos rects: animated mode's AnimatedSticker carries
// (x, y, color) only — no data-pos. That contract is exclusive to
// StaticInner (start card + non-start pre-animation fallthrough), and
// the existing tests above still exercise it via sol-card-0.
//
// Mid-animation assertions are intentionally skipped — animation
// progress depends on rAF timing the e2e environment doesn't control
// precisely. Per-rAF-tick coverage lives in vitest.
// ---------------------------------------------------------------------

test.describe("solution grid — per-card animation post-settle", () => {
  // Three-move solution covers both kinematic families:
  //   R  → v-ribbon slide (R-face rotates, R/L vertical strip slides)
  //   U  → h-ribbon slide (U-face rotates, U/D horizontal strip slides)
  //   F  → ring rotation  (F/B face cycles edges around F-center)
  // Each card therefore exercises a different animated-mode code path
  // through getRenderInstructions / SvgFromRenderPlan.
  const SOLUTION: MoveStr[] = ["R", "U", "F"];

  test.beforeEach(async ({ page }) => {
    // Mock both /api/scramble and /api/solve. We need a deterministic
    // (scrambleState, solution) pair so the per-card facelets are
    // known up-front. Using SOLVED as the scramble keeps the math
    // simple: card N's preFacelet = applyMoves(SOLVED, SOLUTION[0..N-1]).
    await page.route("**/api/scramble", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ state: SOLVED, moves: [] }),
      });
    });
    await page.route("**/api/solve", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          solved: true,
          moves: SOLUTION,
          stats: {
            time_ms: 1,
            beam_width: 64,
            steps_searched: SOLUTION.length,
            final_value: 0,
          },
        }),
      });
    });

    await page.goto("/");
    await expect(page.getByTestId("scramble-button")).toBeEnabled();
    // Trigger scramble (no-op state, but flushes solution + activeIdx
    // through App's resetSolveState path), then solve to populate the
    // grid with SOLUTION's move cards.
    await page.getByTestId("scramble-button").click();
    await page.getByTestId("solve-button").click();
    // The 3 move cards mount once the solve resolves.
    await expect(page.getByTestId("sol-card-3")).toBeVisible({
      timeout: 5_000,
    });
  });

  test("each non-start card settles to its post-move static colors", async ({
    page,
  }) => {
    // Walk the solution; per card, scroll into view → wait the settle
    // window → assert static-layer post-move colors at sampled centers.
    let pre = SOLVED;
    for (let n = 1; n <= SOLUTION.length; n++) {
      const move = SOLUTION[n - 1];
      const post = applyMove(pre, move);

      const card = page.getByTestId(`sol-card-${n}`);
      await card.scrollIntoViewIfNeeded();
      // The first scroll-into-view fires IntersectionObserver → seq.play()
      // → forward leg runs for DUR_FORWARD_MS. We add SETTLE_BUFFER_MS
      // for the trailing rAF tick + React render before reading the DOM.
      // Cards already in the initial viewport will have started animating
      // earlier than this wait; that's fine — we only need them settled
      // by the time we assert, not synchronized with this wait.
      await page.waitForTimeout(DUR_FORWARD_MS + SETTLE_BUFFER_MS);

      // Animated mode actually rendered (not the idle fallthrough).
      // SvgFromRenderPlan tags layers; data-layer="face" is always
      // present (every move has a face rotation group).
      const svg = card.locator("svg").first();
      await expect(svg.locator('g[data-layer="face"]')).toHaveCount(1);

      // Static-layer rects carry their post-move colors at the
      // non-rotating-face centers (always 5 of the 6 centers, since
      // exactly one face's center is on the rotating layer per move).
      // applyMove preserves center-position invariant for those 5, so
      // post[centerIdx] === pre[centerIdx] === canonical face letter.
      const turnedFace = move.charAt(0); // "R", "U", "F", ...
      for (const face of ["U", "R", "F", "D", "L", "B"] as const) {
        if (face === turnedFace) continue;
        const centerIdx = FACE_OFFSETS[face] + 4;
        const expectedLetter = post[centerIdx];
        // Static-layer rects don't carry data-pos in animated mode —
        // they're rendered with data-color only. Asserting at least
        // one static-layer rect carries the expected center letter
        // covers the contract without coupling to viewBox geometry.
        await expect(
          svg.locator(`g[data-layer="static"] rect[data-color="${expectedLetter}"]`),
        ).not.toHaveCount(0);
      }

      // Every face's color letter is still present somewhere on the
      // card — a move is a permutation, so no color drops out.
      for (const letter of ["U", "R", "F", "D", "L", "B"]) {
        await expect(
          svg.locator(`rect[data-color="${letter}"]`),
        ).not.toHaveCount(0);
      }

      pre = post;
    }
  });

  test("start card stays static — no animated layers", async ({ page }) => {
    // sol-card-0 is the static path always (no Cube2D animated mode).
    // This guards against a regression where the start card accidentally
    // gains a sequence and starts animating its own non-existent move.
    const start = page.getByTestId("sol-card-0");
    await expect(start).toBeVisible();
    const svg = start.locator("svg").first();
    // Static path still emits data-pos on all 54 rects.
    await expect(svg.locator("rect[data-pos]")).toHaveCount(54);
    // No animated-mode layer tags.
    await expect(svg.locator("g[data-layer]")).toHaveCount(0);
  });
});
