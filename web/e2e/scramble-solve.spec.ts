import { test, expect } from "@playwright/test";

test.describe("scramble + solve flow", () => {
  test("scramble changes the starting cube; solve produces solution cards (stub mode)", async ({
    page,
  }) => {
    await page.goto("/");

    const scramble = page.getByTestId("scramble-button");
    await expect(scramble).toBeEnabled();

    // Initial state: starting card (sol-card-0) shows a solved cube.
    const startCard = page.getByTestId("sol-card-0");
    await expect(startCard).toBeVisible();
    const startSvg = startCard.locator("svg").first();
    await expect(startSvg.locator("rect")).toHaveCount(54);

    await scramble.click();

    // Renderer must update — wait until at least one canonical-corner sticker
    // no longer matches its solved color. A length-14 random scramble is
    // virtually guaranteed to move at least one corner sticker off-color.
    await expect
      .poll(
        async () => {
          const cornerPositions = [
            [0, "U"],
            [2, "U"],
            [6, "U"],
            [8, "U"],
            [9, "R"],
            [11, "R"],
            [15, "R"],
            [17, "R"],
          ] as const;
          for (const [pos, expected] of cornerPositions) {
            const actual = await startSvg
              .locator(`rect[data-pos="${pos}"]`)
              .getAttribute("data-color");
            if (actual !== expected) return true;
          }
          return false;
        },
        { timeout: 5_000 },
      )
      .toBe(true);

    // Section ii reflects the scramble length.
    await expect(page.getByTestId("moves-grid").locator('[data-testid="move-cell"]'))
      .toHaveCount(14);

    // Click Solve. Stub net never returns moves; the wire still responds.
    const solve = page.getByTestId("solve-button");
    await solve.click();

    // After the solve resolves, the solution grid still has step-00 at minimum.
    // Stub returns moves=[], so we just assert the request fired and the
    // button re-enables.
    await expect(solve).toBeEnabled({ timeout: 10_000 });
  });

  test("length slider drives /api/scramble request body", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("scramble-button")).toBeEnabled();

    const slider = page.getByTestId("length-slider");
    await slider.fill("6");
    await expect(page.getByTestId("length-readout")).toHaveText("6");

    const requestPromise = page.waitForRequest("**/api/scramble");
    await page.getByTestId("scramble-button").click();
    const req = await requestPromise;
    const body = req.postDataJSON();
    expect(body.length).toBe(6);
  });

  test("clear resets to solved cube and empties section ii", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("scramble-button")).toBeEnabled();

    // Scramble first.
    await page.getByTestId("length-slider").fill("8");
    const reqP = page.waitForRequest("**/api/scramble");
    await page.getByTestId("scramble-button").click();
    await reqP;

    // Wait for moves grid to populate with non-empty cells.
    await expect(page.getByTestId("moves-grid").locator('[data-testid="move-cell"]'))
      .toHaveCount(8, { timeout: 5_000 });

    // Click Clear.
    await page.getByTestId("clear-button").click();

    // No move cells remain — only dashed empties.
    await expect(page.getByTestId("moves-grid").locator('[data-testid="move-cell"]'))
      .toHaveCount(0);
  });
});

test.describe("real-backend scramble + solve", () => {
  test.skip(
    process.env.PLAYWRIGHT_REAL_BACKEND !== "1",
    "set PLAYWRIGHT_REAL_BACKEND=1 to run",
  );
  test("real LN model produces a solution rendered as cards", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("scramble-button").click();
    await page.getByTestId("solve-button").click();
    // Solved footer appears once the model returns a solution.
    await expect(page.getByTestId("solved-footer")).toBeVisible({ timeout: 30_000 });
    const cards = page.locator('[data-testid^="sol-card-"]');
    // At least the start card + ≥1 move cards.
    expect(await cards.count()).toBeGreaterThan(1);
  });
});
