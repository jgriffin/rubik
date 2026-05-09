import { test, expect } from "@playwright/test";

test.describe("scramble + solve flow", () => {
  test("scramble changes the cube; solve produces a move list (stub mode)", async ({
    page,
  }) => {
    await page.goto("/");

    // Wait for health → buttons enabled.
    const scramble = page.getByTestId("scramble-button");
    await expect(scramble).toBeEnabled();

    // Initial state: cube is solved.
    const svg = page.getByTestId("flat-cube");
    await expect(svg.locator("rect")).toHaveCount(54);

    // Click Scramble.
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
            const actual = await svg
              .locator(`rect[data-pos="${pos}"]`)
              .getAttribute("data-color");
            if (actual !== expected) return true;
          }
          return false;
        },
        { timeout: 5_000 },
      )
      .toBe(true);

    // Click Solve. Stub net never solves, but the wire should still respond.
    const solve = page.getByTestId("solve-button");
    await solve.click();

    // MoveList becomes visible.
    await expect(page.getByTestId("move-list")).toBeVisible({ timeout: 10_000 });
    const status = page.getByTestId("solve-status");
    await expect(status).toBeVisible();

    // Stub net: solved=false, moves=[]; assertion is loose — we accept either.
    const dataSolved = await page
      .getByTestId("move-list")
      .getAttribute("data-solved");
    expect(["true", "false"]).toContain(dataSolved);
  });
});

test.describe("real-backend scramble + solve", () => {
  test.skip(
    process.env.PLAYWRIGHT_REAL_BACKEND !== "1",
    "set PLAYWRIGHT_REAL_BACKEND=1 to run",
  );
  test("real LN model solves length-14 scramble", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("scramble-button").click();
    await page.getByTestId("solve-button").click();
    await expect(page.getByTestId("move-list")).toHaveAttribute(
      "data-solved",
      "true",
      { timeout: 30_000 },
    );
    const moves = page.getByTestId("move-item");
    expect(await moves.count()).toBeGreaterThan(0);
  });
});
