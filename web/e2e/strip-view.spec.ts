import { test, expect } from "@playwright/test";

test.describe("strip view (real backend)", () => {
  test.skip(
    process.env.PLAYWRIGHT_REAL_BACKEND !== "1",
    "set PLAYWRIGHT_REAL_BACKEND=1 to run",
  );

  test("real LN model: scramble → solve → strip renders → jump-to-cell", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page.getByTestId("scramble-button")).toBeEnabled({
      timeout: 30_000,
    });

    // Scramble length 8 — easy for the LN model, gives 8-move solution roughly.
    await page.getByTestId("length-slider").fill("8");
    await page.getByTestId("scramble-button").click();
    // Wait for state-field to populate from new scramble (not the initial
    // SOLVED placeholder).
    const SOLVED =
      "U".repeat(9) +
      "R".repeat(9) +
      "F".repeat(9) +
      "D".repeat(9) +
      "L".repeat(9) +
      "B".repeat(9);
    await page.waitForFunction(
      (solved: string) => {
        const el = document.querySelector(
          '[data-testid="state-field"]',
        ) as HTMLTextAreaElement | null;
        return el !== null && el.value.length === 54 && el.value !== solved;
      },
      SOLVED,
      { timeout: 10_000 },
    );

    // Solve.
    await page.getByTestId("solve-button").click();
    await expect(page.getByTestId("move-list")).toHaveAttribute(
      "data-solved",
      "true",
      { timeout: 30_000 },
    );

    // Read total moves from move-list.
    const moves = page.getByTestId("move-item");
    const total = await moves.count();
    expect(total).toBeGreaterThan(0);

    // Strip view should have total + 1 cells.
    const cells = page.locator('[data-testid^="strip-cell-"]');
    await expect(cells).toHaveCount(total + 1);

    // Click cell 4 (or min(4, total) if shorter).
    const targetIdx = Math.min(4, total);
    await page.getByTestId(`strip-cell-${targetIdx}`).click();

    // Step counter (in StepControls) should reflect.
    await expect(page.getByTestId("step-counter")).toHaveText(
      `${targetIdx} / ${total}`,
    );

    // The clicked cell is active; cell 0 is not (when targetIdx > 0).
    await expect(page.getByTestId(`strip-cell-${targetIdx}`)).toHaveAttribute(
      "data-active",
      "true",
    );
    if (targetIdx !== 0) {
      await expect(page.getByTestId("strip-cell-0")).toHaveAttribute(
        "data-active",
        "false",
      );
    }
  });
});
