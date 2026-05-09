import { test, expect } from "@playwright/test";

test.describe("step-through animation", () => {
  test.skip(
    process.env.PLAYWRIGHT_REAL_BACKEND !== "1",
    "set PLAYWRIGHT_REAL_BACKEND=1 to run (stub net doesn't actually solve)",
  );

  test("scramble → solve → play returns cube to solved", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("scramble-button").click();
    await page.getByTestId("solve-button").click();
    await expect(page.getByTestId("move-list")).toHaveAttribute(
      "data-solved",
      "true",
      { timeout: 30_000 },
    );

    // Counter shows "0 / N" initially.
    const counter = page.getByTestId("step-counter");
    const initialText = await counter.textContent();
    expect(initialText).toMatch(/^0 \/ \d+$/);
    const total = parseInt(initialText!.split(" / ")[1], 10);
    expect(total).toBeGreaterThan(0);

    // Click play.
    await page.getByTestId("play-button").click();

    // Wait for the counter to read N / N (allow ~500ms/move + slack).
    await expect(counter).toHaveText(`${total} / ${total}`, {
      timeout: total * 1000 + 5_000,
    });

    // Cube should be solved: each face center holds its canonical color.
    const svg = page.getByTestId("flat-cube");
    const centers: ReadonlyArray<readonly [number, string]> = [
      [4, "U"],
      [13, "R"],
      [22, "F"],
      [31, "D"],
      [40, "L"],
      [49, "B"],
    ];
    for (const [pos, color] of centers) {
      await expect(svg.locator(`rect[data-pos="${pos}"]`)).toHaveAttribute(
        "data-color",
        color,
      );
    }

    // Every face: 9 stickers of the same letter.
    for (const letter of ["U", "R", "F", "D", "L", "B"]) {
      await expect(svg.locator(`rect[data-color="${letter}"]`)).toHaveCount(9);
    }
  });
});
