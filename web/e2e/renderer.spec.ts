import { test, expect } from "@playwright/test";

test.describe("FlatCubeRenderer", () => {
  test("mounts with solved facelet", async ({ page }) => {
    await page.goto("/");
    const svg = page.getByTestId("flat-cube");
    await expect(svg).toBeVisible();

    // 54 sticker rects.
    const stickers = svg.locator("rect");
    await expect(stickers).toHaveCount(54);

    // Each color letter appears exactly 9 times (solved state).
    for (const letter of ["U", "R", "F", "D", "L", "B"]) {
      const matching = svg.locator(`rect[data-color="${letter}"]`);
      await expect(matching).toHaveCount(9);
    }

    // U-face center sticker is at pos 4 (the 5th sticker of the U-face block,
    // 0-indexed) and must be 'U' on a solved cube. R-face center at pos 13,
    // F at 22, D at 31, L at 40, B at 49.
    const expected: Array<[number, string]> = [
      [4, "U"],
      [13, "R"],
      [22, "F"],
      [31, "D"],
      [40, "L"],
      [49, "B"],
    ];
    for (const [pos, color] of expected) {
      const center = svg.locator(`rect[data-pos="${pos}"]`);
      await expect(center).toHaveAttribute("data-color", color);
    }
  });
});
