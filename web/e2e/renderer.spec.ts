import { test, expect } from "@playwright/test";

test.describe("FlatCubeRenderer (rendered inside the start card)", () => {
  test("mounts with the solved facelet on first load", async ({ page }) => {
    await page.goto("/");

    const startCard = page.getByTestId("sol-card-0");
    await expect(startCard).toBeVisible();
    const svg = startCard.locator("svg").first();

    const stickers = svg.locator("rect");
    await expect(stickers).toHaveCount(54);

    for (const letter of ["U", "R", "F", "D", "L", "B"]) {
      await expect(svg.locator(`rect[data-color="${letter}"]`)).toHaveCount(9);
    }

    const expected: Array<[number, string]> = [
      [4, "U"],
      [13, "R"],
      [22, "F"],
      [31, "D"],
      [40, "L"],
      [49, "B"],
    ];
    for (const [pos, color] of expected) {
      await expect(svg.locator(`rect[data-pos="${pos}"]`)).toHaveAttribute(
        "data-color",
        color,
      );
    }
  });
});
