import { test, expect } from "@playwright/test";

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

test.describe("paste/copy text fields", () => {
  test.beforeEach(async ({ page, context }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.goto("/");
    await expect(page.getByTestId("scramble-button")).toBeEnabled();
  });

  test("copy state to clipboard reflects current scrambleState", async ({
    page,
  }) => {
    // Scramble length 8 first.
    await page.getByTestId("length-slider").fill("8");
    const reqP = page.waitForRequest("**/api/scramble");
    await page.getByTestId("scramble-button").click();
    await reqP;
    // Wait until state-field updates from the new scrambleState (key remount).
    await expect(page.getByTestId("state-field")).not.toHaveValue(SOLVED, {
      timeout: 5_000,
    });
    const fieldValue = await page.getByTestId("state-field").inputValue();
    expect(fieldValue).toHaveLength(54);

    await page.getByTestId("state-copy").click();
    const clip = await page.evaluate(() => navigator.clipboard.readText());
    expect(clip).toBe(fieldValue);
  });

  test("paste solved state into state field renders solved cube", async ({
    page,
  }) => {
    // Scramble first so the cube isn't already solved.
    const reqP = page.waitForRequest("**/api/scramble");
    await page.getByTestId("scramble-button").click();
    await reqP;
    await expect(page.getByTestId("state-field")).not.toHaveValue(SOLVED, {
      timeout: 5_000,
    });

    // Paste solved state and click Set.
    await page.getByTestId("state-field").fill(SOLVED);
    await page.getByTestId("state-set").click();

    // No error.
    await expect(page.getByTestId("state-error")).toHaveCount(0);

    // Cube renders solved — center stickers are solved colors.
    const svg = page.getByTestId("flat-cube");
    const centers: ReadonlyArray<readonly [number, string]> = [
      [4, "U"],
      [13, "R"],
      [22, "F"],
      [31, "D"],
      [40, "L"],
      [49, "B"],
    ];
    for (const [pos, expected] of centers) {
      await expect(svg.locator(`rect[data-pos="${pos}"]`)).toHaveAttribute(
        "data-color",
        expected,
      );
    }
    // Every face: 9 stickers of the same letter.
    for (const letter of ["U", "R", "F", "D", "L", "B"]) {
      await expect(svg.locator(`rect[data-color="${letter}"]`)).toHaveCount(9);
    }
  });

  test("paste invalid state shows error", async ({ page }) => {
    // Scramble first to get a non-solved state.
    const reqP = page.waitForRequest("**/api/scramble");
    await page.getByTestId("scramble-button").click();
    await reqP;
    await expect(page.getByTestId("state-field")).not.toHaveValue(SOLVED, {
      timeout: 5_000,
    });
    const before = await page.getByTestId("state-field").inputValue();
    expect(before).toHaveLength(54);

    // Paste a 53-char (too short) string.
    await page.getByTestId("state-field").fill("U".repeat(53));
    await page.getByTestId("state-set").click();

    await expect(page.getByTestId("state-error")).toBeVisible();
    await expect(page.getByTestId("state-error")).toContainText("expected 54");
  });

  test("paste valid moves into moves field updates MoveList", async ({
    page,
  }) => {
    const reqP = page.waitForRequest("**/api/scramble");
    await page.getByTestId("scramble-button").click();
    await reqP;

    await page.getByTestId("moves-field").fill("R U R' U' F");
    await page.getByTestId("moves-set").click();

    await expect(page.getByTestId("moves-error")).toHaveCount(0);
    const moveItems = page.getByTestId("move-item");
    await expect(moveItems).toHaveCount(5);
    await expect(moveItems.nth(0)).toHaveText("R");
    await expect(moveItems.nth(3)).toHaveText("U'");
    await expect(moveItems.nth(4)).toHaveText("F");
  });

  test("paste invalid moves shows error", async ({ page }) => {
    const reqP = page.waitForRequest("**/api/scramble");
    await page.getByTestId("scramble-button").click();
    await reqP;

    await page.getByTestId("moves-field").fill("R2 U");
    await page.getByTestId("moves-set").click();

    await expect(page.getByTestId("moves-error")).toBeVisible();
    await expect(page.getByTestId("moves-error")).toContainText(
      "Invalid move 'R2'",
    );
  });
});
