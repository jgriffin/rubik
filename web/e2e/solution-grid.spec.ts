import { test, expect } from "@playwright/test";

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

  test("3D toggle swaps renderer between flat (rect[data-pos]) and iso (polygon[data-face])", async ({
    page,
  }) => {
    // Start card always renders — no scramble/solve needed (stub returns
    // moves=[] anyway, so sol-card-1 is unreliable; sol-card-0 uses the
    // same renderer path and is always present).
    const card = page.getByTestId("sol-card-0");
    await expect(card).toBeVisible();

    // Default: net mode → 54 flat rects, no iso polygons.
    await expect(card.locator("rect[data-pos]")).toHaveCount(54);
    await expect(card.locator("polygon[data-face]")).toHaveCount(0);

    // Click 3D.
    await page.getByTestId("render-mode-iso").click();
    await expect(page.getByTestId("render-mode-iso")).toHaveClass(/on/);

    // Iso mode: 27 polygons (3 visible faces × 9 stickers), no flat rects.
    await expect(card.locator("rect[data-pos]")).toHaveCount(0);
    await expect(card.locator("polygon[data-face]")).toHaveCount(27);

    // Click 2D — flat returns.
    await page.getByTestId("render-mode-net").click();
    await expect(page.getByTestId("render-mode-net")).toHaveClass(/on/);
    await expect(card.locator("rect[data-pos]")).toHaveCount(54);
    await expect(card.locator("polygon[data-face]")).toHaveCount(0);
  });

  test("split toggle renders both flat and iso side-by-side", async ({
    page,
  }) => {
    const card = page.getByTestId("sol-card-0");
    await expect(card).toBeVisible();

    // Click split.
    await page.getByTestId("render-mode-dual").click();
    await expect(page.getByTestId("render-mode-dual")).toHaveClass(/on/);

    // Both pair-testid renderers are present.
    await expect(card.getByTestId("flat-cube-pair")).toBeVisible();
    await expect(card.getByTestId("iso-cube-pair")).toBeVisible();

    // The card carries both flat rects and iso polygons.
    await expect(card.locator("rect[data-pos]")).toHaveCount(54);
    await expect(card.locator("polygon[data-face]")).toHaveCount(27);

    // Toggle back to net — pair testids gone, only flat rects remain.
    await page.getByTestId("render-mode-net").click();
    await expect(page.getByTestId("render-mode-net")).toHaveClass(/on/);
    await expect(card.getByTestId("flat-cube-pair")).toHaveCount(0);
    await expect(card.getByTestId("iso-cube-pair")).toHaveCount(0);
    await expect(card.locator("polygon[data-face]")).toHaveCount(0);
    await expect(card.locator("rect[data-pos]")).toHaveCount(54);
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
