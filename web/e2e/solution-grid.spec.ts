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

  test("v2 toggles are disabled (1-column, 3D, split)", async ({ page }) => {
    await expect(page.getByTestId("columns-1")).toBeDisabled();
    await expect(page.getByTestId("render-mode-iso")).toBeDisabled();
    await expect(page.getByTestId("render-mode-dual")).toBeDisabled();
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
