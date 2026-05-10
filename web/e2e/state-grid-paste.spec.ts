import { test, expect } from "@playwright/test";

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

test.describe("state grid — paste + auto-advance", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("scramble-button")).toBeEnabled();
  });

  test("paste solved URFDLB state into U input fills all 6 faces", async ({ page }) => {
    // Scramble first so inputs aren't already solved.
    const reqP = page.waitForRequest("**/api/scramble");
    await page.getByTestId("scramble-button").click();
    await reqP;
    // Wait until U input value is no longer the initial "UUUUUUUUU".
    await expect(page.getByTestId("state-input-U")).not.toHaveValue("UUUUUUUUU", {
      timeout: 5_000,
    });

    // Simulate paste of full URFDLB string into the U input.
    const uInput = page.getByTestId("state-input-U");
    await uInput.click();
    await uInput.evaluate((el: HTMLInputElement, payload: string) => {
      const dt = new DataTransfer();
      dt.setData("text/plain", payload);
      el.dispatchEvent(new ClipboardEvent("paste", { clipboardData: dt, bubbles: true }));
    }, SOLVED);

    // Each face input now reflects its 9-char slice — display order U L F R B D,
    // storage offsets 0 / 36 / 18 / 9 / 45 / 27 → all "XXXXXXXXX".
    await expect(page.getByTestId("state-input-U")).toHaveValue("UUUUUUUUU");
    await expect(page.getByTestId("state-input-L")).toHaveValue("LLLLLLLLL");
    await expect(page.getByTestId("state-input-F")).toHaveValue("FFFFFFFFF");
    await expect(page.getByTestId("state-input-R")).toHaveValue("RRRRRRRRR");
    await expect(page.getByTestId("state-input-B")).toHaveValue("BBBBBBBBB");
    await expect(page.getByTestId("state-input-D")).toHaveValue("DDDDDDDDD");

    // sol-card-0 (the start card) renders the solved cube after the paste.
    const startSvg = page.getByTestId("sol-card-0").locator("svg").first();
    const centers: ReadonlyArray<readonly [number, string]> = [
      [4, "U"],
      [13, "R"],
      [22, "F"],
      [31, "D"],
      [40, "L"],
      [49, "B"],
    ];
    for (const [pos, expected] of centers) {
      await expect(startSvg.locator(`rect[data-pos="${pos}"]`)).toHaveAttribute(
        "data-color",
        expected,
      );
    }
  });

  test("typing 9 chars in U auto-advances focus to L", async ({ page }) => {
    const u = page.getByTestId("state-input-U");
    await u.click();
    await u.fill("");
    await u.type("UUUUUUUUU");
    await expect(page.getByTestId("state-input-L")).toBeFocused();
  });

  test("backspace at empty L jumps focus back to U", async ({ page }) => {
    const l = page.getByTestId("state-input-L");
    await l.click();
    await l.fill("");
    await l.press("Backspace");
    await expect(page.getByTestId("state-input-U")).toBeFocused();
  });
});
