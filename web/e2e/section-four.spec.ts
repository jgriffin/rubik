import { test, expect } from "@playwright/test";

// Section iv "watch the solve" smoke test. Real-backend only: SectionFour
// returns null while solutionAlg is empty, and stub mode's solve returns
// moves=[] which keeps it null forever. Only a real model produces a
// non-empty solution that flips section iv into existence.

test.describe("section iv — watch the solve", () => {
  test.skip(
    process.env.PLAYWRIGHT_REAL_BACKEND !== "1",
    "set PLAYWRIGHT_REAL_BACKEND=1 to run — needs a non-empty solution",
  );

  test("appears post-solve with embedded twisty-player; hidden pre-solve and post-clear", async ({
    page,
  }) => {
    await page.goto("/");

    // Wait for warmup: scramble button enables once health.warmup_done=true.
    await expect(page.getByTestId("scramble-button")).toBeEnabled({
      timeout: 30_000,
    });

    // Section iv hidden on initial load (no solution → SectionFour returns null).
    await expect(page.getByTestId("section-four")).toHaveCount(0);

    // Scramble — still no solution yet.
    await page.getByTestId("scramble-button").click();
    await expect(page.getByTestId("section-four")).toHaveCount(0);

    // Solve — SectionFour mounts once the model returns a non-empty solution.
    await page.getByTestId("solve-button").click();
    await expect(page.getByTestId("section-four")).toBeVisible({
      timeout: 60_000,
    });

    // Embedded twisty-player wrapper exists inside section iv.
    await expect(page.getByTestId("twisty-section-four")).toBeVisible();

    // Clear → solution resets to null → SectionFour unmounts.
    await page.getByTestId("clear-button").click();
    await expect(page.getByTestId("section-four")).toHaveCount(0);
  });
});
