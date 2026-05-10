import { test, expect } from "@playwright/test";

test("health resolves: wordmark visible + scramble enabled", async ({ page }) => {
  await page.goto("/");

  // Wordmark renders the heading.
  await expect(page.getByRole("heading", { name: "rubik solver" })).toBeVisible();

  // Scramble enables only when /api/health resolves with warmup_done=true.
  // This is the load-bearing assertion that the backend connection works —
  // the new design intentionally drops the dev-mode health JSON dump from
  // the prior layout.
  await expect(page.getByTestId("scramble-button")).toBeEnabled({
    timeout: 10_000,
  });
});
