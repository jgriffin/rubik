import { test, expect } from "@playwright/test";

test("health endpoint renders on page load", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "rubik solver" })).toBeVisible();
  const json = page.getByTestId("health-json");
  await expect(json).toBeVisible();
  const text = await json.textContent();
  expect(text).toBeTruthy();
  const body = JSON.parse(text!);
  expect(body).toMatchObject({
    model_loaded: true,
    warmup_done: true,
    cube_size: 3,
  });
  // Stub mode reports "<stub-net>"; real mode reports a path ending in net_final.pt.
  expect(typeof body.model_path).toBe("string");
});
