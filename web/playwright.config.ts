import { defineConfig, devices } from "@playwright/test";

const REAL_BACKEND = process.env.PLAYWRIGHT_REAL_BACKEND === "1";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // shared backend
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      // FastAPI backend — stub by default (fast); real model when env flag set.
      command: "uv run rubik-serve",
      cwd: "..",
      port: 8000,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
      env: REAL_BACKEND
        ? { ...process.env } // inherit, no stub
        : { ...process.env, RUBIK_USE_STUB_NET: "1" },
    },
    {
      command: "pnpm dev",
      port: 5173,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
  ],
});
