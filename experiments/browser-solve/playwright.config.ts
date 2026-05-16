// Playwright config for the M11 D measurement spec. Lives next to the
// spec so the harness is self-contained and doesn't collide with the
// repo's default e2e suite (web/e2e/, web/playwright.config.ts).
//
// Run with:
//   cd web && pnpm playwright test \
//     --config ../experiments/browser-solve/playwright.config.ts \
//     --reporter=list
//
// We reuse the web/ playwright config's webServer block (Vite + FastAPI
// stub) — the App fetches /api/health on load but the ONNX solve path
// never round-trips through FastAPI, so the stub backend is fine.

import { defineConfig, devices } from "@playwright/test";
import * as path from "node:path";

const WEB_DIR = path.resolve(__dirname, "..", "..", "web");
const REAL_BACKEND = process.env.PLAYWRIGHT_REAL_BACKEND === "1";

export default defineConfig({
  testDir: __dirname,
  testMatch: /measure_browser\.spec\.ts$/,
  fullyParallel: false, // shared backend; ordered measurement is the point
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  // No global timeout — per-test timeouts are set inside the spec since
  // WASM × width=256 × 10 rows can run upwards of 40 minutes.
  timeout: 0,
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "off",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // Headed mode + the installed Chrome channel (NOT Playwright's
        // bundled headless shell). The bundled shell uses swiftshader
        // for WebGPU, which is software-rasterized and ~12× slower than
        // the Metal-backed WebGPU a real Mac user gets. Headed Chrome
        // exercises the actual GPU path we want to characterize.
        headless: false,
        channel: "chrome",
        launchOptions: {
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,UseSkiaRenderer",
          ],
        },
      },
    },
  ],
  webServer: [
    {
      // FastAPI — stub by default. The App's /api/health probe is the
      // only thing that hits this server during ONNX-path measurement.
      command: "uv run rubik-serve",
      cwd: path.resolve(WEB_DIR, ".."),
      port: 8000,
      reuseExistingServer: true,
      timeout: 30_000,
      env: REAL_BACKEND
        ? { ...process.env }
        : { ...process.env, RUBIK_USE_STUB_NET: "1" },
    },
    {
      command: "pnpm dev",
      cwd: WEB_DIR,
      port: 5173,
      reuseExistingServer: true,
      timeout: 30_000,
    },
  ],
});
