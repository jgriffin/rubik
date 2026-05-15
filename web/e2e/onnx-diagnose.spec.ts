// Diagnostic — not a pass/fail test in the traditional sense.
// Switches to the ONNX solver, polls the loading indicator every 2s
// for up to 90s, and reports timings + a count of net_final.onnx
// fetches (a second fetch would mean the OnnxSolver was re-instantiated
// mid-download — the bug class we just fixed).
//
// Run with: `pnpm test:e2e e2e/onnx-diagnose.spec.ts --project=chromium`.

import { expect, test } from "@playwright/test";

const POLL_MS = 2000;
const TIMEOUT_MS = 35_000;

async function diagnose(page: import("@playwright/test").Page, ep: "wasm" | "webgpu") {
  let modelFetches = 0;
  const modelEvents: string[] = [];
  page.on("request", (req) => {
    const u = req.url();
    if (u.includes("/models/net_final.onnx")) {
      modelFetches += 1;
      modelEvents.push(`REQ ${u.split("/").pop()}`);
    }
  });
  page.on("response", (res) => {
    const u = res.url();
    if (u.includes("/models/net_final.onnx")) {
      modelEvents.push(`RES ${u.split("/").pop()} ${res.status()}`);
    }
  });
  page.on("requestfailed", (req) => {
    const u = req.url();
    if (u.includes("/models/net_final.onnx")) {
      modelEvents.push(`FAIL ${u.split("/").pop()} — ${req.failure()?.errorText ?? "(unknown)"}`);
    }
  });
  page.on("requestfinished", (req) => {
    const u = req.url();
    if (u.includes("/models/net_final.onnx")) {
      void req.response().then(async (r) => {
        const sz = (await r?.body().catch(() => null))?.byteLength ?? -1;
        modelEvents.push(`FIN ${u.split("/").pop()} bytes=${sz}`);
      });
    }
  });
  page.on("console", (msg) => {
    const text = msg.text();
    // Capture anything that looks like an ort/error message.
    if (
      text.toLowerCase().includes("ort") ||
      text.toLowerCase().includes("onnx") ||
      msg.type() === "error" ||
      msg.type() === "warning"
    ) {
      console.log(`  [console.${msg.type()}] ${text}`);
    }
  });
  page.on("pageerror", (err) => {
    console.log(`  [pageerror] ${err.message}`);
  });

  console.log(`\n=== EP=${ep} ===`);
  const t0 = Date.now();
  await page.goto(`/?ep=${ep}`);
  // Wait for health to resolve so the solver switch is enabled.
  await page.waitForFunction(() => {
    const tag = document.querySelector("[data-testid='solver-switch']");
    return !!tag;
  }, { timeout: 15_000 });

  const onnxClickAt = Date.now() - t0;
  console.log(`[t=${onnxClickAt}ms] clicking solver-onnx`);
  await page.click("[data-testid='solver-onnx']");

  let lastLoadingText: string | null = null;
  let loadingClearedAt: number | null = null;
  const deadline = Date.now() + TIMEOUT_MS;

  while (Date.now() < deadline) {
    const loadingEl = page.locator("[data-testid='solver-loading']");
    const visible = await loadingEl.isVisible().catch(() => false);
    const elapsed = Date.now() - t0;
    if (!visible) {
      loadingClearedAt = elapsed;
      console.log(`[t=${elapsed}ms] LOADING INDICATOR GONE (ready)`);
      break;
    }
    const text = (await loadingEl.textContent().catch(() => null)) ?? "(empty)";
    if (text !== lastLoadingText) {
      console.log(`[t=${elapsed}ms] loading: ${text}`);
      lastLoadingText = text;
    }
    await page.waitForTimeout(POLL_MS);
  }

  if (loadingClearedAt == null) {
    console.log(`[t=${TIMEOUT_MS}ms] TIMED OUT — loading indicator still visible`);
  }

  console.log(`net_final.onnx fetches observed: ${modelFetches}`);
  console.log(`model events:`);
  for (const e of modelEvents) console.log(`  ${e}`);
  console.log(
    `loading-clear at ${loadingClearedAt ?? "(never)"} ms; fetches=${modelFetches}`,
  );

  // If the loading cleared, try a scramble+solve so we can capture the
  // end-to-end timing too. Stub-backend scramble is instant; the solve
  // exercises the actual ONNX path.
  if (loadingClearedAt != null) {
    const scrambleBtn = page.locator("[data-testid='scramble-button']").or(
      page.getByRole("button", { name: /scramble/i }),
    );
    if (await scrambleBtn.isVisible().catch(() => false)) {
      await scrambleBtn.click();
      await page.waitForTimeout(500);
      const solveBtn = page.locator("[data-testid='solve-button']").or(
        page.getByRole("button", { name: /^solve/i }),
      );
      if (await solveBtn.isVisible().catch(() => false)) {
        const tSolve0 = Date.now();
        await solveBtn.click();
        // Wait up to 30s for solve to finish (look for solved footer text or
        // for the meta line to gain a `ms` suffix).
        await page.waitForFunction(() => {
          const txt = document.body.innerText;
          return /\d+ ms/.test(txt);
        }, { timeout: 30_000 }).catch(() => null);
        const tSolveDt = Date.now() - tSolve0;
        console.log(`[solve completed in ${tSolveDt}ms after click]`);
      } else {
        console.log("solve button not found — skipping solve timing");
      }
    } else {
      console.log("scramble button not found — skipping solve timing");
    }
  }

  // Two fetches expected: the .onnx graph (60 KB) + the .onnx.data
  // sidecar (61 MB). Three or more means the OnnxSolver was
  // re-instantiated mid-load (the recreate bug class).
  expect(modelFetches, "model + sidecar = 2 fetches expected").toBeLessThanOrEqual(2);
  expect(loadingClearedAt, "loading indicator must clear before timeout").not.toBeNull();
}

test("ONNX load diagnostic (wasm)", async ({ page }) => {
  test.setTimeout(120_000);
  await diagnose(page, "wasm");
});

test("ONNX load diagnostic (webgpu)", async ({ page, browserName }) => {
  test.setTimeout(180_000);
  if (browserName !== "chromium") test.skip();
  await diagnose(page, "webgpu");
});
