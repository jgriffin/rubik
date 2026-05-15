// M11 Block D — Playwright-driven browser solve measurement.
//
// Drives the actual UI through both WASM and WebGPU execution providers
// across widths {32, 64, 128, 256} × first 10 rows of corpus.json (the
// Block B parity corpus). Per (ep, width), navigates ONCE to
// `/?ep={ep}&width={width}` so model-load + EP-warmup amortize across
// the 10 rows for that width. Emits the same JSONL row shape as
// measure_fastapi.py — appends to results/latencies.jsonl.
//
// Lives under experiments/browser-solve/ (NOT web/e2e/) so the
// pnpm test:e2e default doesn't pick it up. Run explicitly:
//
//   cd web && pnpm playwright test \
//     ../experiments/browser-solve/measure_browser.spec.ts \
//     --project=chromium --reporter=list --timeout=600000
//
// WASM at width=256 is multi-minute per solve × 10 rows × 4 widths,
// so the WASM test pad is generous. WebGPU is much faster.

import { expect, test } from "@playwright/test";
import * as fs from "node:fs";
import * as path from "node:path";

const EXPERIMENT_DIR = path.resolve(__dirname);
const CORPUS_PATH = path.join(EXPERIMENT_DIR, "corpus.json");
const OUT_PATH = path.join(EXPERIMENT_DIR, "results", "latencies.jsonl");
const SOLVED_3X3 =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);
const WIDTHS = [32, 64, 128, 256];

type CorpusRow = { row_idx: number; facelet: string; py_solve_len: number };
type Corpus = { rows: CorpusRow[] };

function readCorpus(): CorpusRow[] {
  const raw = fs.readFileSync(CORPUS_PATH, "utf8");
  const obj = JSON.parse(raw) as Corpus;
  return obj.rows;
}

function appendRec(rec: Record<string, unknown>): void {
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.appendFileSync(OUT_PATH, JSON.stringify(rec) + "\n");
}

// Map URFDLB-ordered facelet (54 chars) to the six 9-char chunks the
// StateGrid distributes by offset. We push the whole 54-char string
// into the U-face input — the StateGrid's paste handler recognizes a
// full-state paste and distributes by offset.
async function setFacelet(
  page: import("@playwright/test").Page,
  facelet: string,
): Promise<void> {
  expect(facelet.length).toBe(54);
  // First clear: press the clear button to ensure the grid is at solved
  // (avoids any residual state from a prior solve appending moves).
  await page.getByTestId("clear-button").click();
  await expect(page.getByTestId("state-input-U")).toHaveValue("UUUUUUUUU");
  // The StateGrid's onPaste handler treats a 54-char clipboard payload
  // as a full-state URFDLB distribution. We can't directly call paste
  // via .fill() — Playwright's evaluate dispatches a real paste event
  // so the handler fires.
  const target = page.getByTestId("state-input-U");
  await target.focus();
  await page.evaluate(
    ({ facelet }) => {
      const el = document.querySelector(
        "[data-testid='state-input-U']",
      ) as HTMLInputElement | null;
      if (!el) throw new Error("state-input-U not found");
      el.focus();
      const dt = new DataTransfer();
      dt.setData("text/plain", facelet);
      const ev = new ClipboardEvent("paste", {
        clipboardData: dt,
        bubbles: true,
        cancelable: true,
      });
      el.dispatchEvent(ev);
    },
    { facelet },
  );
  // Confirm the U-face value reflects the new state (first 9 chars).
  await expect(page.getByTestId("state-input-U")).toHaveValue(facelet.slice(0, 9));
}

async function selectOnnx(page: import("@playwright/test").Page): Promise<void> {
  await page.getByTestId("solver-onnx").click();
  // Wait for the ONNX session to be ready — solver-loading paragraph
  // disappears when info().ready becomes true.
  await page
    .getByTestId("solver-loading")
    .waitFor({ state: "detached", timeout: 180_000 });
}

async function runOneSolve(
  page: import("@playwright/test").Page,
  row: CorpusRow,
): Promise<{ wall_ms: number; solve_len: number; solved: boolean }> {
  await setFacelet(page, row.facelet);
  // Solve button lives as the trailing cell of MovesGrid.
  const solveBtn = page.getByTestId("solve-button");
  await expect(solveBtn).toBeEnabled({ timeout: 30_000 });

  // Capture solve completion via a sentinel: solved-footer appears (when
  // solved=true) OR the solve-button becomes enabled again (when
  // solved=false / unsolved-but-completed). We time via JS Date.now()
  // around the click → completion window so unsolved cases still get a
  // wall_ms.
  const t0 = Date.now();
  await solveBtn.click();
  // Wait until isSolving has flipped back to false. The solve button is
  // disabled during isSolving=true (App.tsx: disabled={!ready || isSolving});
  // when it flips, either (a) the cube is solved → SolvedFooter visible,
  // or (b) the cube is still scrambled → the button re-enables.
  await page.waitForFunction(
    () => {
      const footer = document.querySelector("[data-testid='solved-footer']");
      if (footer) return true;
      const btn = document.querySelector(
        "[data-testid='solve-button']",
      ) as HTMLButtonElement | null;
      return btn != null && !btn.disabled;
    },
    null,
    { timeout: 1_800_000 }, // 30 min upper bound — WASM × width=256 worst-case
  );
  const t1 = Date.now();
  const clickWallMs = t1 - t0;

  const footer = page.getByTestId("solved-footer");
  const solved = await footer.isVisible().catch(() => false);
  let solverWallMs = -1;
  let moveCount = -1;
  if (solved) {
    const metaText =
      (await footer.locator(".meta").textContent().catch(() => "")) ?? "";
    const m = /(\d+)\s*ms/.exec(metaText);
    if (m) solverWallMs = Number.parseInt(m[1], 10);
    moveCount = await page
      .getByTestId("moves-grid")
      .locator('[data-testid="move-cell"]')
      .count();
  }
  // Prefer the solver's self-reported time_ms (high-resolution
  // performance.now()) when available — matches what FastAPI records.
  // Fall back to click-to-completion wall for unsolved cases.
  const wall_ms = solverWallMs > 0 ? solverWallMs : clickWallMs;

  return { wall_ms, solve_len: solved ? moveCount : -1, solved };
}

// One-shot WebGPU adapter check. Aborts the WebGPU runs if the
// browser only exposes swiftshader (software WebGPU) — those numbers
// would not represent what a real user with Metal/Vulkan-backed
// WebGPU experiences. Headed Chrome on macOS should expose an "Apple"
// vendor adapter; software fallback shows "SwiftShader" or "WARP".
test("webgpu adapter sanity", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/");
  await page.getByTestId("solver-switch").waitFor({ timeout: 30_000 });
  const info = await page.evaluate(async () => {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    if (!gpu) return { ok: false, reason: "navigator.gpu undefined" };
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "requestAdapter returned null" };
    // adapterInfo is the modern API; fall back to legacy fields.
    type AdapterInfo = { vendor?: string; architecture?: string; device?: string };
    const ainfo = ("info" in adapter ? (adapter as unknown as { info: AdapterInfo }).info : {}) as AdapterInfo;
    return {
      ok: true,
      vendor: ainfo.vendor ?? "(unknown)",
      architecture: ainfo.architecture ?? "(unknown)",
      device: ainfo.device ?? "(unknown)",
    };
  });
  console.log("[webgpu adapter]", JSON.stringify(info));
  // Hard-fail if software fallback. We want headed Chrome's real WebGPU.
  const vendorLc = String((info as { vendor?: string }).vendor ?? "").toLowerCase();
  const isSoftware =
    vendorLc.includes("swiftshader") || vendorLc.includes("software") ||
    vendorLc.includes("warp") || vendorLc.includes("microsoft basic render driver");
  expect(info.ok, "WebGPU adapter must be available").toBe(true);
  expect(isSoftware, `WebGPU using software fallback (vendor=${vendorLc}) — re-run with channel=chrome headed`).toBe(false);
});

for (const ep of ["webgpu", "wasm"] as const) {
  for (const width of WIDTHS) {
    test(`measure ep=${ep} width=${width}`, async ({ page }, testInfo) => {
      if (ep === "webgpu" && testInfo.project.name !== "chromium") test.skip();
      // Generous timeout — WASM × width=256 × 10 rows can run ~20 min.
      test.setTimeout(2_400_000); // 40 min per (ep, width) cell

      const rows = readCorpus();
      await page.goto(`/?ep=${ep}&width=${width}`);
      // Wait for the solver-switch to render (means health resolved).
      await page.getByTestId("solver-switch").waitFor({ timeout: 30_000 });
      await selectOnnx(page);

      const solver = `onnx-${ep}`;
      const cellStart = Date.now();
      for (const row of rows) {
        const r = await runOneSolve(page, row);
        appendRec({
          solver,
          width,
          row_idx: row.row_idx,
          facelet: row.facelet,
          wall_ms: r.wall_ms,
          solve_len: r.solve_len,
          solved: r.solved,
        });
        console.log(
          `[${solver} w=${width}] row=${row.row_idx} wall=${r.wall_ms}ms solved=${r.solved} len=${r.solve_len}`,
        );
      }
      const cellWall = ((Date.now() - cellStart) / 1000).toFixed(1);
      console.log(`[${solver} w=${width}] 10 rows total wall=${cellWall}s`);
    });
  }
}
