# experiments/browser-solve

M11 Block D — perf characterization of the in-browser ONNX solver path
versus the FastAPI/MPS server baseline. See `plans/m11-onnx-browser.md`
for milestone context, `LOG.md` for in-progress / closed blocks.

Three measurement subjects:
1. **FastAPI/MPS** — `rubik.server.inference.solve_facelet` measured
   directly (no HTTP roundtrip), Python wall clock.
2. **ONNX/WASM** — onnxruntime-web's WASM EP, single-threaded, in headless
   Chromium via Playwright.
3. **ONNX/WebGPU** — onnxruntime-web's WebGPU EP, in headless Chromium.

## Layout

- `corpus.json` — first 10 rows of `tests/data/m11_parity_corpus_3x3.json`
  (depth-14 scrambles, seeded 0xBEEF). Reused so the same facelets that
  validated TS / Python parity in Block B are the basis for the perf
  comparison.
- `measure_fastapi.py` — runs the Python beam over the corpus × widths,
  appends JSONL to `results/latencies.jsonl`.
- `measure_browser.spec.ts` — Playwright spec that drives WASM + WebGPU
  EPs over the same corpus × widths, appends JSONL to the same file.
- `analysis/analyze.py` — reads JSONL, writes `results/results.md` with
  the per-(solver, width) table.
- `analysis/render_perf_comparison.py` — writes
  `results/perf-comparison.html` (one chart, three lines).
- `intuition.md` — hand-written observations + hypotheses (see
  CLAUDE.md "experiment results.md format").

## Run

```sh
# 1. Stage the corpus (once — reproducible from m11_parity_corpus_3x3.json).
uv run python experiments/browser-solve/stage_corpus.py

# 2. FastAPI/MPS measurements (fast — a few minutes).
uv run python experiments/browser-solve/measure_fastapi.py

# 3. Browser measurements (WASM is slow at width=256; total ~1 hr).
cd web && pnpm playwright test ../experiments/browser-solve/measure_browser.spec.ts \
  --project=chromium

# 4. Analyze + render.
uv run python experiments/browser-solve/analysis/analyze.py
uv run python experiments/browser-solve/analysis/render_perf_comparison.py
```

Open `results/perf-comparison.html` and read `intuition.md`.
