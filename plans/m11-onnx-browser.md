# M11 — ONNX export + browser inference

Forward-looking. See `ROADMAP.md` for milestone status, `LOG.md` for in-progress / closed blocks, `SPEC.md` for the project spec.

## Goal

Take our current 3x3 champion ValueNet, export to ONNX, port beam search to TypeScript, and run the full solve loop client-side in the browser via `onnxruntime-web`. The FastAPI/MPS server path stays — the UI gets a toggle, and both implementations land behind a single `Solver` abstraction so swapping the solver is one method call. End-to-end goal: a fully-static Vercel-deployable solver demo, with empirical intuition about how browser inference (WASM and WebGPU) compares to the MPS-backed server baseline.

This milestone proves out three things at once:
1. The PyTorch ValueNet survives ONNX export with numerical fidelity.
2. The beam search loop is portable to TypeScript + ONNX inference without algorithmic regression.
3. The browser perf envelope is good enough to ship as a real interactive demo.

## Locked decisions

- **Champion model.** `experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt` (the LayerNorm variant from the kmax30 100k-step run that the FastAPI server already loads). M8 may produce a better champion later — the abstraction makes swapping the `.onnx` cheap; do not block M11 on M8 progress.
- **Export format.** ONNX, opset chosen empirically (start at 17; bump only if a needed op requires it). Dynamic batch dimension. FP32 export first; FP16 deferred to perf-characterization phase.
- **Runtime.** `onnxruntime-web` npm package. WebGPU execution provider as the fast path, WASM as the universal fallback (auto-select with a manual override in the UI for diagnostic runs).
- **Solver interface.** TypeScript `Solver` with one method: `solve(state: string, opts?: SolveOptions): Promise<SolveResponse>`. Existing FastAPI-backed code refactors to a `ApiSolver` implementation; ONNX path lands as `OnnxSolver`. Both produce the same `SolveResponse` shape the UI already consumes.
- **Switch UI.** Segmented control in the web app; remembered in localStorage. Default = FastAPI until ONNX path is at parity, then default flips.
- **No retraining for ONNX-friendliness.** Use the model as-is. If a specific op blocks export, fix at the export-script level (decompose / patch); do not retrain.
- **3x3 only.** Matches the M9 scope; the 2x2 path stays Python-only.
- **No quantization in M11.** FP32 → maybe FP16. INT8 is future work.

## Phases

### M11.A — ONNX export + Python-side parity gate

**Goal.** A `.onnx` file written from the champion checkpoint, plus a Python test that loads it through `onnxruntime` (CPU EP) and confirms outputs match the PyTorch forward to a tight numerical tolerance on a corpus of states.

**Scope.**
- Add `onnx` + `onnxruntime` to the project's dev dependencies (`uv add --dev`).
- `scripts/export_onnx_3x3.py` — loads the champion `.pt`, exports `<run-dir>/net_final.onnx` with dynamic batch dim.
- `tests/onnx_parity_test_3x3.py` — loads PyTorch model + ONNX model, feeds N=1000 random states (or the standard test corpus), asserts `max|Δ| < ε` and `mean|Δ| < ε/10`.
- Record file size, opset, parity numbers in the block's Outcome.

**Out of A.** No JS/TS work, no browser, no beam-search changes. Single artifact: a portable `.onnx` that's known-equivalent to the PyTorch forward.

### M11.B — TypeScript beam search + Node-side parity

**Goal.** Port `src/rubik/search/beam.py`'s `beam_solve_batch` to TypeScript, run it under Node with `onnxruntime-node` (Node's native CPU runtime), and prove it produces identical or aggregate-equivalent solutions to the Python path on a fixed scramble corpus.

**Scope.**
- TS implementations of: move-table application (state expansion), beam top-k selection, the beam-search outer loop.
- Move-table data: either generate from `CubeSpec` on the Python side and ship as a static JSON asset, or codegen the equivalent into a TS source file. Decision deferred to phase start.
- Node-side parity runner: same scramble seeds → same solutions (or aggregate-rate-within-SE per the project's aggregate-equivalence convention for statistical algorithms — see memory `feedback_aggregate_not_cell_equivalence.md`).

**Out of B.** No browser-specific code (WebGPU/WASM EP) — that lands in C. No UI changes.

### M11.C — Solver abstraction + browser wire-up + UI switch

**Goal.** Both solver paths live behind a `Solver` interface; the web app has a toggle; the cube can be solved end-to-end in the browser.

**Scope.**
- Define `web/src/solver/Solver.ts` interface (`solve(state, opts) → SolveResponse`).
- Refactor existing `apiSolve` call site into a `ApiSolver` implementation. No behavior change.
- Implement `OnnxSolver` using the TS beam search from M11.B + `onnxruntime-web` (WebGPU EP, WASM fallback).
- Ship the `.onnx` model as a static asset under `web/public/models/`.
- UI: segmented control (FastAPI | ONNX). Persist choice in localStorage. Surface model-loading state separately from solving state (the .onnx download + WebGPU compile is one-time but visible).
- Health/diagnostic surface: show the active solver + execution provider + last solve latency.

**Out of C.** Detailed perf characterization (lands in D). Static Vercel deploy (lands in E if pursued).

### M11.D — Performance characterization

**Goal.** Earn perf intuition for the browser solver vs the MPS server baseline across beam widths, with a reproducible measurement methodology — same pattern as M4 / M8 perf work.

**Scope.**
- New experiment dir: `experiments/browser-solve/`.
- `measure.ts` (or similar) collects per-solve latency at width ∈ {32, 64, 128, 256} across three configurations (FastAPI/MPS, ONNX/WASM, ONNX/WebGPU). Same scramble corpus across all three.
- `analyze.py` + `render_perf_comparison.py` following the standard cycle-reporting pattern (RUNS table, single canonical HTML).
- `intuition.md` with Observations → Hypotheses (with verification plans) → Open questions, per the project convention.
- FP16 model variant lands here if FP32 perf disappoints — empirical decision, not pre-committed.

### M11.E — Static Vercel deploy (optional, can defer)

**Goal.** Push the no-server build to Vercel; demonstrate the end-to-end story works without infra.

**Scope.**
- Vercel project setup (name `rubik` per the cloud-resource-naming convention).
- Build verification: `pnpm build` produces `dist/` with the `.onnx` asset and works when served as flat files.
- Documentation: README pointer + a short writeup on the artifact in the milestone close.

## Out of scope (entire M11)

- Retraining the model with ONNX-friendly choices.
- 2x2 ONNX export.
- INT8 quantization.
- WebGL execution provider — WASM and WebGPU only.
- Solver-implementation switching at the Python level (the Python beam search stays the only Python path; the abstraction is on the TypeScript side).

## Risks / unknowns

1. **Op support.** LayerNorm + Linear + ReLU should all export cleanly at opset 17. If the architecture has anything non-standard, export will surface it in A·P1.
2. **Model size.** A `[5120, 1024]×4 bn LN` net is meaningfully large in FP32 (back-of-envelope: tens of MB). First-load UX may want a banner. FP16 may be required for a comfortable demo.
3. **WebGPU availability.** Solid on recent Chrome and Safari; Firefox still flag-gated in some channels. The WASM fallback is the safety net but is meaningfully slower; perf phase characterizes the gap.
4. **JS top-k at scale.** Beam search top-k at width=256 over thousands of children per step is unbothered in NumPy but needs care in JS. May need a tuned implementation (partial sort, typed arrays).
5. **Cross-EP numerical drift.** WebGPU vs WASM may produce slightly different floats. Aggregate-equivalence is the right gate (per project convention).

## Repo layout (planned)

```
scripts/
  export_onnx_3x3.py                # PyTorch .pt → .onnx
tests/
  onnx_parity_test_3x3.py           # numerical equivalence test
experiments/browser-solve/
  measure.ts | measure.py
  results/                          # JSONL, HTML
  intuition.md
  analysis/
web/public/models/
  rubik-3x3-valuenet.onnx
web/src/solver/
  Solver.ts                         # interface
  ApiSolver.ts                      # FastAPI-backed
  OnnxSolver.ts                     # in-browser ONNX
  beam.ts                           # TS port of beam_solve_batch
  moveTables.ts | moveTables.json   # sticker permutation data
plans/m11-onnx-browser.md           # this file
plans/m11-block-a-onnx-export.md    # written if Block A wants a sub-plan
```

## References

- ONNX Runtime Web: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
- WebGPU execution provider: https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html
- `torch.onnx` exporter: https://pytorch.org/docs/stable/onnx.html
- Beam search reference impl: `src/rubik/search/beam.py`
- Server reference impl: `src/rubik/server/`
- Champion checkpoint: `experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt`
