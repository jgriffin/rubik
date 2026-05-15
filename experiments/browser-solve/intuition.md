# Browser-solve perf — intuition

_Run date 2026-05-15. Hardware: Apple M4 Max (user dev machine). Model: 3x3 ValueNet (M8 champion, kmax30 100k-step, LN). ONNX export: opset 18, FP32, 61 MB co-located graph + sidecar. Browser: headed Chrome via Playwright (channel: chrome) with Metal-3 WebGPU adapter (verified by `webgpu adapter sanity` test). Corpus: 10 rows of depth-14 scrambles, seed `0xBEEF` (Block B parity corpus). N=10 × 1 rep per (solver, width); no statistical replication._

## Observations

1. **FastAPI/MPS scales sub-linearly with beam width.** Median latency at widths {32, 64, 128, 256} is {44, 61, 93, 156} ms — a 3.5× growth over an 8× width range. The curve flattens; per-state cost FALLS as the beam widens.
2. **ONNX/WebGPU scales near-linearly.** Median latencies are {117, 208, 394, 775} ms — a 6.6× growth over the same 8× width range. Curve is close to linear in log-log space.
3. **ONNX/WASM scales linearly.** Median latencies are {2540, 5081, 10380, 20178} ms — a 7.9× growth, essentially 1:1 with beam width.
4. **FastAPI/MPS is the fastest path at every width.** It is 2.7× → 5.0× faster than WebGPU across the width sweep — the lead WIDENS with beam width.
5. **WebGPU is ~22× faster than WASM at every width.** Ratios are 21.7×, 24.3×, 26.3×, 26.0× — basically constant.
6. **Mean solve length is identical across paths at each width** (13.5, 13.5, 15.2, 14.4 moves at widths 32 / 64 / 128 / 256). The three solvers are producing equivalent solutions — Block B's per-row parity holds at the full width sweep.
7. **Solve rate rises with width** (8/10 → 8/10 → 10/10 → 9/10). The two unsolved cells at width=32 and width=64 are the same depth-14 scrambles that need a wider beam. Wider beams help, but width=256 has one regression vs width=128 — a 1-row noise floor at N=10.

## Hypotheses

### H1. The MPS sub-linear scaling = fixed per-call overhead, not flop count

**Claim.** At low widths, FastAPI/MPS latency is dominated by Python beam-loop + MPS kernel launch + host↔device sync, not by net-forward FLOPs. At width=256 these fixed costs amortize and per-state latency drops.

**Confidence:** medium. The observed 3.5× growth over an 8× width sweep is consistent with overhead-bounded behavior. The matching memory `project_mps_throughput_knee_m4_max.md` shows net throughput is flat from 1.5k → 153k states then crashes above ~200k — confirming MPS forward is overhead-bounded at small batches and bandwidth-bounded at large ones.

**Verification.** `torch.profiler` on a single beam step at width=32 vs width=256. Break out: Python beam-loop time, `apply_all_moves`, `net(states)` forward, `is_solved`, dedup. Expect the non-net components to be a large fraction at width=32 and a small fraction at width=256.

### H2. WebGPU is compute-bound on this M4 Max at all measured widths

**Claim.** WebGPU's near-linear scaling means per-state cost is roughly constant across widths. The GPU is doing actual work proportional to batch size, not waiting on shader launch / pipeline state.

**Confidence:** medium-low. The 6.6× scaling for 8× width is close to ideal (8×) but not at the bandwidth-bound knee. May be a mix of compute + per-step kernel launch.

**Verification.** Compute throughput (states-per-second) at each width:
- w=32: 32 × ~14 steps / 117 ms ≈ 3.8k states/sec
- w=64: 64 × ~14 / 208 ≈ 4.3k
- w=128: 128 × ~15 / 394 ≈ 4.9k
- w=256: 256 × ~14 / 775 ≈ 4.6k

Throughput is creeping up — consistent with kernel launch overhead amortizing, but not by much. If WebGPU were purely compute-bound, throughput would be flat. If it were purely overhead-bound, throughput would scale linearly. Reality is closer to compute-bound. Add a w=512 measurement: if throughput keeps rising, kernel-launch overhead is still a factor.

### H3. WASM is single-threaded compute-bound on the linear algebra kernels

**Claim.** WASM's perfectly linear scaling (~8× for 8× width) plus its consistent 22-26× gap to WebGPU is "GPU vs single core CPU on the same workload."

**Confidence:** medium-high. Numbers are within the expected range for headless WASM SIMD on a single core. The current `OnnxSolver` pins `numThreads=1` per the `configureWasmEnvOnce()` call — single-threaded is what we're measuring.

**Verification.** Flip `ort.env.wasm.numThreads = navigator.hardwareConcurrency` (the COOP/COEP headers are already set in `vite.config.ts` so SharedArrayBuffer is available). If pthreads close the gap to ~3-5× of WebGPU, the CPU isn't the bottleneck — the single-threaded constraint was. If they close it to ~10-15×, threading helps but doesn't close the GPU gap.

### H4. The MPS:WebGPU gap is mostly the M4 Max GPU advantage, not a software stack difference

**Claim.** Both MPS and WebGPU run on the SAME GPU silicon. PyTorch/MPS has had years of optimization for Apple Silicon; onnxruntime-web's WebGPU EP is much newer. The 2.7-5× gap likely reflects software maturity, not hardware capability.

**Confidence:** low (this is the most speculative hypothesis). Two confounds: (a) FastAPI runs Python+PyTorch+MPS which has its own per-call overhead; (b) WebGPU goes through ANGLE → Metal whereas MPS is direct Metal.

**Verification.** Strip the Python beam loop and time JUST `net(states)` in PyTorch/MPS at the same batch sizes WebGPU sees per step (e.g. width=128 → batch ≈ 1536). Compare to WebGPU's per-step throughput. If MPS's pure-forward latency at batch=1536 is much smaller than WebGPU's, software stack maturity is the explanation. If they're similar, the Python overhead is contributing more than I'm assuming.

## Open questions

1. **Does multi-threaded WASM close the gap to WebGPU?** Flip `numThreads`, re-measure. Cheap (~30 min wall on the existing harness), high information value — tells us whether ONNX/WASM can be a "good enough" fallback for users without WebGPU or whether it's fundamentally too slow.
2. **What is the first-load cost?** This run amortizes warmup across the 10-row cell. The first solve on a fresh page-load includes: 61 MB model download, WebGPU shader compile, EP init. Block C eyeballed ~3-10s on a fresh hard-reload. A dedicated experiment that measures *just* the first-solve latency per (EP, network condition) would tell us whether the demo UX is bottlenecked on load or on solve.
3. **Does FP16 shift the equilibrium?** Halving weight size means a 30 MB download (vs 61 MB). On a slow connection this could swing the user-experience comparison even if per-solve latency doesn't budge. Implementation cost is small (add `--fp16` flag to `scripts/export_onnx_3x3.py`).
4. **What's the per-step state-throughput on WebGPU at width=512?** Hypothesis H2 predicts throughput would keep creeping up if overhead is still a factor; flat if compute-bound. A 5-min single-cell extension run could resolve this.

## What we haven't verified

- **No statistical replication.** N=10 with 1 rep is enough to see qualitative behavior but the p10/p90 spreads are wide (e.g. WASM w=128 ranges 8.6s → 17.4s, 2× variation; WebGPU w=256 ranges 644 → 1.29s). Any specific production decision should re-run with N=50 × 3 reps to get tight error bars.
- **No cold-start measurement.** Browser cold-load + WebGPU compile is excluded by design here (the spec navigates once per cell and lets the loading indicator clear before measuring). The "first impression" of the ONNX path could be quite different.
- **MPS warmup is implicit, not controlled.** `measure_fastapi.py` calls `net.eval()` at startup but the very first MPS kernel launch still pays a one-time JIT compile. The 10-row cell includes that compile in row 0's measurement; subsequent rows are warm. Same issue affects the browser EPs (per-cell warmup is implicit in the page-load and first solve). For "what does the user feel" the warmup matters; for "what's the underlying compute capability" it should be excluded.
- **Single-machine snapshot.** All numbers are from one M4 Max with the user's Chrome version on 2026-05-15. WebGPU drivers and onnxruntime-web releases shift these numbers. The intuition is for THIS machine on THIS day.
