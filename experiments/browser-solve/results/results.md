# Browser-solve perf — M11 Block D

_Generated: 2026-05-15 22:36 UTC_

## Run conditions

- Hardware: Apple M4 Max (user's dev machine).
- Model: `experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt` (champion 3x3 ValueNet, ~234M params).
- ONNX export: `web/public/models/net_final.onnx` (M11 Block A export, parity-verified).
- Corpus: first 10 rows of `tests/data/m11_parity_corpus_3x3.json` (depth-14 scrambles, seed 0xBEEF). Same facelets used in Block B parity gate.
- Beam: `max_steps=22`; widths swept ∈ {32, 64, 128, 256}.
- FastAPI path: direct Python (`solve_facelet`) — no HTTP round-trip. MPS warmed at every width before measurement.
- Browser path: Playwright-driven Chromium, ONNX Runtime Web (WASM + WebGPU EPs). One page-load per (ep, width) so model-load + EP-warmup amortize across rows.

## Per-(solver, width) summary

| solver | width | n | median ms | p10 ms | p90 ms | solve_rate | mean solve_len |
|---|---|---|---|---|---|---|---|
| fastapi | 32 | 10 | 44 | 38 | 68 | 80% | 13.5 |
| fastapi | 64 | 10 | 61 | 54 | 95 | 80% | 13.5 |
| fastapi | 128 | 10 | 93 | 80 | 148 | 100% | 15.2 |
| fastapi | 256 | 10 | 156 | 132 | 247 | 90% | 14.4 |
| onnx-webgpu | 32 | 10 | 117 | 97 | 248 | 80% | 13.5 |
| onnx-webgpu | 64 | 10 | 208 | 174 | 401 | 80% | 13.5 |
| onnx-webgpu | 128 | 10 | 394 | 330 | 655 | 100% | 15.2 |
| onnx-webgpu | 256 | 10 | 775 | 644 | 1.29s | 90% | 14.4 |
| onnx-wasm | 32 | 10 | 2.54s | 2.11s | 4.47s | 80% | 13.5 |
| onnx-wasm | 64 | 10 | 5.06s | 4.23s | 8.82s | 80% | 13.5 |
| onnx-wasm | 128 | 10 | 10.38s | 8.62s | 17.23s | 100% | 15.2 |
| onnx-wasm | 256 | 10 | 20.18s | 16.82s | 34.14s | 90% | 14.4 |

## Speedups (median latency at each width)

| width | FastAPI ms | WebGPU ms | WASM ms | WebGPU/FastAPI | WASM/FastAPI | WASM/WebGPU |
|---|---|---|---|---|---|---|
| 32 | 44 | 117 | 2.54s | 2.69x | 58.38x | 21.71x |
| 64 | 61 | 208 | 5.06s | 3.42x | 82.88x | 24.25x |
| 128 | 93 | 394 | 10.38s | 4.24x | 111.61x | 26.31x |
| 256 | 156 | 775 | 20.18s | 4.97x | 129.35x | 26.04x |

