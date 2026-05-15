# Browser-solve perf — M11 Block D

_Generated: 2026-05-15 22:23 UTC_

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
| onnx-webgpu | 32 | 10 | 2.38s | 2.03s | 4.24s | 80% | 13.5 |
| onnx-webgpu | 64 | 10 | 4.80s | 4.01s | 8.47s | 80% | 13.5 |
| onnx-webgpu | 128 | 10 | 9.77s | 8.18s | 16.20s | 100% | 15.2 |
| onnx-webgpu | 256 | 10 | 19.97s | 16.86s | 33.82s | 90% | 14.4 |
| onnx-wasm | 32 | 1 | 2.49s | 2.49s | 2.49s | 100% | 14.0 |
| onnx-wasm | 64 | 0 | — | — | — | — | — |
| onnx-wasm | 128 | 0 | — | — | — | — | — |
| onnx-wasm | 256 | 0 | — | — | — | — | — |

## Speedups (median latency at each width)

| width | FastAPI ms | WebGPU ms | WASM ms | WebGPU/FastAPI | WASM/FastAPI | WASM/WebGPU |
|---|---|---|---|---|---|---|
| 32 | 44 | 2.38s | 2.49s | 54.61x | 57.17x | 1.05x |
| 64 | 61 | 4.80s | — | 78.66x | — | — |
| 128 | 93 | 9.77s | — | 105.03x | — | — |
| 256 | 156 | 19.97s | — | 128.04x | — | — |

