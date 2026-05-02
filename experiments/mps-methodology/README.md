# MPS measurement methodology (M4 Max)

## Purpose

Codify how we measure perf on this M4 Max so subsequent milestones
(M5/M7/M8) reuse one trusted protocol instead of each reinventing
timers, sync points, and saturation checks.

## Tools

- **`rubik.perf.bench.time_op`** — primary truth. Warmup → MPS sync →
  `perf_counter` → `fn()` → MPS sync → `perf_counter`. Everything else
  is investigation around this number.
- **`macmon`** — external observer. Independent GPU-busy-fraction
  signal we cross-check against `time_op` throughput; closes the
  "are we actually on the GPU?" question.
- **`torch.profiler`** — investigation tool, not measurement tool.
  Use to find the culprit op when bench and macmon disagree. MPS
  fidelity on torch 2.11 is itself something we probe here.
- **`mactop`** — live monitoring during dev iteration; not part of the
  recorded pipeline.
- **`hyperfine`** — end-to-end CLI benchmarking when scripts/binaries
  are the unit of measurement (not in-process ops).

## How to reproduce

See `results.md` once probes land — the decision-point output for
profiler fidelity on torch 2.11 lives there, plus the canonical
command list.
