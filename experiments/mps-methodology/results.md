# MPS measurement methodology (M4 Max)

*M4 deliverable. Living document — updated as probes confirm.*

## 1. Setup

- **Machine.** Apple M4 Max (this user's box). Unified memory; one GPU
  visible to PyTorch through Apple's Metal Performance Shaders backend.
- **PyTorch.** `torch == 2.11.0`. `torch.backends.mps.is_available()` and
  `is_built()` both `True`. `ProfilerActivity` enum exposes
  `{CPU, XPU, MTIA, CUDA, HPU, PrivateUse1}` — **`MPS` is NOT a member.**
  This is the version we measured against; if it lands in a later torch
  release, this section gets revisited.
- **Methodology version.** 1 — 2026-05-02. Probes:
  - `experiments/mps-methodology/probe_profiler.py`
  - `experiments/mps-methodology/verify_no_cpu_sync.py`
  - macmon probe + correlator land in commit 5.

## 2. Triangulation strategy

We trust no single tool. Three layers cross-check each other:

### Layer 1 — bracket-sync timing (`bench.py`) — *primary truth*

`mps.synchronize` → `perf_counter` → `fn()` → `mps.synchronize` →
`perf_counter`. Code at `src/rubik/perf/bench.py`. Returns per-trial
seconds; bootstrap-CI on the median for confidence intervals. This is
the only number we trust as wall-time; everything else is investigation
*around* this number.

### Layer 2 — `macmon` correlation — *"are we actually on the GPU?"*

External observer. Run `macmon pipe -s N -i 200` in parallel with a
long-running sweep; correlate the GPU-busy fraction over the measurement
window against the bench-predicted busy fraction. If layer 1 reports
"saturated" but macmon shows 30% GPU busy, we're either on a CPU
fallback path or our op is too small to engage the GPU.

*Implementation lands in commit 5.*

### Layer 3 — `torch.profiler` — *investigation tool, not measurement*

Use when layer-1 wall time and layer-2 busy fraction disagree. Trust the
op call graph and dispatch sequence; **do not trust profiler timings as
GPU truth**. On torch 2.11, `ProfilerActivity.MPS` does not exist — the
profiler captures **CPU dispatch only**. The trace is useful for op
order, dispatch ratios, and (especially) finding hidden CPU sync points;
it is *not* useful for per-op GPU kernel times, because those spans
don't exist in the trace.

### Layer 4 — Apple Instruments / Metal System Trace — *escape hatch*

If layer 3 is misleading and the bench-vs-macmon gap can't be explained,
fall through to Apple's GPU instrumentation. **Out of M4 scope** — flag
it, revisit if we hit a wall.

## 3. Protocol (bench layer)

```
for _ in range(warmup): fn()                    # 1. bypass kernel-compile cache
for _ in range(trials):
    torch.mps.synchronize()                     # 2. drain in-flight GPU work
    t0 = perf_counter()
    fn()                                        # 3. dispatch + GPU execute
    torch.mps.synchronize()                     # 4. close the dispatch-vs-work gap
    timings.append(perf_counter() - t0)
```

Why each piece:

- **Warmup (no timing).** First-touch MPS kernels JIT-compile on the
  host. A discarded warmup pass ensures all subsequent dispatches hit a
  cached kernel — otherwise the first trial is anomalous and skews CIs.
- **Pre-call `mps.synchronize`.** PyTorch's MPS backend dispatches
  asynchronously: `fn()` returns a Python tensor whose underlying GPU
  work is *queued*, not finished. Without a pre-sync, your `t0` includes
  unrelated GPU work from the previous trial.
- **Post-call `mps.synchronize`.** Closes the dispatch-vs-work gap. Without
  it, you measure how fast Python pushes ops to the queue, not how fast
  the GPU finishes them — these are wildly different on small batches
  where dispatch time dwarfs kernel time.
- **Bootstrap CI on the median.** Wall-time samples are typically right-
  skewed (long tails from background activity). Median + bootstrap is
  more honest than mean ± stdev for this distribution.

Code: `src/rubik/perf/bench.py` (`time_op`, `bootstrap_ci`).

## 4. Profiler layer (CPU dispatch only on torch 2.11)

### How to run

```
uv run python experiments/mps-methodology/probe_profiler.py
```

Configures `(B, 24)` int8 states and `(B,)` int64 random move indices on
MPS at `B = 8192`. Does 20 uninstrumented warmup calls (beats the
kernel-compile cache before the profiler context opens), then runs
`apply_moves` under `torch.profiler.profile(activities=[ProfilerActivity.CPU])`
with `schedule(wait=1, warmup=2, active=5)`. Exports a Chrome trace JSON
to `experiments/mps-methodology/runs/<ts>/profiler/trace.json` (gitignored).

The trace is a `traceEvents` list — Chrome trace format. Each event
carries a `name` (e.g. `aten::gather`), `ts` (microseconds), `dur`, and
PID/TID metadata. **`dur` is CPU dispatch duration**, not GPU kernel
duration. Ordering is reliable; magnitudes are not directly comparable
to wall time.

### What the trace IS useful for

- **Op order.** Did we dispatch `aten::_to_copy` we didn't expect? Does
  the gather happen once, or N times? Confirmed on this trace: 5 active
  iterations produce exactly 5 `aten::gather` calls (1:1 with the
  intended op).
- **CPU sync points.** Any `aten::is_nonzero`, `aten::item`, or
  `aten::_local_scalar_dense` event represents a host-blocking GPU
  round-trip — the kind of overhead that's invisible to wall-time on a
  fast op but dominates on a small one.
- **Unexpected dispatches.** `aten::_to_copy` appearing in the steady
  state usually means a tensor is migrating devices on each call — see
  finding #2 below.

### What the trace is NOT useful for

- **Per-op GPU times.** `aten::gather` shows ~26μs of CPU `dur` here;
  the actual GPU kernel time is unknown from this trace. For wall time
  use `bench.py`; for GPU-engagement check use macmon.
- **Comparing op costs across runs.** CPU dispatch noise dominates;
  variance is high.

### Verifying "no unexpected CPU round-trips"

```
uv run python experiments/mps-methodology/verify_no_cpu_sync.py
```

Reads the most recent trace, counts `aten::item` + `aten::_local_scalar_dense`
events (the canonical "GPU→host scalar extraction" markers), and asserts
the count is bounded by a known M2 baseline.

The verifier is **baseline-aware, not absolute.** M2's `apply_moves` has
a documented bounds check that produces predictable scalar extractions;
the verifier knows this, asserts on a ceiling, and serves as a
regression detector. When the underlying cost is fixed, the constant
`EXPECTED_BOUNDS_CHECK_ITEMS_PER_CALL` drops to 0 and the verifier
re-locks at the new baseline.

Current baseline (M2 code): 8 scalar-extraction events per
`apply_moves` call × 5 active iterations = 40. Verifier passes when
observed count ≤ 40.

## 4.5. Findings (war stories from the trace)

### Finding #1 — `(t.any() or t.any())` forces a CPU sync per side

**Where.** `src/rubik/cube/env.py:80`:

```python
if (move_idxs_t < 0).any() or (move_idxs_t >= spec.n_moves).any():
    raise ValueError(f"move index out of range [0, {spec.n_moves})")
```

**What the trace shows** (one apply_moves call, edited for clarity):

```
aten::lt              ts=...275  dur=23 μs   # element-wise compare
aten::item            ts=...280  dur=0.5 μs  # ← scalar extract #1
aten::_local_scalar_dense   ts=...280  dur=0.083 μs   # ← #2
aten::any             ts=...312  dur=25 μs   # reduce-or
aten::is_nonzero      ts=...351  dur=323 μs  # ← __bool__ on .any() result
aten::item            ts=...351  dur=323 μs  # ← scalar extract #3
aten::_local_scalar_dense   ts=...351  dur=323 μs   # ← #4
aten::ge              ts=...676  ...         # second comparison, repeats
... [4 more scalar extracts]
```

**Why.** Python's `or` operator evaluates its left operand, calls
`__bool__()` on it (PyTorch routes this through `aten::is_nonzero` →
`aten::item` → `aten::_local_scalar_dense`, which **synchronously
materializes a scalar from the GPU to host**), and then short-circuits or
evaluates the right operand. Since the bounds check is two `.any()`
expressions joined by `or`, both `__bool__`s execute on every call (at
least when the left side is `False`, which is the steady-state).

We empirically observe **8** scalar-extraction events per call, not the
2 you might expect — each comparison emits an item-pair from the `lt`/`ge`
itself in addition to the one from `is_nonzero`. The trace doesn't lie:
4× `aten::item` + 4× `aten::_local_scalar_dense` per call.

**Cost on M4 Max.** Each scalar extraction forces the GPU dispatch queue
to drain so the host can read one byte. The `is_nonzero` → `item`
chain dominates: ~150–325 μs *per side* on this machine for a 8192-elem
input. So one `apply_moves` call pays roughly 0.5–0.7 ms in pure host-
GPU sync overhead before the actual gather even starts. The gather
itself dispatches in ~26 μs of CPU time. **The bounds check is two
orders of magnitude more expensive than the work it guards.** On small
batches this likely *dominates* total wall time.

**Fix pattern.** Replace `or` with `|`, then call `.any().item()` once
(if a check is needed at all):

```python
oob = ((move_idxs_t < 0) | (move_idxs_t >= spec.n_moves)).any()
if oob.item():    # one sync, not two
    raise ValueError(...)
```

Or skip the check entirely on internal hot paths — bounds-check in
upstream callers, not on every dispatch.

**Status.** Documented; fix tracked as a follow-up block. M4 measures,
doesn't fix — by design, so the verifier doubles as a regression
detector for the future PR.

### Finding #2 — perm table migrates to device on EVERY call

**Where.** Same function, `env.py:70`:

```python
perm = _perm_for(spec).to(states.device)
```

**What the trace shows.** 5 active iterations produce **5**
`aten::_to_copy` events. The perm table is a static `(12, 24)` int64
constant — there is no reason to migrate it 5 times.

**Why.** `_perm_for(spec)` returns a CPU tensor from `_MOVE_PERM`;
`.to(states.device)` is a no-op when source and destination devices
match, **except** that the source is CPU and `states.device` is MPS, so
every call performs an actual device-to-device copy of the same data.
The lookup `_MOVE_PERM[spec.name]` always returns the CPU snapshot;
there is no caching of the device-resident copy.

**Cost on M4 Max.** Each `_to_copy` shows ~125–310 μs of CPU dispatch
for a 12 × 24 = 288-byte transfer. Tiny payload, mostly overhead.

**Fix pattern.** Cache the device-resident perm table per `(spec, device)`
pair, e.g. via a `dict[tuple[str, torch.device], torch.Tensor]` keyed
lookup, or eagerly migrate at module import for the expected device.

**Status.** Flagged in `plans/m4-perf-1.md` "Out" scope as a follow-up.
Verifier counts `aten::_to_copy` for context but does not assert on it
yet; that gates on the fix landing.

### Finding #3 — gather and bounds dispatch ratio is 1:2

For 5 calls the trace shows: 5× `aten::gather`, 5× `aten::lt`, 5× `aten::ge`,
10× `aten::any`, 10× `aten::is_nonzero`, 5× `aten::_to_copy`,
5× `aten::index`. The bounds check generates more dispatches than the
actual work — another way to read findings #1 and #2: most of what
`apply_moves` does on MPS in M2 is *not* the gather.

## 5. macmon layer

*To be filled in commit 5 (probe_macmon + correlator).*

## 6. Batch-sensitivity findings

*Folded in from `experiments/batch-sensitivity-2x2/results.md` after the
sweep runs (commit 7). Will lead with intuition observations: where does
throughput plateau? What does the curve shape say about GPU saturation
on M4 Max for this workload?*
