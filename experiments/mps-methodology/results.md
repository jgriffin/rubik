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

## 5. macmon layer (external GPU-engagement observer)

**Finding.** On 2026-05-02 at B=8192, the bench-predicted saturation
fraction was **1.000** (workload-bound — the apply_moves pipeline ran
back-to-back for the full 25 s window) and macmon measured **GPU busy =
0.952** with **mean power = 3.94 W** over 89 in-window samples. Verdict:
**AGREE** (|delta| = 0.048 < 0.10). The hot path is genuinely on the GPU
on this machine; nothing is silently CPU-falling-back at this batch size.

### How to run

```
bash experiments/mps-methodology/probe_macmon.sh
uv run python experiments/mps-methodology/correlate_macmon.py \
    --expected-busy <bench-predicted>
```

The probe launches `macmon pipe --samples 150 --interval 200` (NDJSON,
30 s at 5 Hz) into `runs/<ts>/macmon/samples.ndjson`, runs a ~25 s
back-to-back `apply_moves` workload at `B=8192` between two 1 s baseline
sleeps, then exits. The workload window timestamps land in
`runs/<ts>/macmon/window.json`. The correlator reads both and emits ONE
line. **No streaming output.**

### What the correlator reports

- **`gpu_busy`** — mean of `gpu_usage[1]` (the 0..1 fraction; the array's
  first element is the active frequency in Hz, second is the busy fraction).
- **`gpu_power`** — mean of `gpu_power` in watts.
- **`samples`** — number of NDJSON records that fell inside the workload
  window (out of 150 emitted). Samples before/after the window are dropped.
- **`window`** — workload window duration in seconds.

With `--expected-busy F`: prepends `AGREE` if `|measured - F| < 0.10`,
else `DIVERGE` (and exit code 1).

### Computing the expected-busy fraction

Two valid formulations — pick the one matching your workload regime.

1. **Steady-state pipeline** (recommended for back-to-back workloads):
   `expected = min(1.0, n_calls × per_call_seconds_workload / window_seconds)`
   where `per_call_seconds_workload = window_seconds / n_calls`. For the
   2026-05-02 run: 25.0 / 52529 = 0.476 ms/call → expected = 1.000 →
   AGREE with macmon's 0.952. This is the "are we keeping the GPU
   continuously busy?" question.

2. **Bench-bracket per-call substitution**: `expected = n_calls ×
   bench_median_seconds / window_seconds`. **Don't do this naively.**
   `bench.time_op` measures wall time of one `mps.synchronize() → fn() →
   mps.synchronize()` round trip — which stalls the GPU queue every call.
   For apply_moves at B=8192, bench reports 0.694 ms/call (CI ±0.002 ms).
   Substituted: 52529 × 0.000694 / 25.0 = **1.458** → DIVERGE against
   the macmon's 0.952. The mismatch is real and informative: bench-
   bracket per-call is **not** the same regime as a synchronization-
   batched workload. ~0.22 ms of bench's per-call cost is the bracket
   sync stall, not GPU work.

The methodology rule: **bench measures one-shot latency including
sync stalls; macmon measures continuous GPU utilization over a
window. Use formulation 1 for the busy-fraction comparison.**

### When macmon misleads

- **Sample-rate aliasing.** macmon samples at 200 ms minimum; per-call
  apply_moves at B=8192 takes ~0.5 ms. Each macmon bucket aggregates
  ~400 calls of GPU work — fine for a back-to-back workload, but if you
  call `apply_moves` once every 250 ms, macmon may catch the active
  bucket or the idle one depending on phase, and `gpu_busy` will
  oscillate wildly between samples. **macmon is reliable for sustained
  workloads, unreliable for sparse ones.**
- **Op too small to engage the GPU.** Below some batch size, dispatch
  + sync overhead dominates and the kernel itself is sub-microsecond.
  macmon will report `gpu_busy` near zero even though every call goes
  through MPS. This is not a bug — it's the inherent limit of external
  sampling at 200 ms granularity. Use bench (layer 1) and the profiler
  (layer 3) to characterize this regime; macmon is silent below it.
- **Power without busy.** GPU power can stay elevated briefly after
  the workload window (queued kernels still running). The 1 s
  post-workload sleep in `probe_macmon.sh` lets that drain into the
  out-of-window samples.

### Worked example (the canonical correlation)

```
$ bash experiments/mps-methodology/probe_macmon.sh
workload: 52529 apply_moves calls in 25.000s
samples: experiments/mps-methodology/runs/2026-05-02T05-46-50Z/macmon/samples.ndjson
window: experiments/mps-methodology/runs/2026-05-02T05-46-50Z/macmon/window.json

$ uv run python experiments/mps-methodology/correlate_macmon.py
gpu_busy=0.952 gpu_power=3.94W samples=89 window=25.0s

$ uv run python experiments/mps-methodology/correlate_macmon.py --expected-busy 1.000
AGREE: gpu_busy=0.952 gpu_power=3.94W samples=89 window=25.0s expected=1.000
```

This validates layer-1's claim that `apply_moves` at B=8192 is
GPU-bound on this M4 Max — bench measures 0.694 ms/call, macmon
confirms the GPU was active 95% of the workload window, power draw
matches the 3–4 W band typical of a single MPS workload on this chip.

## 6. Putting it together (the triangulation playbook)

Three short scenarios for resolving disagreement between layers.

### Scenario 1 — "Bench numbers didn't change after my optimization"

You wrote a faster kernel, ran your benchmark, and the wall-time
median is identical to before. Most likely cause: **missing sync
brackets.** Without `mps.synchronize()` before `perf_counter()` on
both sides, you're timing how fast Python pushes to the dispatch
queue, not how long the GPU takes. Both versions push at the same
rate; the GPU does different amounts of work but you can't tell.

- **Layer-1 fix.** Use `bench.time_op` (which brackets every trial) or
  paste in the protocol from section 3. Re-run.
- **Layer-2 cross-check.** If post-fix bench still shows no change,
  run `probe_macmon.sh` and check `gpu_power`. A faster kernel that
  saturates the GPU should show higher power; same power = same GPU
  work, your "faster" code didn't run more in parallel.

### Scenario 2 — "Bench says we're saturated but macmon shows 30%"

Layer 1 reports your op pegs a CPU core at 100% (you measured 25.0 s
wall in a 25.0 s window) but macmon reports `gpu_busy=0.30`. The GPU
is mostly idle — where's the wall time going?

- **Op too small.** If batch size is tiny (B<256 typically), the
  kernel runs in microseconds and macmon's 200 ms sampling window
  catches mostly idle GPU between dispatches. Confirm with a larger
  B; if `gpu_busy` jumps, this was the cause.
- **Silent CPU fallback.** Some ops have no MPS implementation and
  fall back to CPU through PrivateUse1 dispatch. Run `probe_profiler.py`
  and inspect the trace — look for a high density of `cpu_op` events
  or a missing `aten::gather` (or whatever your hot op is). If you see
  the work happening on CPU dispatch with no GPU equivalent, you've
  found the leak.
- **Hidden host syncs.** Even with MPS implementations, a single
  `aten::item` or `aten::is_nonzero` per call drains the queue.
  `verify_no_cpu_sync.py` catches the canonical patterns; finding #1
  in section 4.5 is the war story.

### Scenario 3 — "Profiler shows fast op but macmon shows long busy window"

The Chrome trace says `aten::gather` took 26 µs of `dur`. macmon
shows `gpu_busy=0.95` for a workload that called gather 50 K times
in 25 s — that's 1.25 s of total dispatch (50 K × 26 µs), which only
explains 5% of a 25 s window, not 95%.

- **The dispatch-vs-work gap.** On torch 2.11, the profiler captures
  CPU dispatch only (section 4 — `ProfilerActivity.MPS` doesn't
  exist). `dur` is "how long Python was inside the dispatch call",
  not "how long the GPU kernel ran." The GPU may continue executing
  long after `aten::gather` returned.
- **Trust split.** macmon owns "is the GPU engaged?" Bench owns "how
  long does end-to-end take?" Profiler owns "what's the dispatch
  sequence?" Don't ask any one tool a question outside its scope.

### Reproducing all measurements

```
# Layer 1 — bracket-sync timing primitive (sanity test).
uv run pytest tests/perf -q

# Layer 2 — macmon correlation on apply_moves at B=8192.
bash experiments/mps-methodology/probe_macmon.sh
uv run python experiments/mps-methodology/correlate_macmon.py
uv run python experiments/mps-methodology/correlate_macmon.py --expected-busy 1.000

# Layer 3 — profiler trace + CPU-sync verifier.
uv run python experiments/mps-methodology/probe_profiler.py
uv run python experiments/mps-methodology/verify_no_cpu_sync.py

# Batch sensitivity sweep (lands in commit 6).
uv run python experiments/batch-sensitivity-2x2/run.py
uv run python experiments/batch-sensitivity-2x2/analyze.py

# View the writeups.
open experiments/mps-methodology/results.md
open experiments/batch-sensitivity-2x2/results.md
```

All `runs/<ts>/` artifacts are gitignored — only the writeup files
and probe scripts are versioned. Re-running any probe writes a fresh
timestamped directory; `correlate_macmon.py` and
`verify_no_cpu_sync.py` auto-pick the newest.

## 7. Cleanup validation (M4 follow-up, 2026-05-02)

This section is the worked example for the **measure → fix → measure →
analyze** loop. M4 documented two CPU-sync gotchas (findings #1 and #2)
but deferred fixes — by design, so the verifier could double as a
regression detector once the fix landed. This section logs the fix
loop. The methodology pattern that comes out of this is in §8.

### 7.1 Hypothesis and baseline

**The two fixes.**

1. **Bounds-check relocation.** Move the bounds check at `env.py:80`
   to *before* the `move_idxs.to(states.device)` migration. When the
   caller passes a CPU `move_idxs` (the realistic pattern: built on
   CPU inside `random_scrambles`'s inner loop and the future beam-
   search children expansion), the bounds check now runs on CPU — no
   GPU sync. When a caller pre-migrates to MPS (the bench-convenience
   pattern in `batch-sensitivity-2x2/run.py`), the check still runs on
   MPS and pays the same sync cost as today (no regression).
2. **Per-(spec, device) perm cache.** Cache the device-resident perm
   table the first time a `(spec.name, device)` pair is requested.
   Subsequent calls do a dict lookup instead of a tensor `_to_copy`.

**Predicted savings.** Both fixes remove fixed-cost-per-call overhead.
That predicts a specific shape: **absolute time savings stay roughly
constant per call across batch sizes, but relative savings shrink as
batch size grows** (because GPU work scales with batch but the fixes
don't touch GPU work).

- Fix #1: removes ~0.5–0.7 ms of bounds-check sync per call (4 sync
  pairs × ~125–325 μs each, observed at B=8192 in finding #1).
- Fix #2: removes ~125–310 μs of perm-migration overhead per call
  (one `_to_copy` event for a 288-byte payload).
- **Combined**: ~0.6–1.0 ms reduction per call, *regardless of batch
  size*.

**Predicted post-fix wall-times** (combining both fixes):

| B | Baseline median (μs) | Predicted post-fix | Predicted relative win |
| ---: | ---: | ---: | ---: |
| 1 | 684.5 | ~50–200 | **70–93%** |
| 64 | 620.0 | ~50–200 | **68–92%** |
| 8192 | 628.3 | ~50–200 | **68–92%** |
| 2,097,152 | 7,770.9 | ~6,800–7,200 | **7–13%** |

The dispatch+bracket-sync floor (everything left after we strip the
bounds-check sync and the perm copy) is what determines the post-fix
small-B numbers. We don't know that floor a priori — it's a discovery,
not a prediction. **The hypothesis is the *shape*: small B → big
relative win, large B → small relative win, absolute win roughly
constant.** If the data shows otherwise, our model of the cost is
wrong.

**Baseline (pre-fix) — bench, 2026-05-02 17:31, branch `env-cpu-sync-cleanups`.**

`uv run python experiments/mps-methodology/bench_apply_moves.py --label baseline --out-dir experiments/mps-methodology/runs/cleanup-loop`

| B | median (μs) | 95% CI (μs) | throughput |
| ---: | ---: | ---: | ---: |
| 1 | 684.54 | [652.77, 694.38] | 1.46 K st/s |
| 64 | 620.00 | [612.75, 625.79] | 103.23 K st/s |
| 8192 | 628.33 | [627.33, 631.21] | 13.04 M st/s |
| 2,097,152 | 7,770.92 | [7,751.79, 7,798.75] | 269.87 M st/s |

Realistic call pattern: `move_idxs` built on CPU; `states` on MPS;
each call does a `.to(states.device)` migration of move_idxs inside
`apply_moves` followed by a perm-table migration and the bounds check
(currently on MPS, post-migration). Trials = 100, warmup = 10.

**Baseline trace.** `probe_profiler.py` confirms 40 scalar extractions
total = 8 per call × 5 active iterations (current verifier baseline);
5 `aten::_to_copy` events for the per-call perm migration. Verifier
PASS at the M2 ceiling.

### 7.2 Fix #1 — bounds-check relocation

*To be filled in after the fix lands.*

### 7.3 Fix #2 — per-(spec, device) perm cache

*To be filled in after the fix lands.*

### 7.4 Analysis

*To be filled in after both fixes have measurements.*
