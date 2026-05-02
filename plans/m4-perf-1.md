# M4 — Perf-1: MPS measurement methodology + batch sensitivity (2x2)

## Context

M4 is the project's first **perf** milestone. M2 landed a correctness-only
tensor cube (`apply_moves`, `random_scrambles`, `is_solved`,
`valid_next_moves_mask`) with no throughput targets. M4's job per `SPEC.md`:

1. Stand up a **reproducible MPS measurement methodology** for this
   M4 Max — written up as `experiments/mps-methodology/results.md`.
   Cover the tools (`torch.profiler`, `macmon`, `mactop`, `hyperfine`),
   the protocol (warmup → sync → measure), and the gotchas (CPU
   round-trips, MPS dispatch vs. work timing, warmup discard).
2. Run a **batch-size sensitivity sweep** on `apply_moves` and
   `random_scrambles`, capturing throughput + 95% CIs.
3. **Verify no CPU round-trips** in the hot path via profiler trace.
4. **Establish the experiment-loop pattern** — `experiments/<name>/run.py`
   + `config.yaml` + `results.md` + `runs/` (gitignored). M8 reuses it
   for training hyperparams.

The writeup IS the deliverable (per SPEC.md §6), not just an output.
M4 is measurement-first; optimization headroom is left for follow-ups.

A co-equal goal beyond cube-solving: **build genuine intuition about
what runs well on this M4 Max GPU**. Results writeups should lead with
saturation/scaling observations, not just numbers — what's parallelizable
on this hardware, where dispatch overhead dies, where memory bandwidth
or kernel launch cost becomes the wall.

**Branch:** `m4-perf-1`.

## Triangulation strategy (the authentic measurement loop)

MPS gives one sync primitive (`torch.mps.synchronize`) and almost no
introspection. `torch.profiler` on MPS is **historically patchy** — pre-2.4
it captured CPU dispatch only; 2.4+ improved but GPU spans can still be
aggregated weirdly. We're on 2.11 and don't yet know the fidelity — that
is itself a thing M4 probes.

We use **three layers that cross-check each other**:

- **Layer 1 — bracket-sync timing** (`bench.py`). Primary truth.
  `mps.sync` → `perf_counter` → `fn()` → `mps.sync` → `perf_counter`.
  The only reliable wall-time number; everything else is investigation
  around it.
- **Layer 2 — `macmon` correlation.** External observer. Run macmon at
  200ms in parallel with a long sweep (~30s of repeated calls). Math
  check: `(layer-1 throughput × calls) / 30s ≈ macmon GPU-busy fraction`.
  If layer-1 says we're saturated but macmon shows 30%, we're either
  CPU-falling-back or our op is too small to engage the GPU.
- **Layer 3 — `torch.profiler`.** *Investigation tool, not measurement
  tool.* Use when layer-1 and layer-2 disagree, to find the culprit op.
  Trust ratios and call graphs; don't trust per-op timings absolutely.

Layer 4 (Apple's Metal System Trace via Instruments) is the escape hatch
if layer-3 is misleading. **Out of M4 scope** — flag it, revisit if we
hit a wall.

## Scope

**In:**
- Two experiment dirs (`mps-methodology/`, `batch-sensitivity-2x2/`).
- One small new module `src/rubik/perf/bench.py` — warmup+sync+timer
  helper that codifies the protocol so experiment scripts use it
  consistently. Re-exported from `rubik.__init__` so future training/
  search code inherits the same primitive.
- Throughput numbers logged in `results.md` files with bootstrap 95% CIs.
- Profiler trace artifact + a verifier asserting no
  `aten::_local_scalar_dense` / `aten::item` / unexpected `aten::_to_copy`
  events in the steady-state gather hot loop.
- Test coverage for `bench.py` (warmup discard, trial count, MPS sync).
- macmon-correlation script: takes `(start, end)` window, pulls macmon
  CSV, returns measured GPU-busy fraction. Used to close the "are we
  actually on the GPU?" question.

**Out:**
- 3x3 perf (M5/M8).
- Beam search / training perf (M7/M8).
- Hot-path *fixes* — the per-call `_perm_for(spec).to(states.device)`
  micro-cost gets *measured* and *flagged* if material; actually fixing
  it is a follow-up block. M4 is measurement, not optimization.
- Apple Instruments / Metal System Trace integration. Flagged as
  layer-4 escape hatch; revisit only if profiler is misleading.
- New runtime deps. Stays at `torch`, `numpy`. Adds `pyyaml` as a dev
  dep (only — config files for experiments).

## Critical files

**New:**
- `src/rubik/perf/__init__.py` (re-exports)
- `src/rubik/perf/bench.py` (~80 lines: warmup+sync+timed-trials helper +
  bootstrap CI)
- `tests/perf/__init__.py`, `tests/perf/test_bench.py` (~60 lines)
- `experiments/mps-methodology/README.md`
- `experiments/mps-methodology/probe_profiler.py`
- `experiments/mps-methodology/verify_no_cpu_sync.py`
- `experiments/mps-methodology/probe_macmon.sh` + `correlate_macmon.py`
- `experiments/mps-methodology/results.md`
- `experiments/batch-sensitivity-2x2/README.md`
- `experiments/batch-sensitivity-2x2/config.yaml`
- `experiments/batch-sensitivity-2x2/run.py`
- `experiments/batch-sensitivity-2x2/analyze.py`
- `experiments/batch-sensitivity-2x2/results.md`

**Modified:**
- `src/rubik/__init__.py` — re-export `time_op`, `bootstrap_ci`.
- `LOG.md`, `ROADMAP.md`.
- `pyproject.toml` — add `pyyaml` to dev deps.

## Files reused from existing code

- `src/rubik/cube/env.py:apply_moves`, `apply_move_sequence`,
  `random_scrambles`, `is_solved`, `valid_next_moves_mask` — operations
  under measurement.
- `src/rubik/cube/spec.py:CUBE_2X2`, `CubeSpec.solved_state` — measurement
  targets.

## Implementation sections

### 1. Open the block (LOG + plan + branch)

LOG.md gets a new 🟡 block at the top; `plans/m4-perf-1.md` written;
ROADMAP.md M4 row links to plan with status flipped to in-progress.
Single commit on `m4-perf-1`.

### 2. `src/rubik/perf/bench.py` — measurement primitive

Minimal helper. API:

```python
def time_op(
    fn: Callable[[], Any],          # closure that does ONE op invocation
    *,
    warmup: int = 5,
    trials: int = 30,
    device: str = "mps",
) -> list[float]:
    """Run warmup → sync → timed trials. Returns per-trial seconds."""

def bootstrap_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Returns (median, lo, hi) for the given confidence level."""
```

Responsibilities:
- Warmup calls: discard outputs.
- Each trial: `mps.synchronize()` → `perf_counter()` → `fn()` →
  `mps.synchronize()` → `perf_counter()`. (CPU device path: skip syncs.)
- Defensive: `device == "mps"` but MPS unavailable → raise immediately
  rather than silently time CPU.
- Stateless. No knowledge of cubes — pure timing.

**Tests** (`tests/perf/test_bench.py`):
- `test_time_op_returns_correct_trial_count` — `len(time_op(...))` matches.
- `test_time_op_warmup_runs_but_not_timed` — closure side-effect counter
  shows `warmup + trials` invocations but only `trials` results returned.
- `test_time_op_mps_sync_called` — monkeypatch `torch.mps.synchronize`,
  count calls; assert `2 * trials` for an MPS run. (Skip via
  `@pytest.mark.mps` if MPS not available on the runner.)
- `test_time_op_raises_on_mps_unavailable` — when `device="mps"` but
  unavailable.
- `test_bootstrap_ci_basic_shape` — `lo ≤ median ≤ hi`, deterministic
  under fixed seed.

### 3. `experiments/mps-methodology/`

**`README.md`** — purpose, tool list, reproduction commands.

**`probe_profiler.py`**:
- Sets up MPS device, builds `(B, 24)` int8 batch at `B=8192`.
- Wraps `apply_moves(...)` in `torch.profiler.profile(...)` with
  `schedule(wait=1, warmup=2, active=5)` and a `tensorboard_trace_handler`
  exporting Chrome traces to `runs/<ts>/profiler/`.
- Prints trace path; nonzero exit on profiler init failure.

**`verify_no_cpu_sync.py`**:
- Reads newest trace under `runs/`.
- Walks active-window events. **Asserts**: no `aten::_local_scalar_dense`
  / `aten::item`. `aten::_to_copy` is *allowed at most once* (perm
  migration on first call) — counts and bounds.
- Returns nonzero on failure.

**`probe_macmon.sh`** + **`correlate_macmon.py`**:
- Shell script prints the canonical macmon invocation
  (`macmon --interval 200ms --duration 30s --output runs/<ts>/macmon.csv`).
- Python correlator reads the CSV, takes a `(start, end)` window, returns
  mean GPU-busy fraction. Compares against the bench-predicted busy
  fraction passed via CLI; prints a verdict (`AGREE` / `DIVERGE`).

**`results.md`** — written iteratively. Sections:
1. *Setup* — machine identity, torch version.
2. *Triangulation strategy* — bench → macmon → profiler, with the
   "if these disagree" failure modes.
3. *Protocol* — warmup → MPS sync → time → MPS sync. Why each piece.
4. *Tools* — concrete commands; what each tells you, when each misleads.
5. *Gotchas list* — warmup undercount, dispatch-vs-work, hidden CPU
   round-trips, generator device for `random_scrambles`, macmon sample
   rate vs op duration.
6. *How to verify "no CPU round-trips"* — points at `verify_no_cpu_sync.py`
   with profiler trace excerpt embedded.
7. *Profiler fidelity note (decision-point output)* — what we found about
   MPS profiler on torch 2.11: which events show up cleanly, which don't,
   and what we're going to trust.

### 4. `experiments/batch-sensitivity-2x2/`

**`config.yaml`**:
```yaml
device: mps
ops: [apply_moves, random_scrambles]
batch_sizes: [1, 8, 64, 512, 4096, 32768, 262144, 2097152]
random_scrambles_depth: 20
warmup: 5
trials: 30
seed: 0
```

**`run.py`**:
- Parse config. For each `(op, batch_size)` cell:
  - Pre-build inputs on `mps` (e.g. `(B, 24)` int8 states + `(B,)` int64
    move idxs).
  - Build closure → `time_op(...)`.
  - Record per-trial seconds.
- Write `runs/<ts>/data.json` with full grid + metadata (config snapshot,
  torch version, machine, git SHA).
- Stream a one-line summary per cell to stdout.

**`analyze.py`**:
- Read newest `runs/<ts>/data.json` (or `--run path`).
- Per cell: throughput = `batch_size / median_seconds`, bootstrap 95% CI
  on throughput.
- Render markdown table to `results.md`.
- Lead with **intuition observations**: where does throughput plateau?
  Where does the dispatch ceiling lift? Is `random_scrambles` bottle-
  necked by per-step `multinomial`? What does the curve shape say about
  GPU saturation on M4 Max for this workload?

### 5. Lint, test, doc gate

- `uv run ruff check` and `uv run ruff format --check` clean.
- `uv run pytest` green; perf tests skip cleanly without MPS.
- LOG block closed with **Outcome**, **Decisions**, **Deviations**,
  **Commits**.
- ROADMAP.md M4 → ✅ done.

## Acceptance criteria (from SPEC.md §M4)

1. Methodology doc exists and is reproducible from a clean shell.
2. Throughput numbers logged with confidence intervals.
3. No CPU round-trips in the hot path — verified via profiler trace.
4. Experiment-loop pattern established — both dirs serve as templates
   for M8.

## Verification (end-to-end)

```bash
uv run pytest                              # all green incl. new perf tests
uv run ruff check && uv run ruff format --check
uv run python experiments/mps-methodology/probe_profiler.py
uv run python experiments/mps-methodology/verify_no_cpu_sync.py
bash experiments/mps-methodology/probe_macmon.sh   # in a separate term
uv run python experiments/mps-methodology/correlate_macmon.py --window <s,e>
uv run python experiments/batch-sensitivity-2x2/run.py
uv run python experiments/batch-sensitivity-2x2/analyze.py
open experiments/batch-sensitivity-2x2/results.md
open experiments/mps-methodology/results.md
```

Each step exits 0; `results.md` files contain real numbers (not
placeholders); profiler trace asserts pass.

## Atomic commit ladder

1. **Open** — LOG block opening + this plan + ROADMAP update.
2. **`perf/` module** — `src/rubik/perf/{__init__.py,bench.py}` + tests.
3. **Skeletons** — both `experiments/<name>/` dirs with README + config +
   placeholder results.md. `pyyaml` dev dep added.
4. **Profiler probe + verifier** — `probe_profiler.py` +
   `verify_no_cpu_sync.py` + first slice of `mps-methodology/results.md`
   (decision point: pause to eyeball trace fidelity).
5. **Macmon probe + correlator + methodology fill** — `probe_macmon.sh` +
   `correlate_macmon.py` + complete `mps-methodology/results.md`.
6. **Batch sweep driver** — `batch-sensitivity-2x2/run.py` + `analyze.py`.
7. **Run sweep + write results** — execute, commit `results.md` with real
   numbers + intuition observations (raw `runs/` gitignored).
8. **Close** — LOG outcome, ROADMAP flip, final lint+test pass, merge to
   `main` with `--no-ff`.

## Open questions (none blocking)

- **Profiler fidelity on torch 2.11.** Probe will tell us whether GPU
  spans land cleanly. Methodology degrades gracefully if not.
- **`random_scrambles` device.** `move_seqs` is built on CPU
  (`env.py:155`); per-step `multinomial` may be the actual bottleneck
  rather than the gather. M4 reports; M2 behavior unchanged.
- **Largest sweep batch size.** `2_097_152 × 24 = 50MB` int8. Trim if
  allocator pressure shows up; not worth blocking on.
- **Trace verifier scoping.** `aten::_to_copy` is expected once (perm
  migration); verifier counts and bounds rather than absolute-rejecting.
