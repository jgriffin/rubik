# Batch-size sensitivity sweep (2x2)

## What this dir is

Sweeps `apply_moves` and `random_scrambles` throughput across batch sizes on this M4 Max to find the dispatch-overhead floor, the saturation ceiling, and any allocator/memory cliffs. The curve shapes are inputs to M7 (beam search) and M8 (DAVI training) hyperparameter choices, and a deliberate exercise in building GPU-saturation intuition for this hardware.

Sweep grid lives in `config.yaml`. Driver: `run.py`. Steady-state spot-check: `run_steady_state.py`. Renderer: `analysis/analyze.py`.

## Latest run

- **Timestamp:** `2026-05-02T06:01:58.011866Z`
- **Machine:** `macOS-26.3-arm64-arm-64bit`
- **Torch:** `2.11.0`
- **Git SHA:** `3a0b530a34fb11c3a511316931e9c9f284717c19`
- **Data:** `experiments/batch-sensitivity-2x2/runs/2026-05-02T06-01-37Z/data.json` (gitignored)
- **Sweep ops:** `['apply_moves', 'random_scrambles']`
- **Batch sizes:** `[1, 8, 64, 512, 4096, 32768, 262144, 2097152]`
- **Trials per cell:** 30; warmup: 5

## Methodology pointers

- `experiments/mps-methodology/results.md` — full triangulation strategy (bench-bracket → macmon → profiler), gotchas list, tools matrix. Read first if these numbers look surprising.
- `src/rubik/perf/bench.py` — `time_op` (warmup → sync → perf_counter → fn → sync → perf_counter) and `bootstrap_ci` (median + 95% CI via numpy resampling).
- `experiments/batch-sensitivity-2x2/config.yaml` — sweep grid.

## Throughput tables

### `apply_moves` (states/s)

| batch | median (ms) | CI lo (ms) | CI hi (ms) | throughput (median) | throughput CI lo | throughput CI hi |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.626 | 0.607 | 0.639 | 1.598 kstates/s | 1.564 kstates/s | 1.648 kstates/s |
| 8 | 0.688 | 0.673 | 0.693 | 11.625 kstates/s | 11.549 kstates/s | 11.880 kstates/s |
| 64 | 0.687 | 0.614 | 0.724 | 93.184 kstates/s | 88.347 kstates/s | 104.235 kstates/s |
| 512 | 0.617 | 0.612 | 0.627 | 829.962 kstates/s | 816.777 kstates/s | 836.316 kstates/s |
| 4096 | 0.564 | 0.560 | 0.577 | 7.264 Mstates/s | 7.102 Mstates/s | 7.312 Mstates/s |
| 32768 | 0.714 | 0.702 | 0.724 | 45.891 Mstates/s | 45.257 Mstates/s | 46.649 Mstates/s |
| 262144 | 1.661 | 1.633 | 1.679 | 157.783 Mstates/s | 156.104 Mstates/s | 160.498 Mstates/s |
| 2097152 | 7.454 | 7.435 | 7.478 | 281.339 Mstates/s | 280.445 Mstates/s | 282.069 Mstates/s |

### `random_scrambles` (scrambles/s)

| batch | median (ms) | CI lo (ms) | CI hi (ms) | throughput (median) | throughput CI lo | throughput CI hi |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.332 | 0.331 | 0.333 | 3.013 kscrambles/s | 3.001 kscrambles/s | 3.025 kscrambles/s |
| 8 | 0.350 | 0.349 | 0.351 | 22.884 kscrambles/s | 22.818 kscrambles/s | 22.940 kscrambles/s |
| 64 | 0.497 | 0.496 | 0.498 | 128.837 kscrambles/s | 128.552 kscrambles/s | 129.059 kscrambles/s |
| 512 | 2.560 | 2.497 | 2.588 | 200.037 kscrambles/s | 197.828 kscrambles/s | 205.082 kscrambles/s |
| 4096 | 14.519 | 14.453 | 14.590 | 282.113 kscrambles/s | 280.733 kscrambles/s | 283.408 kscrambles/s |
| 32768 | 67.067 | 66.944 | 67.117 | 488.585 kscrambles/s | 488.223 kscrambles/s | 489.485 kscrambles/s |
| 262144 | 471.510 | 470.360 | 473.483 | 555.967 kscrambles/s | 553.651 kscrambles/s | 557.326 kscrambles/s |

## Observations

Reading these as GPU-saturation intuition for this M4 Max: dispatch overhead dominates at the smallest batches (per-call time is flat regardless of how many states we hand the GPU); the curve lifts when batch is large enough to amortize kernel launch; it plateaus when memory bandwidth or compute throughput is the wall.

### `apply_moves`

- Dispatch-bound region: B=1..512 per-call seconds vary by only 10.4% — call overhead dominates work.
- Scaling efficiency curve (per 8x batch step): 1→8: 0.91, 8→64: 1.00, 64→512: 1.11, 512→4096: 1.09, 4096→32768: 0.79, 32768→262144: 0.43, 262144→2097152: 0.22.
- Saturation onset: scaling efficiency drops below 80% starting at B=4096 (throughput grows but no longer linearly with batch).
- Heavy saturation: efficiency below 40% starting at B=262144 — throughput is approaching a hard ceiling.
- Peak observed throughput: 281.339 Mstates/s at B=2097152.

### `random_scrambles`

- Scaling efficiency curve (per 8x batch step): 1→8: 0.95, 8→64: 0.70, 64→512: 0.19, 512→4096: 0.18, 4096→32768: 0.22, 32768→262144: 0.14.
- Saturation onset: scaling efficiency drops below 80% starting at B=8 (throughput grows but no longer linearly with batch).
- Heavy saturation: efficiency below 40% starting at B=64 — throughput is approaching a hard ceiling.
- Peak observed throughput: 555.967 kscrambles/s at B=262144.

## Bench-bracket vs steady-state regime

Calibration pending — run `run_steady_state.py` and rerun `analysis/analyze.py --steady-state-run <path>` to embed the bench-vs-pipelined correction factor here. Until then: the sweep numbers are sync-bracket per-call seconds, which underestimate steady-state pipelined throughput by a regime-dependent factor (see `experiments/mps-methodology/results.md` §5).

## Intuition

*As of 2026-05-02. torch 2.11, M4 Max, methodology version 1. Re-derive
if conditions change. Distinguishes **observations** (mechanical, from
this run's data) from **hypotheses** (interpretive claims with confidence
+ evidence) from **open questions** (what we haven't verified yet).*

### Observations (mechanical)

- `apply_moves` per-call wall-time is essentially flat across B=1..4096
  (variance ~10%) — call/dispatch overhead dominates work in this regime.
- `apply_moves` saturation onset between B=4096 and B=32768 (efficiency
  0.79). Heavy saturation by B=262144→2M (efficiency 0.22).
  Peak observed throughput: **281 M states/s at B=2M**.
- `random_scrambles` saturates aggressively — efficiency drops to 0.19
  already at B=64→512. Peak: **556 K scrambles/s at B=262K**.
- `random_scrambles` throughput is **~25× lower than `apply_moves × depth`
  would predict**: predicted ~14 M scrambles/s if a depth-20 scramble is
  exactly 20 chained `apply_moves` calls; measured 556 K/s.
- macmon measured **95.2% GPU busy** at B=8192 (sustained 25s pipelined
  workload, commit 5). At the same B, bench-bracket per-call is ~15%
  conservative vs. pipelined per-call (correction factor 0.866).

### Hypotheses (with confidence + evidence)

**H1 — Below B≈4K, `apply_moves` is dispatch-bound, not work-bound.**
*High confidence.*
Evidence: per-call wall-time is flat at ~0.6 ms across B=1..4096;
throughput grows linearly with batch only because each call costs the
same fixed overhead regardless of batch — i.e. the GPU is mostly idle
and finishing each call before more work arrives.
Verifies via: macmon GPU% per cell, expected near-zero at small B and
ramping up around B≈4K. **Not yet measured per cell** — only B=8192 has
GPU% data so far.

**H2 — `apply_moves` peak (281 M states/s) is well below the
memory-bandwidth ceiling.** *Medium confidence.*
Evidence: 281 M × 24 bytes ≈ 13 GB/s of useful state read+write. M4 Max
unified memory bandwidth is ~400 GB/s. So the wall is **not** raw
bandwidth. Likely candidates: kernel launch rate, MPS dispatch overhead,
or the per-call perm-table migration (`_perm_for(spec).to(device)` —
gotcha #2 in mps-methodology/results.md).
Verifies via: profiler trace at B=2M to identify dominant op + cumulative
`_to_copy` time; targeted experiment with perm cached on-device.

**H3 — `random_scrambles` is bottlenecked on per-step CPU work, not
the GPU gather.** *Medium confidence — correlation + code-reading, not
yet measurement-verified.*
Evidence: the 25× throughput gap below `apply_moves × depth` is too
large to be explained by gather alone. `env.py:155` allocates `move_seqs`
on CPU (no `device=`); `torch.multinomial` (with `prune_same_face`) runs
per step with a CPU-built probability mask. The cumulative CPU work +
CPU↔GPU transfers per step are the suspected bottleneck.
Verifies via: profile `random_scrambles` at B=8192; look for cumulative
`aten::multinomial` and CPU↔GPU transfer events per step. Run macmon
during random_scrambles and compare GPU-busy% to the apply_moves baseline
— if H3 holds, GPU% should be substantially lower than 95%.

**H4 — Bench-bracket measurement underreports pipelined throughput by
~15% on warm caches, ~30% on cold.** *High confidence at B=8192;
extrapolation to other batch sizes is unverified.*
Evidence: two independent steady-state spot-checks at B=8192:
commit 5 measured factor 0.686 (cold-ish); commit 7 measured factor 0.866
(warmer). Ratios across cells stay honest because the sync stall is a
fixed per-call cost, but absolute numbers are conservatively low.
Verifies via: steady-state spot-check at multiple batch sizes — currently
only B=8192 is calibrated. A per-cell calibration is expensive but
tractable; the curve of correction-factor vs batch size would itself
be diagnostic.

### Open questions / next experiments

1. **Per-cell GPU%.** Run macmon in parallel with each cell of the sweep
   and correlate timestamps. Lightweight upgrade to `run.py` (~30 lines).
   Closes H1 directly, tightens H2 and H3.
2. **What is `apply_moves`'s actual ceiling at B=2M?** Profile at the
   top batch size; identify the dominant op. Likely the perm migration
   or kernel-launch overhead, not memory bandwidth (per H2).
3. **What if `_perm_for(spec).to(device)` is cached per device instead
   of migrated per call?** ~5-line change in `env.py`. Rerun sweep,
   measure throughput delta — should be most visible at small B where
   per-call overhead dominates. (Surfaced as gotcha #2 in
   `experiments/mps-methodology/results.md`.)
4. **What if the `apply_moves` bounds check at `env.py:80` is removed?**
   The `(t.any() or t.any())` pattern triggers 8 scalar extractions per
   call, ~0.5–0.7 ms of host-blocking sync (gotcha #1). Predicted:
   meaningful speedup at small B; negligible at large B where the cost
   amortizes. ~5-line change in `env.py`.
5. **Does `random_scrambles` depth matter for the bottleneck?**
   `depth=20` is the current default. Sweep depth at fixed B to see
   whether the multinomial-per-step or the apply-moves-chain dominates.
6. **At what batch size does macmon's 200ms sample interval become
   useless?** Aliasing point for our op; tells us where to switch to
   profiler-based introspection.

### What we **haven't** verified that we should be careful about

- The "memory bandwidth is not the wall" claim (H2) is an arithmetic
  inference, not a profiler-measured one. If MPS internally does extra
  copies we don't see, the effective bandwidth use could be higher.
- The "random_scrambles is CPU-bound" claim (H3) is the most
  speculative. It's plausible from the code but hasn't been profiled.
  Don't quote it as fact in downstream work — quote it as "hypothesis,
  to verify in M6 prep."
- Per-cell GPU% is unmeasured. We *infer* saturation from scaling
  efficiency dropping. That's standard but not the same as direct
  observation.
