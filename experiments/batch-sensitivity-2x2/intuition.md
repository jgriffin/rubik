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
