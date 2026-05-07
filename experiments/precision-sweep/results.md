# precision-sweep — BF16 / FP16 inference for ValueNet on MPS

**Datestamp:** 2026-05-07
**Run conditions:**
- Machine: M4 Max (96 GB)
- Branch / commit: `beam-solve-perf` at `a50c08c` (the precision-flag commit)
- Checkpoint: `experiments/davi-3x3/runs/20260507T043533Z_full_train/net_final.pt`
- Architecture: 5120 / 1024 ValueNet, 4 residual blocks, BatchNorm
- Beam-eval params: `n_per_depth=100`, `walk_depths=1..14`, `seed=0`
- All sweeps run on the **same** beam.py (post-A1+A2 cross-scramble batched)
- Producer: `scripts/beam_eval_run.py --precision {fp32,bf16,fp16}` (this commit)

## What we ran

**Phase 1 — width=128 baseline at all three precisions.** Single width, three runs, same seed. Establishes the per-precision floor for the production default beam width.

**Phase 2 — 7-width sweep, FP32 vs BF16 only.** FP16 skipped at the 7-width tier — Phase 1 showed BF16 ≈ FP16 wall (within 0.2s) and identical solve rates, so the FP16 7-width cell would be uninformative.

## Phase 1 — width=128 results

| precision | wall (s) | speedup | solve@d=14 | solve@d=13 | solve@d=12 |
| --------- | -------: | ------: | ---------: | ---------: | ---------: |
| fp32      |    59.63 |   1.00× |      0.640 |      0.820 |      0.970 |
| bf16      |    43.58 |   1.37× |      0.640 |      0.820 |      0.970 |
| fp16      |    43.75 |   1.36× |      0.640 |      0.820 |      0.970 |

**Per-depth solve rate (all three precisions).** All 14 depths × 3 precisions are bit-identical. avg_solve_len also bit-identical across precisions at every depth.

| d   | fp32  | bf16  | fp16  |
| --- | ----: | ----: | ----: |
| 1   | 1.000 | 1.000 | 1.000 |
| 2   | 1.000 | 1.000 | 1.000 |
| 3   | 1.000 | 1.000 | 1.000 |
| 4   | 1.000 | 1.000 | 1.000 |
| 5   | 1.000 | 1.000 | 1.000 |
| 6   | 1.000 | 1.000 | 1.000 |
| 7   | 1.000 | 1.000 | 1.000 |
| 8   | 1.000 | 1.000 | 1.000 |
| 9   | 1.000 | 1.000 | 1.000 |
| 10  | 1.000 | 1.000 | 1.000 |
| 11  | 1.000 | 1.000 | 1.000 |
| 12  | 0.970 | 0.970 | 0.970 |
| 13  | 0.820 | 0.820 | 0.820 |
| 14  | 0.640 | 0.640 | 0.640 |

**States/sec implied** (states scored = `n_per_depth × max_walk_depth × beam_width × 12`):

| precision | states/sec |
| --------- | ---------: |
| fp32      |     36,065 |
| bf16      |     49,340 |
| fp16      |     49,154 |

## Phase 2 — 7-width sweep (FP32 vs BF16)

**Wall time per width:**

| width | fp32 (s) | bf16 (s) | speedup |
| ----: | -------: | -------: | ------: |
| 8     |     4.98 |     4.26 |   1.17× |
| 16    |     7.76 |     6.33 |   1.23× |
| 32    |    15.47 |    12.10 |   1.28× |
| 64    |    29.58 |    22.59 |   1.31× |
| 128   |    74.93 |    42.62 |   1.76× |
| 256   |   578.25 |   112.78 |  *5.13×* |
| 512   |   818.50 |   797.86 |   1.03× |
| tot   |  1529.46 |   998.53 |   1.53× |

**Headline:** width=256 went from 578s → 113s (5.13×). The whole sweep dropped from 25.5 min → 16.6 min.

**Per-cell solve rate Δ (BF16 − FP32) across all 98 cells (7 widths × 14 depths):**

```
w/d     d1   d2   d3   d4   d5   d6   d7   d8   d9   d10  d11  d12  d13  d14
  8    .00  .00  .00  .00  .00  .00 -.01  .00  .00  .00  .00  .00 +.01 -.01
 16    .00  .00  .00  .00  .00  .00  .00  .00  .00  .00 +.01  .00  .00 +.01
 32    .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00
 64    .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00 -.01 -.01  .00
128    .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00
256    .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00 +.01
512    .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00 +.02 +.01
```

**Max |Δ| across all 98 cells: 0.02** (2 percentage points).

For binomial samples at n=100 with p≈0.7, SE = √(0.7·0.3/100) ≈ 0.046 — every per-cell delta is well within 1×SE. At width=512, BF16 is *marginally better* (+0.02 at d=13, +0.01 at d=14), but this is within noise.

## Verdict

**Recommend BF16 as the new production inference default.**

Criteria check:

1. ✅ **Solve rate at d=11..14 within binomial SE of FP32** — every cell within ±0.02 (≤0.5×SE).
2. ✅ **Wall reduction > 20%** — 1.37× at width=128, 1.53× across the whole 7-width sweep, 5.13× at width=256.
3. ✅ **No NaN, no kernel errors** — clean run on MPS, all forward passes finite.

BF16 over FP16 because:
- Wall is essentially tied (43.58s vs 43.75s at width=128). The single-layer probe predicted a similar tie.
- BF16 has the wider exponent range (8-bit exp matching FP32) — no risk of overflow in BatchNorm running stats or intermediate activations as the network is later fine-tuned or grown deeper. FP16's 5-bit exponent has caused overflow surprises in other ML pipelines; BF16 doesn't.
- Solve rates are identical between BF16 and FP16 at width=128, so there's no quality reason to pick FP16.

**This is data, not policy.** The default in `beam_eval_run.py` remains `fp32` so prior runs reproduce bit-exactly. The user makes the call on flipping the production default.

**Gate documentation for future cycle decisions:** if a future cycle wants to flip the default to BF16 globally, it should re-run this sweep on its own fresh checkpoint and confirm the same per-cell |Δ| ≤ binomial-SE bound. The beam search is a chaotic system in the limit (small score perturbations *can* in principle change topk ordering); we got lucky here that the trained net's beam selections are robust to BF16 quantization, but that's an empirical property of *this* net, not a general guarantee.

## Files

- `results/sweep_fp32_w128.json`, `sweep_bf16_w128.json`, `sweep_fp16_w128.json` — Phase 1
- `results/sweep_fp32_full.json`, `sweep_bf16_full.json` — Phase 2
- `run.py` — reproducer documenting the canonical invocations
- `intuition.md` — observations / hypotheses / open questions (per project convention)
