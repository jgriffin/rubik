# beam-search-2x2 — M6 acceptance run

## Context

M6 lands batched beam search using V_θ as the scorer with within-beam dedup, parametric on `CubeSpec`. Source: `src/rubik/search/beam.py`. SPEC acceptance is 100% solve on 1000 depth-≤14 scrambles at `beam_width=256` with mean solution length within 1 move of BFS-optimal.

The bet (carrying over from M5 cycle-3): the `sync500_kmax20-30k` checkpoint's *ordering* signal (pred_std=1.51) is sharper than greedy's *calibration*-consuming policy can extract. Beam at width=256 should close the d11/d13 capability gap that left M5's gate unmet — or surface a substantive ordering-quality finding.

## Question

Does beam search at width=256 against `sync500_kmax20-30k/net_final.pt` reach the M6 acceptance gate (≥99.9% solve per depth on 1000 scrambles, mean V*-excess ≤ 1.0 across all depths)?

If yes, M5's "decent enough" framing was correct and the value net's ordering signal is sufficient. If no, the failure shape (which depths break, how far above V*-optimal the solutions sit) tells us whether to retrain (V*-stratified resampling, curriculum) or accept and proceed to M7.

## Layout

- `configs/sweep.yaml` — width sweep `(1, 4, 16, 64, 256)` × depths `1..14` × `n_per_depth=200`. Cycle entry; intuition formation. Equivalent to ~30–60 min wall on M4 Max.
- `configs/acceptance.yaml` — width=256 × depths `1..14` × `n_per_depth=1000`. SPEC's exact gate.
- `run.py` — entry point. Loads `BeamEvalConfig`, instantiates `ValueNet`, sweeps the `(depth, beam_width)` grid, logs per-cell metrics + raw `solve_lens` arrays.
- `run.sh` — one-line wrapper invoking the sweep.
- `analyze.py` — *(written in C4)* reads `metrics.jsonl` → `results.md` tables.
- `render_beam_curves.py` — *(written in C4)* produces `beam_curves.html` with per-depth small-multiples (5/5/4 banding from M5 cycle-3).
- `results.md` — sweep tables + acceptance verdict.
- `intuition.md` — hand-written Observations → Hypotheses (with verification plans) → Open questions.

## Earned vs first-try-defensible knobs

The trained net architecture (`body_widths=[4096, 1024]`, `n_residual_blocks=4`, `normalization=bn`) is **fixed by the checkpoint** — these aren't tunable here, only matched.

- **`beam_widths: [1, 4, 16, 64, 256]`** — first-try-defensible. Five cells, power-of-4 spacing covering 256× capability range. The `256` upper bound is dictated by SPEC; the spacing was chosen to surface the ordering-vs-width curve cleanly without burning compute on intermediate steps. If the curve has structure between 64 and 256 we'll add cells in C4.
- **`max_steps: 20`** — first-try-defensible. 2x2 QTM diameter is 14; +6 buffer for V_θ-induced suboptimal moves. Earnable adjustment if cells fail with `-1` despite reasonable solve-rate at lower depths.
- **`n_per_depth: 200` (sweep)** — first-try-defensible. Matches M5 cycle-3 chart sample size; binomial SE ≈ 0.035 per cell, adequate for ordering across widths.
- **`n_per_depth: 1000` (acceptance)** — earned by SPEC. Verbatim per the M6 acceptance gate.
- **`seed: 0`, `device: mps`** — bookkeeping defaults.

## Pass criteria

- **Sweep run (C3):** no acceptance gate; intuition only. Read `beam_curves.html` for the solve-rate-vs-width curves per depth.
- **Acceptance run (C5):** every depth's `solve_rate ≥ 0.999` (≤ 1 failure per 1000-scramble cell) AND `mean_v_star_excess ≤ 1.0` averaged across all 14 depths.
