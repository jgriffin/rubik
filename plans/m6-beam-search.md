# M6 — Beam search (2x2)

## Context

M5 closed with a "decent enough" value net (`experiments/davi-2x2/davi-baseline/runs/sync500_kmax20-30k/net_final.pt` — `body_widths=(4096, 1024)`, `n_residual_blocks=4`, `normalization=bn`, ~17M params). Greedy hits ~40% solve at d11 / ~35% at d13 — M5's greedy-based acceptance gate is **not met**. Per the M5 cycle-3 close, we are explicitly declaring M5 done and betting that beam search closes the capability gap, because cycle-2's diagnosis identified V_θ's *ordering* signal as sharper than its absolute calibration (cycle-3 `pred_std=1.51` vs target std 1.16) — and greedy consumes calibration while beam consumes ordering.

**M6 SPEC acceptance (2x2):**
- 100% solve rate on 1000 depth-≤14 scrambles at `beam_width=256`
- Mean solution length within 1 move of BFS-optimal (V\* via the cached oracle at `data/v_star_2x2.npz`)
- Implementation: batched beam search using V as scorer, within-beam dedup via state packing, parametric on `CubeSpec` from day one — no 2x2-specific branches

The empty `src/rubik/search/__init__.py` already exists and is the destination per SPEC.

---

## Approach

Single branch `m6-beam-search`. Five atomic commits. Each delivers a self-contained, testable slice.

### C1 — `src/rubik/search/beam.py` + tests

Public API mirroring `greedy_solve_batch` so the experiment scaffold can call either interchangeably:

```python
def beam_solve_batch(
    net: torch.nn.Module,
    spec: CubeSpec,
    states: torch.Tensor,        # (N, n_stickers) int8
    *,
    beam_width: int,
    max_steps: int,
) -> BeamSearchResult
```

`BeamSearchResult` dataclass: `solve_lens: (N,) int64` (-1 = failed, mirroring greedy), `solve_paths: list[list[int]]` (move-index sequences; empty list for failed rows), `n_expansions: int` (diagnostic).

**Internal shape:** sequential per-parent loop; within each parent, batched-beam expansion using `apply_all_moves` (existing primitive at `src/rubik/cube/env.py:129-147`, already earmarked for M6 in its docstring).

Per-parent inner loop (from beam frontier of size B ≤ `beam_width`):
1. `children = apply_all_moves(beam_states, spec).reshape(B*12, n_stickers)` → `(B*12, 24)`
2. `child_v = net(children).flatten()` → `(B*12,)` — single forward, batch sits well under the M4 saturation knee (B=4096; max here is 256×12=3072)
3. Mark solved children with sentinel score (`-1e9`, mirroring greedy at `src/rubik/solve/greedy.py:84`)
4. **Within-beam dedup via raw bytes:** pull `children.cpu().numpy()` → loop building `dict[bytes, (best_v, src_idx, move_idx)]` keyed on `state.tobytes()`; emit only the dedupped survivors. Raw bytes (not canonicalization) — see Decisions §1.
5. Top-K (lowest V) → next beam frontier; carry `(parent_beam_idx, move_idx)` back-pointers per slot for path reconstruction
6. Early exit if any beam slot is solved → walk back-pointers to emit move sequence

**Parametricity:** uses `spec.n_moves`, `spec.n_stickers`, `is_solved(states, spec)` — no `CUBE_2X2` references in `beam.py`.

**Tests** at `tests/search/test_beam.py` mirroring style of `tests/solve/test_greedy.py` (helper `_VStarNet` / `_ConstantNet` pattern, deterministic seeds, no fixtures):
- `test_beam_width_1_matches_greedy_on_corpus` — 100-state corpus across depths 1–10, width=1 produces identical `solve_lens` to `greedy_solve_batch`
- `test_beam_oracle_solves_in_v_star_steps` — oracle-V\* net at width≥4 solves in exactly V\*(state) moves on 50 random states
- `test_beam_width_k_never_worse_than_width_1` — width=4/16 solve_lens ≤ width=1 elementwise on a fixed corpus
- `test_max_steps_zero_marks_unsolved_failed` — only pre-solved rows return 0; others return -1
- `test_does_not_mutate_input_states` — input tensor preserved
- `test_restores_train_mode_after_call` — `net.training` state restored
- `test_solve_paths_apply_to_solve_state` — emitted move sequences, when applied to the input state, reach `spec.solved_state` (path correctness, not just length correctness)
- `test_within_beam_dedup_drops_duplicates` — synthetic case where multiple paths land on the same physical state; assert dedup keeps the better-scoring one
- Error-case parity with greedy: `test_rejects_wrong_shape`, `test_rejects_negative_max_steps`, `test_rejects_zero_beam_width`

Update `src/rubik/search/__init__.py` to export `beam_solve_batch` and `BeamSearchResult`.

### C2 — `experiments/beam-search-2x2/` scaffold

Directory layout mirroring `experiments/davi-2x2/davi-baseline/`:

```
experiments/beam-search-2x2/
├── configs/
│   ├── sweep.yaml           # beam_widths=(1, 4, 16, 64, 256), n_per_depth=200, depths=1..14
│   └── acceptance.yaml      # beam_widths=(256,), n_per_depth=1000, depths=1..14
├── run.py
├── run.sh
├── analyze.py
├── render_beam_curves.py
├── README.md
├── results.md (skeleton)
├── intuition.md (skeleton)
```

`BeamEvalConfig` dataclass at `src/rubik/search/config.py` (frozen, yaml round-trip via pyyaml — same pattern as `DAVIConfig` at `src/rubik/training/config.py:27-78`, every field required, no defaults):
```python
@dataclass(frozen=True)
class BeamEvalConfig:
    checkpoint_path: str
    body_widths: tuple[int, int]
    n_residual_blocks: int
    normalization: str
    depths: tuple[int, ...]
    beam_widths: tuple[int, ...]
    n_per_depth: int
    max_steps: int
    seed: int
    device: str
```

`run.py` flow: load `BeamEvalConfig` → instantiate `ValueNet` → load checkpoint (copy `_load_checkpoint`'s dual-format dict/legacy-weights pattern locally; don't import from `experiments/davi-2x2/run.py`) → for each `(depth, beam_width)` cell: generate `n_per_depth` scrambles with `random_scrambles(prune_same_face=True)`, call `beam_solve_batch`, log JSONL `event="cell"` record with `solve_rate`, `avg_solve_len`, full `solve_lens` array, V\*-excess (via `compute_excess_vs_v_star` from `rubik.solve.v_star_compare`). Output dir: `runs/<run-id>/{config.yaml, metrics.jsonl, solve_histograms_beam.json}`.

3 new tests in `tests/search/test_config.py`: yaml round-trip, missing-field raises, tuple types preserved through yaml.

### C3 — First eval run (sweep)

Execute `bash experiments/beam-search-2x2/run.sh` with `configs/sweep.yaml` against `runs/sync500_kmax20-baseline/`. Cell count: 5 widths × 14 depths × 200 scrambles = 14,000 scrambles total. Per-scramble cost at width=256 is ~20 expansions × 3072-state forward ≈ a few ms on M4 Max — full sweep target wall time ~30–60 min. Captures `solve_histograms_beam.json`. Commit the run dir (small files only — `.gitignore` already excludes `*.pt`; gate from M5 cycle-3 commit `8cfb85e` keeps `experiments/*/runs/` tracked).

### C4 — Analysis + chart + intuition

- `analyze.py` reads `metrics.jsonl` + `solve_histograms_beam.json` → writes `results.md` tables: solve_rate(depth × beam_width), avg_solve_len(depth × beam_width), V\*-excess(depth × beam_width).
- `render_beam_curves.py` produces `beam_curves.html` using the inline-SVG / `line_path()` pattern from `experiments/davi-2x2/davi-baseline/render_error_trajectories.py`. Two views:
  1. **Per-depth small multiples** (5/5/4 banding from the M5 cycle-3 layout convention): one cell per depth, x=beam_width (log scale), y=solve_rate, dotted reference at 1.0.
  2. **V\*-excess overlay:** one chart, x=depth, lines per beam_width, y=mean solve_len − V\*-optimal, dotted reference at 1.0 (the SPEC threshold).
- `intuition.md` written by hand following the project convention (Observations → Hypotheses with verification plans → Open questions). Specifically: did beam at width=256 close the d11/d13 gap that greedy (width=1) couldn't? Where does the ordering signal break down? Is the V\*-excess curve flat or growing with depth?

### C5 — Acceptance gate run

Execute with `configs/acceptance.yaml`: width=256, n_per_depth=1000, depths 1..14. Pass criteria (per SPEC):
- **Solve rate:** every depth ≥ 99.9% (1000 scrambles × 14 depths; SPEC's "100%" tolerates ≤ 1 failure per 1000-scramble cell)
- **Mean V\*-excess:** ≤ 1.0 across all depths

Append "Acceptance gate" section to `results.md` with the per-depth pass/fail table. If a depth fails:
- Tighten `max_steps` budget (current default 20; lift to 30 if needed — earnable, not a borrowed value)
- If solve rate fails *despite* unbounded `max_steps`, the failure is in V_θ's ordering signal at deep states — that's a substantive finding, write it up in `intuition.md` and consider whether to retrain (returning to M5) or accept and proceed to M7. **Don't** silently widen the beam past 256 to chase the gate — that's a SPEC violation.

Mark M6 ✅ in `ROADMAP.md` and close the LOG block with full `Outcome:` + commit SHAs.

---

## Decisions baked in (recorded here for the LOG `Outcome` to reference)

1. **Raw-bytes dedup, not canonicalization.** Within-beam dedup compares `state.cpu().numpy().tobytes()` for exact equality. The 24-rotation orbit canonicalization (`canonicalize_states_batch` at `src/rubik/oracle/v_star_2x2.py:108-119`) is correct for V\* lookup but **wrong** for in-beam dedup: it would collapse states whose paths from the original scrambled cube reach different physical states that happen to be in the same rotation orbit. Path reconstruction would emit moves that, applied to the original state, do not reach the canonical representative (rotated relative to the actual cube). Raw bytes is faster *and* correct here.
2. **Sequential per-parent processing.** Inner-beam expansion is batched (3072-state forward at width=256). Cross-parent batching (folding the N parent dimension into the same beam) is a perf optimization the M6 acceptance gate doesn't require — M4 Max at width=256 sequential should run the 1000-scramble acceptance well under a minute total. Defer cross-parent batching to M8 (3x3, where wall-time is in SPEC).
3. **Beam width sweep `(1, 4, 16, 64, 256)`.** Five cells, span 256× capability range, covers greedy baseline (width=1) through SPEC gate (width=256). Power-of-4 chosen by exploration intent (intuition formation), not borrowed from prior published work.
4. **`max_steps=20` default.** 2x2 QTM diameter is 14; +6 buffer for V_θ-induced suboptimal moves. Earnable adjustment if C3 cells fail with -1; do not pre-emptively raise.
5. **`n_per_depth=200` for the sweep, `1000` for acceptance.** Sweep matches the M5 cycle-3 chart resolution (binomial SE ≈ 0.035 per cell — adequate for ordering across widths). Acceptance matches SPEC verbatim.

---

## Files to add / modify

**Add:**
- `src/rubik/search/beam.py` — `beam_solve_batch`, `BeamSearchResult`
- `src/rubik/search/config.py` — `BeamEvalConfig`
- `tests/search/__init__.py`, `tests/search/test_beam.py`, `tests/search/test_config.py`
- `experiments/beam-search-2x2/{run.py, run.sh, analyze.py, render_beam_curves.py, README.md, results.md, intuition.md}`
- `experiments/beam-search-2x2/configs/{sweep.yaml, acceptance.yaml}`

**Modify:**
- `src/rubik/search/__init__.py` — export `beam_solve_batch`, `BeamSearchResult`, `BeamEvalConfig`
- `ROADMAP.md` — flip M6 from "not yet planned" to "✅ done" at C5 close (will reference this plan)
- `LOG.md` — open 🟡 block at C1, amend through C5, close on acceptance pass

## Reused primitives (no rewrite)

- `rubik.cube.env.apply_all_moves(states, spec)` — children expansion (single fancy-index, ~zero MPS sync if states already on device) — `src/rubik/cube/env.py:129-147`
- `rubik.cube.env.is_solved(states, spec)` — terminal check — `src/rubik/cube/env.py:149-160`
- `rubik.cube.env.random_scrambles(spec, batch_size, depth, generator, prune_same_face=True)` — eval scramble generator — `src/rubik/cube/env.py:177-223`
- `rubik.cube.spec.CUBE_2X2` — spec — `src/rubik/cube/spec.py:38-45`
- `rubik.solve.compute_excess_vs_v_star(solve_lens, states, v_star)` — V\*-excess — `src/rubik/solve/v_star_compare.py:22-59`
- `rubik.oracle.v_star_2x2.load_v_star(path)` — V\* dict loader — `src/rubik/oracle/v_star_2x2.py:200-205`
- `rubik.model.network.ValueNet(spec, body_widths, n_residual_blocks, normalization)` — net constructor — `src/rubik/model/network.py:66-130`

## Verification (end-to-end)

1. `uv run pytest tests/search/ tests/solve/ tests/cube/ tests/model/ tests/oracle/ tests/training/` — full suite green; ~9 new tests on top of the existing 238
2. `uv run ruff check && uv run ruff format --check` — clean
3. `bash experiments/beam-search-2x2/run.sh` against `configs/sweep.yaml` — produces `solve_histograms_beam.json`, `beam_curves.html` (open with `open` to spot-check before acceptance)
4. `uv run python experiments/beam-search-2x2/run.py --config experiments/beam-search-2x2/configs/acceptance.yaml --out-dir experiments/beam-search-2x2/runs/sync500_kmax20-acceptance` — **acceptance gate**: every depth's solve rate ≥ 99.9%, mean V\*-excess ≤ 1.0 across all depths

## Out of scope (recorded so they don't leak into the block)

- Cross-parent batched beam (perf — defer to M8 if 3x3 wall-time forces it)
- 3x3 beam (M8 swaps in `CUBE_3X3`; M6 acceptance is 2x2-only per SPEC)
- W&B integration (still backlogged from M5)
- V\*-stratified retraining or curriculum scheduling (user dropped these from M5 backlog this session)
- Alternate search algorithms (A\*, MCTS) — SPEC commits to beam search for M6
