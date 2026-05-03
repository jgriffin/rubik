# M5 — DAVI: scramble pipeline + value network + training loop (2x2)

## Context

M5 is the project's first ML milestone — the pivot from "correct cube
infrastructure" (M0–M4) to "actually train a value network." Lineage is
DeepCubeA / EfficientCube: backward-generate random scrambles from solved,
train a value net `V_θ` to estimate cost-to-go via approximate value
iteration (DAVI), evaluate against BFS-optimal `V*` (the 2x2 has only
~3.6M reachable states, so BFS V* gives per-state ground truth — the
single strongest correctness signal in the project).

**Why M5 matters beyond shipping a model.** M5 stress-tests two earlier
bets: (a) `CubeSpec` parameterization stays clean once a network and
loss enter the picture (M8 will swap in 3x3 by changing one spec, not
adding code paths); (b) the `experiments/<name>/` + iteration-loop
pattern from M4 generalizes from perf to training. Failures here would
surface as 3x3 forks creeping in, or as a training run that can't be
analyzed without bespoke tooling.

**Acceptance gate (from SPEC §M5):**
1. Training loss decreases monotonically (modulo noise) over 100k steps.
2. Mean absolute error vs BFS-optimal `V*` < 1.0 across all reachable 2x2 states.
3. Greedy solve rate ≥ 99% on depth ≤ 11 scrambles (2x2 God's Number is 11 QTM).
4. Lint + format + full test suite green.

## Approach overview

Branch `m5-davi`. Six deliverables, each its own atomic commit. Sequenced
so each commit ships something testable end-to-end (or close to it):

1. **BFS V\* enumerator** — `oracle/v_star_2x2.py`. Runs once, produces a `dict[bytes, int]` mapping packed states → min-cost-to-solve. Reusable in M6 (verify beam returns optimal lengths) and M7 (hparam eval). State-packing here is CPU `state.numpy().tobytes()` — no GPU pack needed for M5, defer that to M6.
2. **`apply_all_moves` env helper + ADI scramble pipeline** — extend `cube/env.py` with `apply_all_moves(states, spec) -> (B, n_moves, n_stickers)`; add `training/scrambles.py` with `generate_adi_batch(spec, batch_size, max_depth, generator) -> (states, depths, last_faces)`. ADI batches use **balanced per-depth slicing** (B/max_depth states per depth) — no padding waste, reuses existing `random_scrambles`.
3. **Value network** — `model/network.py`. MLP per draft spec but parametric on `CubeSpec`: input `(B, n_stickers, n_colors)` one-hot → flat `(B, n_stickers × n_colors)` → 5000 → 1000 → 4×residual(1000→1000) → 1. BatchNorm + ReLU on body, no activation on output. ~10M params on 2x2.
4. **DAVI loop** — `training/davi.py` + `training/config.py`. Per-step: expand all 12 children via `apply_all_moves`, run `V_target` on `(B*12, n_stickers)`, mask solved children to 0, take `min_a (1 + V_target(child))` per row, MSE against `V_θ(states)`. Periodic `V_target ← V_θ` sync. Adam, LR 1e-4, batch 1000, target-sync every 5000 steps (DeepCubeA values, the M7 sweep retunes).
5. **Experiment dir + training run** — `experiments/davi-2x2/` with `config.yaml`, `run.py`, `analyze.py`, `runs/` (gitignored). One full 100k-step run end-to-end. JSONL training log; text-summary analyzer; checkpoints every 10k steps. **No tensorboard/wandb/matplotlib** — same JSONL-then-digest discipline as M4.
6. **Acceptance evaluation + writeup** — `analyze.py` loads final checkpoint, evaluates against V\*: per-state MAE, MAE bucketed by depth, greedy solve rate at depth 1..11. `results.md` (auto-generated tables) + `intuition.md` (hand-written, hypotheses with verification plans per the M4 convention) → appended to `results.md`.

## Key decisions

### State-packing — `bytes` for M5, defer GPU pack to M6

For M5's only consumer (BFS V* enumeration on CPU, 3.6M states one-shot),
`state.cpu().numpy().tobytes()` as dict key is correct, simple, and fast
enough — ~80 MB dict, ~30s BFS wall clock. The GPU-resident uint64 pack
the draft spec mentions for beam-search dedup is **M6 work**, where it's
actually used. M5 doesn't pre-build infrastructure for unborn callers.

### Network width — keep draft architecture, parametric on `CubeSpec`

Draft says 5000 → 1000 → 4×residual(1000→1000) on 3x3 (input 288). On
2x2 (input 144) the same body width is ~10M params. Tempting to shrink
for 2x2, but: (a) 5000-wide first layer with 144 input is only 725k
params (the 9M+ lives in the residual body, which we keep identical
for M8 transfer); (b) M7's hparam sweep is the right place to find the
2x2 sweet spot, not M5; (c) keeping the architecture matches DeepCubeA
numerically — divergences become bugs, not "did we change the recipe?"
The constructor takes `CubeSpec` and derives input dim from
`spec.n_stickers × spec.n_colors`, so 3x3 drops in at M8 unchanged.

### Depth curriculum — uniform 1..14 for 2x2

Draft uses 1..30 (3x3-shaped). 2x2 God's Number is 11 QTM; sampling past
11 gets diminishing returns (every reachable state already lies within
11 moves of solved). Cap at 14 (3-move buffer above God's Number) so
the network sees a "saturated" regime that lets it learn the upper
plateau. Balanced per-depth slicing means depth=14 gets B/14 states,
not B/30 — sharper signal at the depths that matter.

### V\* enumerator location — `oracle/v_star_2x2.py`

V* is correctness ground truth for the 2x2, same role as the cubie
oracle. The size-specific suffix (`_2x2`) follows the project naming
convention (purpose-first, size-trails). 3x3 has 4.3×10¹⁹ states and is
**not** BFS-enumerable — so V* is genuinely 2x2-only, and the suffix
honestly signals that. Returns a `dict[bytes, int]` mapping packed
solved-state-relative states → optimal QTM distance. Cached to disk
under `data/v_star_2x2.pkl` (gitignored) on first compute.

### Logging — JSONL + text analyzer, no chart tooling

Per M4's locked-in streaming-output policy: training writes
`runs/<ts>/log.jsonl` (one JSON object per logging event — train step,
target sync, eval) and `runs/<ts>/checkpoints/`. `analyze.py` reads
JSONL and prints a text summary. No tensorboard, no wandb, no
matplotlib. If we want a curve later, we add SVG plotting (consistent
with M3 viz stack) — not a runtime dependency.

### `apply_all_moves` lives in `cube/env.py`

It's a cube primitive (input/output are state tensors, no model
involvement), used by DAVI now and by beam-search children-expansion at
M6. Co-located with `apply_moves`. Implementation reuses the cached
`_perm_for_device` from M4's follow-up.

### ADI batch construction — balanced per-depth slicing

`generate_adi_batch(spec, B, max_depth)` constructs B/max_depth states
at each depth d in 1..max_depth, calls existing `random_scrambles` per
depth, concats, shuffles, returns `(states, depths, last_faces)`.
Avoids the "draw K then waste" of per-row variable depth. `last_faces`
is `(move_seqs[:, -1] >> 1)` per row, threaded through for downstream
non-trivial-move pruning during search-time eval (greedy solve uses
`valid_next_moves_mask`).

## Critical files

**New:**
- `src/rubik/oracle/v_star_2x2.py` — BFS V* enumerator, packed-state dict.
- `src/rubik/training/scrambles.py` — `generate_adi_batch`.
- `src/rubik/training/davi.py` — DAVI step + target sync.
- `src/rubik/training/config.py` — `DAVIConfig` dataclass.
- `src/rubik/model/network.py` — value network (`ValueNet` class).
- `experiments/davi-2x2/config.yaml`, `run.py`, `analyze.py`, `results.md`, `intuition.md`.
- Tests: `tests/oracle/test_v_star_2x2.py`, `tests/cube/test_apply_all_moves.py`, `tests/training/test_scrambles.py`, `tests/training/test_davi.py`, `tests/model/test_network.py`.

**Modified:**
- `src/rubik/cube/env.py` — add `apply_all_moves`. Reuses existing perm cache.
- `src/rubik/__init__.py` — re-export `ValueNet`, `DAVIConfig`, `generate_adi_batch`.
- `pyproject.toml` — add `pyyaml` if not already there (M4 added it; verify).
- `.gitignore` — `data/v_star_2x2.pkl`, `experiments/davi-2x2/runs/`.

## Reused existing primitives

- `cube/env.py::apply_moves`, `random_scrambles`, `valid_next_moves_mask`, `_perm_for_device` (M4 cache).
- `cube/spec.py::CUBE_2X2`, `CubeSpec.solved_state`, `n_stickers`, `n_colors`, `n_moves`.
- `notation/state.py::state_to_dict` (only needed if we render a sample-solve in `analyze.py`).
- `perf/bench.py::time_op`, `bootstrap_ci` — for the per-step timing reported in `results.md`.

## Sequencing — atomic commits

Each commit is independently testable; the block stays mergeable at any
point if scope flexes.

| # | Commit | Lines (est.) | New tests |
|---|--------|--------------|-----------|
| 0 | Open block: branch + `plans/m5-davi.md` committed | – | – |
| 1 | `oracle/v_star_2x2.py` + tests; `data/` gitignored | ~120 src / ~80 test | 5 |
| 2 | `apply_all_moves` + `training/scrambles.py` + tests | ~80 src / ~120 test | 6 |
| 3 | `model/network.py` + tests | ~140 src / ~80 test | 5 |
| 4 | `training/davi.py` + `config.py` + tests (no run yet) | ~150 src / ~120 test | 6 |
| 5 | `experiments/davi-2x2/` scaffold + first 100k run | ~180 src + run output | 0 |
| 6 | Acceptance eval against V*; `results.md` + `intuition.md` | ~80 src + writeup | 0 |
| 7 | (If gate misses) hparam fix; rerun | – | 0 |
| 8 | Close block: LOG.md outcome | – | – |

Total: ~22 new tests, ~750 LoC source + ~480 test, plus the experiment dir and writeup.

## Test strategy

- **`oracle/v_star_2x2`** — V*(solved) = 0; reachable count = 3,674,160 (known); V*(R) = 1; V*(R · R) = 0; V*(scramble of depth d) ≤ d for a corpus.
- **`cube/apply_all_moves`** — equivalence vs per-move `apply_moves` over a corpus; shape `(B, n_moves, n_stickers)`; identity (`M_a` then `M_a^{-1}` returns to start for all a).
- **`training/scrambles`** — depth distribution roughly uniform (chi-square at α=0.05 with seeded gen); `last_faces[i] >> 1 == move_seqs[i][-1] >> 1`; deterministic under fixed seed; `B=0` and `max_depth=1` edge cases.
- **`model/network`** — input `(B, 24)` int8 → output `(B,)` float32; gradient flows; param count within 5% of analytical estimate; constructor accepts `CubeSpec` and adapts input dim; deterministic forward under `torch.manual_seed`.
- **`training/davi`** — target for an all-solved batch is 0; target for "one-move-from-solved" is 1 (since one child is solved → 1 + 0); one training step on a synthetic batch reduces loss; target-sync copies parameters bit-exactly; `DAVIConfig` round-trips through YAML.

Pattern follows M3/M4: plain test functions, no conftest, parametrize
identities, deterministic seeds, `CUBE_2X2` passed inline.

## Verification — how to validate end-to-end

```bash
# Lint + format + test suite (always clean before close)
uv run ruff check
uv run ruff format --check
uv run pytest

# Per-deliverable smoke (commit-by-commit)
uv run python -c "from rubik.oracle.v_star_2x2 import compute_v_star_2x2; from rubik.cube import CUBE_2X2; v = compute_v_star_2x2(CUBE_2X2); print(len(v), max(v.values()))"  # expect: 3674160, 11

uv run python -c "from rubik.training.scrambles import generate_adi_batch; from rubik.cube import CUBE_2X2; s, d, lf = generate_adi_batch(CUBE_2X2, 1400, 14); print(s.shape, d.unique(return_counts=True))"

uv run python -c "from rubik.model import ValueNet; from rubik.cube import CUBE_2X2; import torch; net = ValueNet(CUBE_2X2); print(sum(p.numel() for p in net.parameters()), net(CUBE_2X2.solved_state.unsqueeze(0)).shape)"

# Full training run (~10-20 min wall time on M4 MPS)
uv run python experiments/davi-2x2/run.py --config experiments/davi-2x2/config.yaml

# Acceptance gate evaluation
uv run python experiments/davi-2x2/analyze.py --run experiments/davi-2x2/runs/<ts>
# expects in stdout: MAE vs V* < 1.0; greedy solve rate ≥ 99% at depth ≤ 11
```

## Open questions logged for `intuition.md` (not blocking M5 acceptance)

These get verification plans in the experiment's `intuition.md`, not
fixed in M5:

1. Does loss-by-depth flatten or stay tilted? (Hypothesis: deeper states under-converge until target sync stabilizes.)
2. Does greedy solve rate degrade smoothly past depth 11, or cliff? (Network has never seen depth>14 — interpolation vs. extrapolation question.)
3. Does balanced per-depth slicing actually help vs. uniform random K? (Defer to M7 sweep — A/B with same wall-clock budget.)
4. How much of training wall-clock is `V_target` forward vs. `V_θ` forward+backward? (Profile at one checkpoint; expect target-forward ~25% by op count.)

## Risks + mitigations

- **MPS wall-clock overrun.** 100k steps × ~10 ms/step = ~17 min — within budget. Mitigation: if forward+backward exceeds 20 ms/step, pre-budget cuts to 50k steps and we re-evaluate gate at half budget; the M7 sweep tightens it.
- **DAVI divergence (loss climbs).** Most common failure mode is target-sync interval too long for 2x2's small state space. Mitigation: log loss every 100 steps; if loss > 2× minimum-so-far at any point, halve target-sync interval and restart.
- **BFS memory blowup.** 3.6M state bytes-keyed dict is ~80 MB. Fits. Mitigation if not: stream BFS layer-by-layer, write final dict to disk in chunks.
- **Greedy solve rate < 99% at acceptance.** First diagnostic is MAE-by-depth. If shallow MAE good but deep MAE poor → curriculum issue; if uniform poor MAE → optimizer/architecture. Concrete remediation paths per case in `intuition.md`.
