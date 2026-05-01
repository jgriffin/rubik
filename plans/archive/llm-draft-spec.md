# Rubik's Cube Deep RL — Project Spec

## Context

This project implements a deep RL solver for the 3×3 Rubik's Cube on Apple
Silicon (M4 Max, MPS backend). The approach follows the DeepCubeA / EfficientCube
lineage: train a heuristic network via **Autodidactic Iteration (ADI) / DAVI**
on random scrambles generated backward from the solved state, then solve new
scrambles via **batched beam search** on GPU.

The author is a senior ML engineer. Favor concise, idiomatic PyTorch. Prefer
clarity over cleverness. Type hints everywhere. `pytest` for testing. No
premature optimization — but the environment and search loops are the hot
paths and *must* be vectorized on GPU from the start.

**Stack:** Python 3.12, PyTorch 2.x (MPS backend), numpy, pytest, matplotlib
(for cube visualization). No JAX, no TF. Use `uv` for deps.

## High-Level Approach

1. **Environment**: vectorized cube state as `[B, 48]` int8 tensors. Moves are
   precomputed index permutations; a step is a gather, not a matmul.
2. **Training**: DAVI (approximate value iteration). Generate random scrambles
   from solved state; train value network to estimate cost-to-go.
3. **Search**: GPU-resident batched beam search using the trained net as a
   policy (or value — see Phase 3).
4. **Visualization**: render any cube state to an unfolded 2D "cross" diagram
   for sanity-checking at every stage.

## Milestones

Work strictly in order. Each milestone has acceptance criteria. Do not begin
a milestone until the previous one's tests pass.

---

### Milestone 1: Cube Environment (Vectorized, GPU-Native)

**Goal:** a correct, fast, batched cube environment. This is the foundation
for everything else. Get it right and benchmarked before touching RL.

#### State Encoding

- **Representation**: `[B, 48]` tensor of `torch.int8`.
- **Why 48, not 54**: the 6 face centers are fixed — they never move under
  any face turn, so they carry no information. Drop them. This simplifies the
  move tables and saves 12.5% of state memory.
- **Sticker indexing**: stickers numbered 0-47. Document the exact numbering
  in a module docstring with an ASCII diagram. Suggest this convention:
  - 0-7: U face (reading order, skipping center)
  - 8-15: L face
  - 16-23: F face
  - 24-31: R face
  - 32-39: B face
  - 40-47: D face
- **Values**: each sticker holds its *color* (0-5), not its *identity*. Color
  corresponds to the face it started on. Solved state: first 8 stickers are
  color 0, next 8 are color 1, etc. This makes the solved check trivial:
  `(state == SOLVED).all(dim=-1)`.
- **Dtype**: `int8` is sufficient (values 0-5). Prefer `int8` over `uint8`
  because MPS has occasionally had issues with unsigned types — verify.

#### Move Set

- **12 moves** in quarter-turn metric (QTM): {U, U', D, D', L, L', R, R', F, F', B, B'}
- **Action indices** 0-11. Document the exact mapping in a constants module.
- **Move tables**: precompute a `[12, 48]` int64 tensor `MOVE_PERMS` where
  `MOVE_PERMS[a]` is a permutation such that `new_state = old_state[MOVE_PERMS[a]]`
  applies action `a`.
- **How to generate the permutations**: derive them from the geometry. For each
  face turn, list which sticker slot ends up where. Verify correctness via:
  (a) doing any move 4 times returns to start;
  (b) R U R' U' repeated 6 times returns to start (the "sexy move" identity);
  (c) all moves preserve the multiset of colors.
- **Inverse relation**: `M' = M · M · M` (3 applications). Use this as a test,
  not a derivation.

#### Non-Trivial Move Pruning (Critical for Efficiency)

After a move on face F, the next move should not be on F (wasteful: either
identity or should have been a double move). After moves on (F, opposite-F),
next should not be F again (commutation redundancy).

- Define `OPPOSITE_FACE` table: U↔D, L↔R, F↔B.
- Define `VALID_NEXT_MOVES`: a `[13, 9]` int64 tensor indexed by
  `last_move_face` (0-5 for the six faces, 6 for "no previous move" / -1
  encoded as 12 or sentinel, plus cases for "last two were F then opposite-F"
  which we'll track as a secondary state). Actually — simpler: track
  `last_face` and `second_last_face`. Compute valid next moves from those.
  See DeepCubeA's `environments/cube3.py` for a reference implementation of
  this pruning.
- For the initial implementation, **just track `last_face`** and prune same-face
  moves. That gets you from 12→9 branching and is 90% of the win. The
  opposite-face optimization (9→8) is a Phase-2 refinement.

#### API

```python
class Cube:
    SOLVED: torch.Tensor  # [48] int8
    MOVE_PERMS: torch.Tensor  # [12, 48] int64
    
    @staticmethod
    def solved_state(batch_size: int, device) -> torch.Tensor: ...
    
    @staticmethod
    def apply_moves(states: Tensor, actions: Tensor) -> Tensor:
        """states: [B, 48], actions: [B] -> new states: [B, 48]"""
    
    @staticmethod
    def apply_move_sequence(states: Tensor, action_seq: Tensor) -> Tensor:
        """states: [B, 48], action_seq: [B, T] -> [B, 48]"""
    
    @staticmethod
    def is_solved(states: Tensor) -> torch.Tensor:
        """states: [B, 48] -> [B] bool"""
    
    @staticmethod
    def random_scrambles(batch_size: int, depth: int, device) -> tuple[Tensor, Tensor]:
        """Returns (scrambled_states [B, 48], scramble_sequences [B, depth]).
        Uses non-trivial move pruning during generation."""
    
    @staticmethod
    def valid_next_moves_mask(last_face: Tensor) -> torch.Tensor:
        """last_face: [B] int, -1 for no previous move -> [B, 12] bool mask"""
```

#### Visualization (build in this milestone!)

- Function `render_cube(state: Tensor) -> Figure`: takes a single `[48]` state
  and returns a matplotlib figure showing the unfolded cube cross layout:
  ```
        [U U U]
        [U U U]
        [U U U]
  [L..] [F..] [R..] [B..]
  [L..] [F..] [R..] [B..]
  [L..] [F..] [R..] [B..]
        [D D D]
        [D D D]
        [D D D]
  ```
  - Use a fixed color palette matching standard WCA (White U, Yellow D, Green F,
    Blue B, Red R, Orange L).
  - Draw the fixed centers explicitly even though they're not in the state.
  - Function `render_sequence(initial_state, actions) -> Figure`: grid of
    subplots showing each intermediate state with the move label above.
- This exists so you can *visually verify* your move tables are correct. Write
  a test that scrambles a real cube (on paper or with a physical cube) and
  compares to the rendered output.

#### Acceptance Criteria

1. `Cube.apply_moves` is correct: all standard identities hold (M^4 = I,
   (R U R' U')^6 = I, Sune applied 6 times = I, etc.). Add these as pytest
   tests.
2. Throughput benchmark on M4 MPS: `apply_moves` on a batch of 100k states
   should hit **>10M transitions/sec**. Write a `benchmarks/` script.
3. `random_scrambles(10000, depth=30)` produces 10k distinct states (~99%+
   unique) with no trivial backtracking.
4. Visualization: can render the solved state, a single-move state (e.g., R),
   and a depth-20 scramble. Include a pytest snapshot test.
5. All state tensors live on MPS device throughout. No CPU round-trips in the
   hot path.

#### Deliverables

- `cube/state.py` — constants, move tables, encoding
- `cube/env.py` — the `Cube` class
- `cube/viz.py` — rendering
- `tests/test_env.py`
- `benchmarks/bench_env.py`
- A markdown doc `cube/ENCODING.md` explaining the sticker numbering with
  ASCII diagrams. This will be the reference for anyone (including future
  Claude) working on the code.

---

### Milestone 2: Scramble Generation & ADI Training Data Pipeline

**Goal:** generate training data for ADI efficiently, entirely on GPU.

#### Design

- `generate_adi_batch(batch_size, max_depth, device) -> dict`:
  - For each example: pick a depth K uniformly from 1..max_depth.
  - Apply K non-redundant random moves to the solved state.
  - Return: `{states: [B, 48], depths: [B], last_faces: [B]}`.
- **Non-redundant walks**: during scramble generation, never pick a move on
  the same face as the previous move. Same pruning as search.
- Depth distribution: uniform over 1..30 is a reasonable default. DeepCubeA
  trained up to depth 30; God's Number is 20 QTM (should be 26, double check).
  Empirically depth 26-30 covers the full difficulty range.

#### Acceptance Criteria

1. `generate_adi_batch(100_000, 30)` runs in under 1 second on M4.
2. Distribution test: depth histogram is uniform; sampled moves are uniform
   within the valid-move mask.
3. No trivial cancellations in scramble sequences.

#### Deliverables

- `cube/scrambles.py`
- `tests/test_scrambles.py`

---

### Milestone 3: Value Network + DAVI Training

**Goal:** train a value network that estimates cost-to-go. This is the
standard DeepCubeA recipe.

#### Architecture

Start with DeepCubeA's MLP. It works; don't innovate yet.

- Input: `[B, 48]` int8 → one-hot to `[B, 48, 6]` → flatten to `[B, 288]`.
- Body:
  - Linear(288, 5000) + BN + ReLU
  - Linear(5000, 1000) + BN + ReLU
  - 4x residual blocks: [Linear(1000, 1000) + BN + ReLU] x 2 with skip
  - Linear(1000, 1) — scalar cost-to-go
- No activation on output (regression target).

Parameter count: ~10M. Fits easily on M4.

#### DAVI Training Loop

```
Initialize V_theta, V_target (copy)
for iteration in range(N):
    # Generate training batch
    states, _ = generate_adi_batch(B, max_depth=30)
    
    # Compute target: for each state, look at all 12 children,
    # target = min over children of (1 + V_target(child))
    # Exception: if a child is solved, target for that child is 0.
    children = apply_all_moves(states)  # [B, 12, 48]
    with torch.no_grad():
        child_values = V_target(children.reshape(B*12, 48)).reshape(B, 12)
        solved_mask = Cube.is_solved(children.reshape(B*12, 48)).reshape(B, 12)
        child_values = torch.where(solved_mask, 0.0, child_values)
        targets = (1.0 + child_values).min(dim=1).values  # [B]
    
    # Regress V_theta(states) toward targets
    pred = V_theta(states)
    loss = MSE(pred, targets)
    # backward, step
    
    # Target network update
    if iteration % target_update_freq == 0:
        V_target.load_state_dict(V_theta.state_dict())
```

Key hyperparameters (DeepCubeA values as starting points):
- Batch size: 1000 (increase to 10k if memory permits)
- LR: 1e-4, Adam
- Target update: every 5000 steps (or when training loss stabilizes)
- Max depth: grow curriculum? DeepCubeA uses fixed max_depth=30. Start fixed.

#### Monitoring

- Log: loss, mean target value, mean prediction, histogram of predictions
  by scramble depth.
- Eval every N steps: generate 100 scrambles at each depth 1..25, greedy-solve
  using V (pick child with min V), report solve rate by depth.
- Use `tensorboard` or `wandb` — author preference.

#### Acceptance Criteria

1. Training loss decreases monotonically (modulo noise) over 100k steps.
2. At convergence: greedy solve rate ≥ 95% for scrambles of depth ≤ 15
   (no search, just greedy argmin V).
3. Predicted V correlates with actual distance: for scrambles of depth d,
   mean V should be roughly d ± small error.
4. Training runtime: complete a full training run (~1M steps) in <3 days on M4.

#### Deliverables

- `model/network.py` — the MLP
- `training/davi.py` — training loop
- `training/config.py` — dataclass config with all hyperparameters
- `tests/test_model.py` — shape tests, gradient tests
- `scripts/train.py` — entry point
- A `reports/training_run_1.md` after first full training describing what worked

---

### Milestone 4: Batched Beam Search on GPU

**Goal:** solve scrambled cubes using the trained network + beam search,
entirely on GPU.

This is the payoff milestone. Get this working and you have a functioning
solver.

#### Design

Convert the value network into a policy for search. Two options:

**Option A (cleaner): add a policy head.** After training V, either:
- Fine-tune: add a policy head and distill from greedy-V teacher.
- Or: re-train from scratch with a two-headed network (V + π). EfficientCube
  style but with V as regularizer.

**Option B (works, simpler): use V directly.** In beam search, score each
candidate by `-V(s)` (negative cost-to-go). Take top-B by this score.

Start with Option B. Move to A if Phase 4 results are underwhelming.

#### Beam Search Loop

```
def solve(scramble: [48], beam_width: int, max_depth: int) -> list[int]:
    beam_states = scramble.unsqueeze(0)          # [1, 48]
    beam_scores = torch.zeros(1)                  # [1]
    beam_last_face = torch.tensor([-1])           # [1]
    beam_paths = torch.zeros(1, 0, dtype=int)    # [1, 0] — action history
    
    for depth in range(max_depth):
        B = beam_states.shape[0]
        
        # Generate valid children (non-trivial moves only)
        mask = Cube.valid_next_moves_mask(beam_last_face)  # [B, 12] bool
        # Expand: [B, 12, 48]
        children = Cube.apply_all_moves(beam_states)
        # Mask invalid moves by setting their score to -inf
        
        # Score: V(s) gives cost-to-go, score = -V for top-k
        flat_children = children.reshape(B*12, 48)
        child_v = V_net(flat_children).reshape(B, 12)
        child_scores = -child_v  # high = promising
        child_scores[~mask] = -inf
        
        # Add parent scores (cumulative)
        # For beam search proper, score should be cumulative log-prob or
        # just the heuristic. DeepCubeA-style uses f = g + lambda*h.
        # For pure beam search with V: just -V of the child.
        # Tradeoff: g+h gives more optimal paths, but h alone is simpler
        # and the EfficientCube result shows it works. Start with h alone.
        
        # Check solved
        solved = Cube.is_solved(flat_children).reshape(B, 12)
        if solved.any():
            # Reconstruct path, return
            ...
        
        # Dedup + top-k
        flat_scores = child_scores.reshape(B * 12)
        flat_states = flat_children  # [B*12, 48]
        
        # Within-beam dedup via pack_state + torch.unique
        packed = pack_state(flat_states)  # [B*12] uint64
        # Keep best score per unique state (scatter_reduce amax)
        unique_packed, inverse = torch.unique(packed, return_inverse=True)
        best_score_per_unique = torch.full((unique_packed.numel(),), -inf)
        best_score_per_unique.scatter_reduce_(0, inverse, flat_scores, reduce='amax')
        # Recover index into flat arrays
        ...
        
        # Top-B
        top_scores, top_idx = best_scores.topk(beam_width)
        beam_states = unique_states[top_idx]
        beam_scores = top_scores
        beam_last_face = action_to_face(last_actions[top_idx])
        beam_paths = torch.cat([parent_paths[top_idx], new_actions[top_idx, None]], dim=1)
    
    return None  # failed within max_depth
```

#### State Packing for Dedup

- Pack `[48]` int8 state into a single `uint64` (or two — 48*3 = 144 bits of
  info since 6 colors fit in 3 bits). Actually two uint64s needed.
- For simplicity use a byte string / tuple hashing for correctness first,
  then optimize to packed uint64 later.
- `torch.unique` is available on MPS but benchmark carefully. Fallback:
  CPU round-trip using numpy `unique` might be faster for modest beam sizes.

#### Hyperparameters

- Beam width: start 1024, scale to 65536. Report results at multiple widths.
- Max depth: 30 (a bit past God's Number).
- Dedup: within-beam always; global closed-set optional (Phase 4b).

#### Acceptance Criteria

1. Correctness: solutions returned actually solve the cube (verify by
   replaying action sequence from the original scramble and checking it
   equals solved).
2. Solve rate: 100% on 1000 depth-20 scrambles with beam_width=4096.
3. Solution length: mean ≤ 24 moves with beam_width=4096 (DeepCubeA
   reports ~21 mean; aim for within 3 moves of that).
4. Throughput: solve a single scramble in <10s wall clock at beam_width=4096
   on M4 MPS.
5. Visualization: can render the full solve (initial scramble →
   intermediate states → solved) as a grid of cube renderings.

#### Deliverables

- `search/beam.py` — the beam search
- `search/pack.py` — state packing for dedup
- `tests/test_search.py`
- `scripts/solve.py` — CLI: takes a scramble, outputs solution + visualization
- `benchmarks/bench_search.py` — solve-rate and time vs. beam width tables
- `reports/search_results.md`

---

### Milestone 5 (Stretch): Policy Network + Beam Search with Policy

**Goal:** match or beat the V-based beam search using a policy head.

This is where you close the gap to EfficientCube. Skip if Milestone 4 results
are good enough.

Tasks:
- Add policy head to network.
- Either distill from V (supervised on greedy-V teacher) or train jointly.
- Beam search scoring by cumulative log-prob of action sequence.
- Compare: V-beam vs. π-beam vs. V+π-beam at equal compute.

---

### Milestone 6 (Stretch): Analysis

**Goal:** answer the "does it learn human algorithms?" question.

- Solve 1000 random scrambles. Extract all 3-move and 5-move subsequences.
- Frequency analysis: which subsequences appear more than chance?
- Check for conjugate patterns (aba⁻¹): DeepCubeA reports 13.1%.
- Compare solution traces to CFOP method: does the network ever pass
  through "white cross solved" states? (Probably not often.)
- Write up findings.

---

## General Engineering Guidelines

- **Device discipline**: one `device` variable, threaded through. All tensors
  should live on the same device. Add asserts liberally during development.
- **Dtype discipline**: states are `int8`, actions are `int64`, V outputs are
  `float32` (`float16` is fine for inference, not training).
- **No Python loops in hot paths**: env steps, scramble generation, beam
  expansion — all vectorized.
- **Reproducibility**: seed everything. Log seeds in training reports.
- **Checkpointing**: save model + optimizer + RNG state every N steps.
- **Config as code**: use dataclasses or Hydra, not argparse-soup.
- **Tests first for the env**: Milestone 1 correctness is non-negotiable —
  every subsequent milestone depends on it. If you suspect a bug anywhere,
  the first thing to re-verify is the move tables.

## File Layout

```
rubiks-rl/
  cube/
    __init__.py
    state.py           # constants, encoding
    env.py             # Cube class
    scrambles.py       # scramble generation
    viz.py             # rendering
    ENCODING.md        # doc
  model/
    __init__.py
    network.py
  training/
    __init__.py
    davi.py
    config.py
  search/
    __init__.py
    beam.py
    pack.py
  scripts/
    train.py
    solve.py
  benchmarks/
    bench_env.py
    bench_search.py
  tests/
    test_env.py
    test_scrambles.py
    test_model.py
    test_search.py
  reports/
    training_run_1.md
    search_results.md
  pyproject.toml
  README.md
  SPEC.md              # this file
```

## References

Primary:
- Agostinelli et al., DeepCubeA (2019): https://www.nature.com/articles/s42256-019-0070-z
- McAleer et al., ADI (2018): https://arxiv.org/abs/1805.07470
- Takano, EfficientCube (2023): https://arxiv.org/abs/2106.03157

Code:
- https://github.com/forestagostinelli/DeepCubeA — reference for move tables
  and training hyperparameters
- https://github.com/kyo-takano/efficientcube — reference for beam search
  with policy

Context:
- https://kociemba.org/cube.htm — classic references on cube group theory
  and move pruning
