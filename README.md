# rubik

A Rubik's Cube solver built from first principles, using modern DNN techniques running on modern desktop hardware — specifically the GPU on an M4 Mac Studio, via PyTorch's MPS backend.

**Stack:** Python 3.12 · PyTorch (MPS / BF16) · `uv` · `ruff` · `pytest`. Apple Silicon M-series.

## Why

Ever since I was a young computer programmer, I've often thought about solving a Rubik's Cube from scratch. I don't really care about the answers per se — I'm not interested in implementing the algorithms that other people have already discovered. It's more about tackling the combinatorial complexity of the move space and figuring the cube out as if nobody had ever solved it before.

Over the years I've tried it many times. My earlier attempts were pretty naive — I played with a lot of different encodings, move representations, search techniques — but I never really made any progress on the computational side. Modern desktop GPUs (in my case, an M4 Mac Studio) seem to put things within reach now, and having Claude Code with the custom agentic workflows I've built around it makes solving the cube accessible in a way it hadn't been in the past. There's also a side mission here: play with DNN and other techniques and see what happens on the GPU with a literal toy problem.


## General Idea

The general shape of the approach is straightforward: Start from a solved cube and scramble it, i.e. make random moves, a bunch of times — the more times you scramble it, the further the state is likely to be from solved. Train a neural network on batches of those scrambled states and the move count to get there in order to teach it to estimate the **depth**, i.e. number of moves required from the state to solved. This is conventionally called the **Value** network, where the value is correlated with the expected **depth**. With a trained value network, you then flip the process around: take a fresh scrambled cube and walk it back toward solved, at each step looking at all states resulting from one of 12 rotations, using the value network to predict the depth from that adjacent state, keeping the ones with the smallest expected depth, and iterating. The walk is implemented as a **batched beam search** on the GPU — exploring many candidate paths in parallel and keeping the most promising ones each step until one lands on the solved state.

That's the high-level picture. The tricky bit is that number of scrambles taken to get to a particular state, is systematically higher than the true ideal depth. There are a lot of move sequences that loop back to the same states reachable from solved in fewer steps, as you explore higher depths, the drift grows. In practice what that means is the network's MAE consistently drifts higher (toward the mean error of all states), even though in a different sense the network is actually improving, i.e. becoming more useful.

The key insight/methodology/trick I got from previous **DeepCubeA** efforts (references below). DAVI (Deep Approximate Value Iteration) trains a neural network to estimate cost-to-go by repeatedly generating random scrambles, computing Bellman targets from a frozen copy of the network, and regressing the live network toward those targets — with the frozen copy periodically synced to the live one. The trick is that the only ground-truth signal is V(solved) = 0, hard-coded as an override; everything else is bootstrapped by iteratively refining the network's own predictions. I know what all that means now, I'm not sure I would have come up with it on my own.

I started on the **2x2** cube because it's small enough to fully enumerate (3.6M states; a BFS oracle gives ground-truth distance for every state) but rich enough that all the same algorithmic plumbing has to work. Once the loop ran end-to-end on 2x2, the same code path took on 3x3 by swapping the spec.

## How it works

The shape of the algorithm follows a **value-iteration** pattern that's appeared in cube-solving research before, but the specific design — architecture, training schedule, search settings — is all built up here from running experiments on this hardware. No hyperparameters borrowed from prior work. Every load-bearing knob has been picked by running small tier-0 experiments on this specific machine and this specific problem; the project's point is to learn what the loop *actually* does on an M4 Max GPU, not to recreate someone else's published numbers.

### State, moves, and the parameterized cube

A single `CubeSpec` describes everything size-specific in one place: sticker count, faces, color count, and precomputed move tables that say "applying move R takes sticker 17 to position 25." The whole pipeline — environment, training, search, visualizers — consumes a `CubeSpec`. Adding a new puzzle size or move set is a config change, not a fork.

State is a flat sticker-position permutation; the network sees a one-hot color-per-position. The current move set is **QTM** — six faces × two directions = 12 quarter-turn moves, no double-turns.

There are scripts and tests for generating this stuff and a favorite pattern of mine is generating human-verifiable outputs in parallel, for this project often html: [2x2 rotations](visuals/oracle_rotations_2x2.html) [3x3 rotations](visuals/oracle_rotations_3x3.html)

### The value network

A residual MLP with layer norm. Training generates cubes by applying random scrambles to a solved cube; for each cube, the target value is computed one move ahead — try every move, and the target is `1 + V_target(s')` for the move that lands on the state V_target thinks is closest to solved. `V_target` is a slowly-updated copy of the network being trained, so the optimizer is chasing its own slightly-stale opinion. From random initialization this bootstraps a useful value function, no labels needed.

The whole training loop runs on the M4 Max GPU via PyTorch's MPS backend, with BF16 inference for search.

### Search

At inference time, scrambles are solved with **batched beam search** on the GPU. Greedy decoding doesn't work well — the network's ordering of "good" moves is noisier than its overall distance estimate, so the argmax gets fooled. Beam search at width 256 is the production default; it consumes the network's output quality where it actually lives — across the top-K, not at a single argmax.

The eval pipeline and the production solver share the same primitive (`beam_solve_batch`), so improvements to either one are improvements to both.

## Where we are

The work has gone in roughly the order you'd expect, milestone by milestone, each one a concrete thing that had to work before the next one made sense:

- **A slow, hand-rolled cube oracle.** Corners as position + orientation, moves applied as physical rotations. Slow but obviously correct, and easy to reason about. This is the truth that everything else gets checked against.
- **A fast tensor cube.** Same semantics as the oracle but moves are precomputed sticker permutations applied as batched index-gathers on the GPU. Equivalence-tested against the oracle on a corpus of random move sequences — same input, same output, every time.
- **A visualization stack.** ASCII for tests and the REPL; HTML/SVG renders for human eyeballing. The "open the file in a browser and look at it" loop turned out to be one of the highest-leverage things in the project.
- **Performance methodology before performance optimization.** Before tuning anything, I built the measurement story for MPS on this machine — batch-sensitivity sweeps, throughput curves, the experiment-loop pattern (`experiments/<name>/` with reproducible scripts and a hand-written `intuition.md`).
- **DAVI on 2x2, then beam search on 2x2.** The 2x2 has only 3.6M states, so a BFS oracle gives ground-truth distance-to-solved for every single one of them. That's a rare thing to have when you're training a value network — you can compare what the net thinks against what's actually true. Trained the network, then built the beam search on top, then evaluated against the oracle. There's a small documented gap at the deepest depths that I have plans for but haven't closed.
- **3x3 by spec swap, not by fork.** The whole codebase is parameterized through a single `CubeSpec`. When 2x2 was working, 3x3 came online by adding a 3x3 spec — the env, training, search, and visualizers all pick it up automatically. No parallel 2x2/3x3 code paths anywhere in the repo.
- **Several 3x3 training cycles.** Most of the recent work has been on the 3x3 training loop — dialing in scramble depth, sync intervals, network capacity, and warm-start vs fresh-init. The current best is what's in the v0.1.0 release.

**v0.1.0** ([release](https://github.com/jgriffin/rubik/releases/tag/v0.1.0)) is the best 3x3 ValueNet I've trained so far — 100k training steps total, structured as a 30k fresh-init phase plus a 70k warm-start continuation.

### Solve rate by depth

Random-walk scrambles, beam-256, BF16. n=50 per depth at d≥7; smaller n at the very shallow depths (d=1..6) where every state is trivially reachable.

| depth | solve rate |
|----------:|-----------:|
| 1   | 1.00 |
| 2   | 1.00 |
| 3   | 1.00 |
| 4   | 1.00 |
| 5   | 1.00 |
| 6   | 1.00 |
| 7   | 1.00 |
| 8   | 1.00 |
| 9   | 1.00 |
| 10  | 1.00 |
| 11  | 1.00 |
| 12  | 1.00 |
| 13  | 1.00 |
| 14  | 0.98 |
| 15  | 0.98 |
| 16  | 0.88 |
| 17  | 0.82 |
| 18  | 0.72 |
| 19  | 0.82 |
| 20  | 0.78 |
| 25  | 0.80 |
| 30  | 0.76 |

I'm happy enough with where this lands to pivot for a bit and play with the visualization side next — a proper 3D viewer, a step-by-step solve trace, that kind of thing. I don't yet know how good "good enough" needs to be on the visualization side; that's part of what I'll find out.

A few honest caveats so the framing stays straight:

- Not a fully-trained model. d=20–30 capability is still climbing at step 100k — there's headroom in another long training run.
- Not a competitive solver. A tuned IDA* with classical pruning tables, or Kociemba's algorithm, will wipe the floor with this both in speed and solution length. What this network has going for it is that it figured out the geometry on its own without being told anything about cubes.
- Not a generic solver. QTM moves only, M4 Max / MPS / BF16 inference, no comparison against classical solvers, no human-readable solution explanation.
- No 3D viewer or web frontend yet. ASCII renderer for tests, HTML+SVG for static previews; that's the visualization story for now.
- 2x2 has fallen behind 3x3 on a few quality-of-life details (provenance bundle, capture pipeline). If I revisit it, I'll mirror.

## Running it

```bash
# Setup (uv-managed, Python 3.12)
uv venv && uv sync

# Tests
uv run pytest

# Train a 3x3 model (~3 hours on M4 Max for 30k steps)
uv run python experiments/davi-3x3/run.py \
  --config experiments/davi-3x3/configs/ln_kmax30.yaml

# Eval an existing checkpoint
uv run python scripts/beam_eval_model.py path/to/net_final.pt --config fast_kmax30
```

## References

**DeepCubeA** - Solving the Rubik's Cube with Deep Reinforcement Learning and Search
  Agostinelli, McAleer, Shmakov, Baldi (Nature Machine Intelligence, 2019)
  https://www.nature.com/articles/s42256-019-0070-z
  https://github.com/forestagostinelli/DeepCubeA
  https://deepcube.igb.uci.edu/

## License

Wrote this for myself — do whatever you want with it.

