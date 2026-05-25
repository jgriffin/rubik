# rubik

A Rubik's Cube solver built from first principles, using modern DNN techniques running on modern desktop hardware — specifically the GPU on an M4 Mac Studio, via PyTorch's MPS backend.

**Stack:** Python 3.12 · PyTorch (MPS / BF16) · `uv` · `ruff` · `pytest`. Apple Silicon M-series.

## Why

Ever since I was a young computer programmer, I've often thought about solving a Rubik's Cube from scratch. I don't really care about the answers per se — I'm not interested in implementing the algorithms that other people have already discovered. It's more about tackling the combinatorial complexity of the move space and figuring the cube out as if nobody had ever solved it before.

Over the years I've tried it many times. My earlier attempts were pretty naive — I played with a lot of different encodings, move representations, search techniques — but I never really made any progress on the computational side. Modern desktop GPUs (in my case, an M4 Mac Studio) seem to put things within reach now, and having Claude Code with the custom agentic workflows I've built around it makes solving the cube accessible in a way it hadn't been in the past. There's also a side mission here: play with DNN and other techniques and see what happens on the GPU with a literal toy problem.


## How it works

### Random walks from solved estimating depth

The general shape of the approach is straightforward: Start from a solved cube and scramble it, i.e. make random moves, a bunch of times — the more times you scramble it, the further the state is likely to be from solved. Train a neural network on batches of those scrambled states and the move count to get there in order to teach it to estimate the **depth**, i.e. number of moves required from the state to solved. This is conventionally called the **Value** network, where the value is correlated with the expected **depth**. 

### Informed searches back to solved

With a trained value network, you then flip the process around. take a fresh scrambled cube and walk it back toward solved, at each step looking at all states resulting from one of 12 rotations, using the value network to predict the depth from that adjacent state, keeping the ones with the smallest expected depth, and iterating. The walk is implemented as a **batched beam search** on the GPU — exploring many candidate paths in parallel and keeping the most promising ones each step until one lands on the solved state.

### DAVI - Bellman + groundstate, with wavefront

The tricky bit is that number of scrambles taken to get to a particular state, is systematically higher than the true ideal depth. There are a lot of move sequences that loop back to the same states reachable from solved in fewer steps, as you explore higher depths, the drift grows. In practice what that means is the network's MAE consistently drifts higher (toward the mean error of all states), even though in a different sense the network is actually improving, i.e. becoming more useful.

The key insight/methodology/trick I got from previous **DeepCubeA** efforts (references below). DAVI (Deep Approximate Value Iteration) trains a neural network to estimate cost-to-go by repeatedly generating random scrambles, computing Bellman targets from a frozen copy of the network, and regressing the live network toward those targets — with the frozen copy periodically synced to the live one. The trick is that the only ground-truth signal is V(solved) = 0, hard-coded as an override; everything else is bootstrapped by iteratively refining the network's own predictions. I know what all that means now, I'm not sure I would have come up with it on my own. The hyperparameters and architectural choices, on the other hand, are mine — picked by running small experiments on this hardware rather than borrowed from prior work.

### Start smaller first

I started on the **2x2** cube because it's small enough to fully enumerate (3.6M states; a BFS oracle gives ground-truth distance for every state) but rich enough that all the same algorithmic plumbing has to work. Once the loop ran end-to-end on 2x2, the same code path took on 3x3 by swapping the spec.

## A tour of the code

If you want to read the code, here's where to start. Everything is parameterized on a single `CubeSpec` — add a new puzzle size or move set by adding a spec, not by forking.

- **`src/rubik/cube/spec.py` — `CubeSpec`.** The single source of truth: sticker count, face count, color count, precomputed move tables that say "applying move R takes sticker 17 to position 25." Every layer downstream consumes one of these.
- **`src/rubik/cube/env.py` — the fast tensor cube.** `apply_moves`, `apply_all_moves`, `is_solved`, `random_scrambles`, `valid_next_moves_mask`. Pure tensor ops on MPS, batched, no Python loops in the hot path.
- **`src/rubik/oracle/cubie.py` — the slow witness.** Hand-rolled corners-as-position-plus-orientation cube. Slow, obviously correct, used in equivalence tests against the env.
- **`src/rubik/oracle/v_star_2x2.py`, `v_star_bounded_3x3.py` — V\* oracles.** Full BFS for 2x2 (3.6M states, every distance known), bounded BFS up to depth 6 for 3x3. Ground-truth labels for evaluation.
- **`src/rubik/model/network.py` — the value network.** Residual MLP parameterized on `CubeSpec` (input dim derives from `n_stickers × n_colors`). `body_widths`, `n_residual_blocks`, and `normalization` are required kwargs — no opinionated default. fp32 during training, BF16 at inference.
- **`src/rubik/training/davi.py` — the DAVI step.** `compute_targets` (Bellman 1-step), `davi_step` (a training step), `sync_target` (refresh the frozen target net). The actual training run loops live one level out, in `experiments/davi-3x3/run.py` and `experiments/davi-2x2/run.py`.
- **`src/rubik/search/beam.py` — `beam_solve_batch`.** The single primitive shared by the eval pipeline and the production solver. Width 256 is the default; greedy (width-1) doesn't work well — the network's *ordering* of moves is noisier than its overall distance estimate, so the argmax gets fooled. Beam search consumes the ordering where it actually lives, across the top-K.
- **`src/rubik/notation/{moves,state}.py`.** The 12 QTM moves (`R R' L L' U U' D D' F F' B B'`), sticker-indexing conventions.
- **`src/rubik/viz/`.** `ascii.py` for tests/REPL, `svg.py` + `colors.py` for the static-HTML preview pattern (e.g. [2x2 rotations](visuals/oracle_rotations_2x2.html), [3x3 rotations](visuals/oracle_rotations_3x3.html)).
- **`experiments/davi-2x2/`, `experiments/davi-3x3/`.** Each has reproducible run scripts, a `runs/` directory of training output, an `analysis/` layer (analyze → capture → render), and a top-level `results.md` + `intuition.md` capturing observations and hypotheses across cycles.
- **`src/rubik/server/`.** FastAPI app (`/api/health`, `/api/scramble`, `/api/solve`) that wraps the same `beam_solve_batch` the training/eval pipeline uses. The web demo's FastAPI/MPS solver path hits this.
- **`web/`.** React + Vite frontend. `web/src/solver/` is the runtime-agnostic solver layer: `Solver.ts` is the interface; `ApiSolver.ts` wraps the FastAPI path; `OnnxSolver.ts` wraps `onnxruntime-web` with WebGPU primary and WASM fallback; `beam.ts` is the TypeScript port of `beam_solve_batch` taking a `ValueFn` so the runtime is pluggable. `SolverSwitch.tsx` toggles between them. Block B's parity numbers and Block D's perf characterization both live in repo.
- **`experiments/browser-solve/`.** M11 Block D — three-way latency comparison (FastAPI/MPS vs ONNX/WebGPU vs ONNX/WASM) across beam widths {32, 64, 128, 256} on a fixed 10-row corpus. `intuition.md` has the hypotheses; `results/perf-comparison.html` has the chart.
- **`bin/site`.** Dev-site launcher. `bin/site up` ensures the FastAPI backend (port 8000) and the Vite frontend (port 5173) are both running; idempotent, fingerprints existing processes, manages pid files under `.dev/`. `bin/site down` / `status` / `logs` round it out.

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

After v0.1.0 I pivoted to the visualization side and then to deployment. The result is a small **web UI** — scramble a cube, solve it, step through the solution (or auto-play it), and edit the move sequence directly. You can switch between 2x2 and 3x3, toggle the renders between 2D, 3D, or both side-by-side, set the scramble depth, and rearrange the layout across a few column counts. It runs the same beam search the eval pipeline uses; on the FastAPI/MPS path the round-trip on a width=128 solve is ~93 ms median.

Then I made the whole thing **deployable without a server**. The trained `.pt` exports to ONNX with parity verified to a few × 10⁻⁵ (CPU PyTorch vs onnxruntime). The Python beam search ports to TypeScript; the value-net forward gets injected as a `ValueFn`, so the algorithm is identical and the runtime is pluggable. The browser loads the 61 MB `.onnx` via `onnxruntime-web` and runs the entire solve on the user's GPU through WebGPU. A `Solver` abstraction toggles between the FastAPI/MPS path and the in-browser ONNX path in real time from a switch in the page header. On page load it probes `/api/health` and auto-picks the best available — FastAPI/MPS when running locally with the server up; ONNX/WebGPU when there's no backend (e.g. a static deploy). WebGPU lands at ~394 ms median per width=128 solve on an M4 Max — ~4× the MPS server but well under "feels instant." WASM is the fallback for browsers without WebGPU, ~22× slower than WebGPU. Numbers, methodology, and an `intuition.md` are in `experiments/browser-solve/`.

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
- Web demo is live at **[rubik-johngrif.vercel.app](https://rubik-johngrif.vercel.app)** — a static Vercel deploy running the entire solve in-browser via ONNX/WebGPU (no backend; WASM fallback where WebGPU is unavailable). The FastAPI/MPS path still works locally via `bin/site up`.
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

# Run the web demo (FastAPI :8000 + Vite :5173) — both servers, idempotent.
bin/site up
# → open http://localhost:5173/
bin/site status     # see what's running
bin/site down       # stop both
```

## References

**DeepCubeA** - Solving the Rubik's Cube with Deep Reinforcement Learning and Search
  Agostinelli, McAleer, Shmakov, Baldi (Nature Machine Intelligence, 2019)
  https://www.nature.com/articles/s42256-019-0070-z
  https://github.com/forestagostinelli/DeepCubeA
  https://deepcube.igb.uci.edu/

## License

Wrote this for myself — but do whatever you want with it.

