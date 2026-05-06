"""Capture per-attempt solve-length distributions for each run × strategy × depth.

The eval pipeline's ``greedy_solve`` (eval.py) only logs per-depth summary
stats (``solve_rate``, ``avg_solve_len``). For the post-hoc histogram view
we need the raw per-attempt ``solve_lens`` tensor: each value is either
``-1`` (failed within the 2*depth move budget) or the number of moves
used to reach solved.

This script:
1. Loads each run's terminal checkpoint (``net_final.pt`` if present, else
   the last ``net_step_*.pt``). Handles both the legacy bare-state-dict
   format (M5 baseline-30k) and the new dict format ({net_state, ...}).
2. Runs a high-sample-size greedy solve (``n_per_depth=200``) at each
   test depth (1..14 contiguous, matching eval.py's default depth set
   plus the extra +1 to cover 2x2 QTM diameter), under each
   **sampling strategy**:
   - ``random_walk_depth``: scramble = random walk of length ``d`` (V* ≤ d).
   - ``v_star_stratified``: state drawn uniformly from ``{V* == d}`` via
     the BFS oracle table. ``rotate=False`` so the solver's solved-state
     equality check still recognizes solved cubes.
3. Writes the raw per-attempt arrays to ``solve_histograms.json`` under
   ``experiments/davi-2x2/results/`` so the renderer can read one file.

   Schema::

       { "config": {...},
         "runs": { "<run_subdir>": {
             "label": "...",
             "ckpt": "...",
             "<strategy>": { "lens": { "<depth>": [...] } },
             ...
         } } }

Seeds are deterministic per ``(run_idx, strategy_idx, depth)`` so adding
a run or strategy doesn't reshuffle previously-captured cells.

Usage:
    uv run python experiments/davi-2x2/analysis/capture_solve_histograms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ValueNet
from rubik.oracle.v_star_2x2 import load_v_star_arrays, sample_states_at_v_star
from rubik.solve import greedy_solve_batch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = EXPERIMENT_DIR / "runs"
OUT_PATH = EXPERIMENT_DIR / "results" / "solve_histograms.json"
V_STAR_PATH = REPO_ROOT / "data" / "v_star_2x2.npz"

RUNS: list[tuple[str, str]] = [
    ("cycle-1 baseline-30k  (K=18, sync=500)", "baseline-30k"),
    ("cycle-3 sync500_kmax20-30k  (K=20)", "sync500_kmax20-30k"),
    ("cycle-3 sync1000_kmax20-30k  (K=20)", "sync1000_kmax20-30k"),
    ("cycle-4 kmax28_warm-30k  (K=28, warm-start)", "kmax28_warm-30k"),
]

STRATEGIES: tuple[str, ...] = ("random_walk_depth", "v_star_stratified")

# Match the trained net architecture (all four runs share this).
BODY_WIDTHS = [4096, 1024]
N_RESIDUAL_BLOCKS = 4
NORMALIZATION = "bn"

DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
N_PER_DEPTH = 200
DEPTH_BUDGET_FACTOR = 2  # matches eval.py default
SEED_BASE = 17  # any int — combined with (run_idx, strategy_idx, depth) per cell


def find_terminal_checkpoint(run_dir: Path) -> Path:
    """Return net_final.pt if present, else the highest-step net_step_*.pt."""
    final = run_dir / "net_final.pt"
    if final.exists():
        return final
    candidates = sorted(
        run_dir.glob("net_step_*.pt"),
        key=lambda p: int(p.stem.replace("net_step_", "")),
    )
    if not candidates:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    return candidates[-1]


def load_net(ckpt_path: Path, device: torch.device) -> ValueNet:
    net = ValueNet(
        CUBE_2X2,
        body_widths=BODY_WIDTHS,
        n_residual_blocks=N_RESIDUAL_BLOCKS,
        normalization=NORMALIZATION,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "net_state" in ckpt:
        net.load_state_dict(ckpt["net_state"])
    else:
        net.load_state_dict(ckpt)
    net.eval()
    return net


def _cell_seed(run_idx: int, strategy_idx: int, depth: int) -> int:
    """Stable seed for a (run, strategy, depth) cell.

    Mixing run_idx and strategy_idx with the depth and a base constant
    keeps each cell independently reproducible — adding a new run or
    strategy doesn't shift seeds for previously-captured cells.
    """
    # Deterministic mix; the specific arithmetic doesn't matter as long
    # as distinct (run, strategy, depth) → distinct seed.
    return SEED_BASE + 100_003 * run_idx + 1_009 * strategy_idx + depth


def greedy_solve_random_walk(
    net: torch.nn.Module,
    *,
    depth: int,
    n: int,
    seed: int,
) -> list[int]:
    """``random_walk_depth``: scramble = random walk of length ``d``."""
    spec = CUBE_2X2
    gen = torch.Generator(device="cpu").manual_seed(seed)
    states, _ = random_scrambles(
        spec, batch_size=n, depth=depth, generator=gen, prune_same_face=True
    )
    solve_lens = greedy_solve_batch(
        net, spec, states, max_steps=DEPTH_BUDGET_FACTOR * depth
    )
    return solve_lens.cpu().tolist()


def greedy_solve_v_star_stratified(
    net: torch.nn.Module,
    *,
    depth: int,
    n: int,
    seed: int,
    v_star_states: np.ndarray,
    v_star_depths: np.ndarray,
) -> list[int]:
    """``v_star_stratified``: state ~ Uniform({s : V*(s) == depth}).

    ``rotate=False`` is critical when feeding the solver: the solver
    tests is_solved(state) against ``spec.solved_state`` exactly, so a
    rotated solved state would not be recognized as solved.
    """
    spec = CUBE_2X2
    rng = np.random.default_rng(seed)
    states_np = sample_states_at_v_star(
        v_star_states, v_star_depths, target_v_star=depth, n=n, rng=rng, rotate=False
    )
    states = torch.from_numpy(states_np.astype(np.int64))
    solve_lens = greedy_solve_batch(
        net, spec, states, max_steps=DEPTH_BUDGET_FACTOR * depth
    )
    return solve_lens.cpu().tolist()


def _summarize(lens: list[int], n: int) -> str:
    n_solved = sum(1 for x in lens if x >= 0)
    if n_solved == 0:
        return f"solved {n_solved}/{n}  avg_len=N/A"
    avg = sum(x for x in lens if x >= 0) / n_solved
    return (
        f"solved {n_solved}/{n} ({100 * n_solved / n:5.1f}%)  "
        f"avg_len={avg:.2f}"
    )


def main() -> None:
    device = torch.device("mps")
    out: dict[str, dict] = {
        "config": {
            "n_per_depth": N_PER_DEPTH,
            "depth_budget_factor": DEPTH_BUDGET_FACTOR,
            "depths": list(DEPTHS),
            "strategies": list(STRATEGIES),
            "seed_base": SEED_BASE,
        },
        "runs": {},
    }

    # Load V* table once (shared across all runs × depths for the
    # v_star_stratified strategy).
    if not V_STAR_PATH.exists():
        raise FileNotFoundError(
            f"V* table not found at {V_STAR_PATH} — required for v_star_stratified"
        )
    v_star_states, v_star_depths = load_v_star_arrays(V_STAR_PATH)
    print(f"loaded V* table: {len(v_star_states)} canonical states")

    for run_idx, (label, run_subdir) in enumerate(RUNS):
        run_dir = BASELINE_DIR / run_subdir
        if not run_dir.exists():
            print(f"skip {label}: {run_dir} not found")
            continue
        ckpt_path = find_terminal_checkpoint(run_dir)
        print(f"\n=== {label} ===")
        print(f"  ckpt: {ckpt_path.relative_to(REPO_ROOT)}")
        net = load_net(ckpt_path, device)

        run_entry: dict = {
            "label": label,
            "ckpt": str(ckpt_path.relative_to(REPO_ROOT)),
        }

        for strategy_idx, strategy in enumerate(STRATEGIES):
            print(f"  -- strategy: {strategy}")
            per_depth: dict[str, list[int]] = {}
            for d in DEPTHS:
                seed = _cell_seed(run_idx, strategy_idx, d)
                if strategy == "random_walk_depth":
                    lens = greedy_solve_random_walk(
                        net, depth=d, n=N_PER_DEPTH, seed=seed
                    )
                elif strategy == "v_star_stratified":
                    lens = greedy_solve_v_star_stratified(
                        net,
                        depth=d,
                        n=N_PER_DEPTH,
                        seed=seed,
                        v_star_states=v_star_states,
                        v_star_depths=v_star_depths,
                    )
                else:
                    raise ValueError(f"unknown strategy: {strategy}")
                print(f"     d={d:>2}: {_summarize(lens, N_PER_DEPTH)}")
                per_depth[str(d)] = lens
            run_entry[strategy] = {"lens": per_depth}

        out["runs"][run_subdir] = run_entry

    with OUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
