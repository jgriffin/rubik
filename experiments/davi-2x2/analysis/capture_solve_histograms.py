"""Capture per-attempt solve-length distributions for each run's final checkpoint.

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
   test depth (1..13 contiguous, matching eval.py's default depth set).
3. Writes per-(run, depth) raw solve-length arrays to
   ``solve_histograms.json`` under the experiment's results/ dir
   (experiments/davi-2x2/results/) so the renderer can read one file.

Usage:
    uv run python experiments/davi-2x2/analysis/capture_solve_histograms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ValueNet
from rubik.solve import greedy_solve_batch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = EXPERIMENT_DIR / "runs"
OUT_PATH = EXPERIMENT_DIR / "results" / "solve_histograms.json"

RUNS: list[tuple[str, str]] = [
    ("cycle-1 baseline-30k  (K=18, sync=500)", "baseline-30k"),
    ("cycle-3 sync500_kmax20-30k  (K=20)", "sync500_kmax20-30k"),
    ("cycle-3 sync1000_kmax20-30k  (K=20)", "sync1000_kmax20-30k"),
]

# Match the trained net architecture (all four runs share this).
BODY_WIDTHS = [4096, 1024]
N_RESIDUAL_BLOCKS = 4
NORMALIZATION = "bn"

DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
N_PER_DEPTH = 200
DEPTH_BUDGET_FACTOR = 2  # matches eval.py default
SEED = 17  # different from training-eval seed; just reproducibility


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


def greedy_solve_with_lens(
    net: torch.nn.Module,
    *,
    depth: int,
    n: int,
    generator: torch.Generator,
) -> list[int]:
    """Return per-attempt solve_lens (length n). -1 means failed.

    Thin wrapper over ``rubik.solve.greedy_solve_batch`` that handles the
    per-depth scramble generation + ``2 * depth`` budget convention used
    by every histogram-style writeup in this experiment dir.
    """
    spec = CUBE_2X2
    states, _ = random_scrambles(
        spec, batch_size=n, depth=depth, generator=generator, prune_same_face=True
    )
    solve_lens = greedy_solve_batch(
        net, spec, states, max_steps=DEPTH_BUDGET_FACTOR * depth
    )
    return solve_lens.cpu().tolist()


def main() -> None:
    device = torch.device("mps")
    out: dict[str, dict] = {
        "config": {
            "n_per_depth": N_PER_DEPTH,
            "depth_budget_factor": DEPTH_BUDGET_FACTOR,
            "depths": list(DEPTHS),
            "seed": SEED,
        },
        "runs": {},
    }

    for label, run_subdir in RUNS:
        run_dir = BASELINE_DIR / run_subdir
        if not run_dir.exists():
            print(f"skip {label}: {run_dir} not found")
            continue
        ckpt_path = find_terminal_checkpoint(run_dir)
        print(f"\n=== {label} ===")
        print(f"  ckpt: {ckpt_path.relative_to(REPO_ROOT)}")
        net = load_net(ckpt_path, device)
        # Fresh generator per run for reproducibility.
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        per_depth: dict[str, list[int]] = {}
        for d in DEPTHS:
            lens = greedy_solve_with_lens(net, depth=d, n=N_PER_DEPTH, generator=gen)
            n_solved = sum(1 for x in lens if x >= 0)
            avg = sum(x for x in lens if x >= 0) / n_solved if n_solved > 0 else None
            print(
                f"  d={d:>2}: solved {n_solved}/{N_PER_DEPTH} "
                f"({100 * n_solved / N_PER_DEPTH:5.1f}%)  "
                f"avg_len={avg:.2f}"
                if avg is not None
                else f"  d={d:>2}: solved {n_solved}/{N_PER_DEPTH}  avg_len=N/A"
            )
            per_depth[str(d)] = lens
        out["runs"][run_subdir] = {
            "label": label,
            "ckpt": str(ckpt_path.relative_to(REPO_ROOT)),
            "lens": per_depth,
        }

    with OUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
