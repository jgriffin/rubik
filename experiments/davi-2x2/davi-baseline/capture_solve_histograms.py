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
   test depth (1, 3, 5, 7, 9, 11, 13).
3. Writes per-(run, depth) raw solve-length arrays to
   ``solve_histograms.json`` in the run's parent dir
   (experiments/davi-2x2/davi-baseline/) so the renderer can read one file.

Usage:
    uv run python experiments/davi-2x2/davi-baseline/capture_solve_histograms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rubik.cube.env import apply_all_moves, is_solved, random_scrambles
from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ValueNet

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "experiments" / "davi-2x2" / "davi-baseline" / "runs"
OUT_PATH = (
    REPO_ROOT / "experiments" / "davi-2x2" / "davi-baseline" / "solve_histograms.json"
)

RUNS: list[tuple[str, str]] = [
    ("baseline-30k (sync=500, fresh)", "baseline-30k"),
    ("warm sync=100 (killed)", "baseline-continue-sync100-30k"),
    ("warm sync=500 (control)", "baseline-continue-sync500-7k"),
    ("warm sync=2000", "baseline-continue-sync2000-7k"),
]

# Match the trained net architecture (all four runs share this).
BODY_WIDTHS = [4096, 1024]
N_RESIDUAL_BLOCKS = 4
NORMALIZATION = "bn"

DEPTHS = (1, 3, 5, 7, 9, 11, 13)
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


@torch.no_grad()
def greedy_solve_with_lens(
    net: torch.nn.Module,
    *,
    depth: int,
    n: int,
    generator: torch.Generator,
) -> list[int]:
    """Return per-attempt solve_lens (length n). -1 means failed."""
    spec = CUBE_2X2
    device = next(net.parameters()).device
    budget = DEPTH_BUDGET_FACTOR * depth

    states, _ = random_scrambles(
        spec,
        batch_size=n,
        depth=depth,
        generator=generator,
        prune_same_face=True,
    )
    states = states.to(device)

    active = torch.ones(n, dtype=torch.bool, device=device)
    solve_lens = torch.full((n,), -1, dtype=torch.int64, device=device)
    already = is_solved(states, spec)
    if already.any():
        solve_lens[already] = 0
        active = active & ~already

    for step_idx in range(budget):
        if not active.any():
            break
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        active_states = states[active_idx]
        children = apply_all_moves(active_states, spec)
        a = active_states.shape[0]
        children_flat = children.reshape(a * spec.n_moves, spec.n_stickers)
        child_v = net(children_flat).reshape(a, spec.n_moves)
        solved_children = is_solved(children_flat, spec).reshape(a, spec.n_moves)
        child_v = torch.where(
            solved_children, torch.full_like(child_v, -1e9), child_v
        )
        best_move = child_v.argmin(dim=1)
        new_states = children[torch.arange(a, device=device), best_move]
        states[active_idx] = new_states
        now_solved = is_solved(new_states, spec)
        if now_solved.any():
            solved_active_idx = active_idx[now_solved]
            solve_lens[solved_active_idx] = step_idx + 1
            active[solved_active_idx] = False

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
            n_failed = N_PER_DEPTH - n_solved
            avg = (
                sum(x for x in lens if x >= 0) / n_solved if n_solved > 0 else None
            )
            print(
                f"  d={d:>2}: solved {n_solved}/{N_PER_DEPTH} "
                f"({100 * n_solved / N_PER_DEPTH:5.1f}%)  "
                f"avg_len={avg:.2f}" if avg is not None else
                f"  d={d:>2}: solved {n_solved}/{N_PER_DEPTH}  avg_len=N/A"
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
