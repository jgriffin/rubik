"""Capture per-attempt solve-length distributions for 3x3 runs.

Captures (run × strategy × method × depth) cells. Mirrors
``experiments/davi-2x2/analysis/capture_solve_histograms.py``
cell-for-cell, swapping ``CUBE_2X2`` → ``CUBE_3X3`` and pointing at the
3x3 bounded oracle (P1b — ``data/v_star_bounded_3x3_k6.npz``) for the
``v_star_stratified`` strategy.

Phase-1 caveat — V*-stratified is only meaningful at d ≤ 6 (the bounded
oracle's K). For d > 6 the bounded oracle has no entries; this script
restricts the ``v_star_stratified`` strategy to ``DEPTHS_V_STAR``
(``range(1, 7)``) and runs only ``random_walk_depth`` over the full
``DEPTHS`` range. See plan §"Cycle reporting pipeline" — extend the
existing capture, never parallel-build.

Net architecture constants (BODY_WIDTHS / N_RESIDUAL_BLOCKS /
NORMALIZATION) are TBD in the P1a scaffold — earned in P2a's calibration
+ P2b's first run, then filled in here so the capture loads each
checkpoint with the matching architecture. Until then the script
short-circuits with a stub message rather than attempting to load nets
with wrong shapes.

Usage:
    uv run python experiments/davi-3x3/analysis/capture_solve_histograms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_3X3
from rubik.model.network import ValueNet
from rubik.search import beam_solve_batch
from rubik.solve import greedy_solve_batch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = EXPERIMENT_DIR / "runs"
OUT_PATH = EXPERIMENT_DIR / "results" / "solve_histograms.json"
# Bounded V* table (P1b) — 1M states, depths 0..6. Only consulted for
# the v_star_stratified strategy at depths <= K.
V_STAR_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"
V_STAR_K = 6  # plan P1b — bounded oracle covers depths 0..6

# (label, run_subdir) — empty until first run lands in P2b. Appended as
# cycles come in. Same RUNS shape as davi-2x2's capture.
RUNS: list[tuple[str, str]] = []

STRATEGIES: tuple[str, ...] = ("random_walk_depth", "v_star_stratified")

# (method_name, beam_width). Greedy uses greedy_solve_batch; beam uses
# beam_solve_batch. Inputs are sampled once per cell and shared across
# methods for clean comparison.
METHODS: tuple[tuple[str, int], ...] = (("greedy", 1), ("beam", 256))
BEAM_MAX_STEPS = 20

# Architecture constants — TBD, earned in P2a + filled here in P2b once
# the first run picks a config. Empty/None in P1a scaffold; a non-stub
# capture run will need these before loading checkpoints.
BODY_WIDTHS: list[int] | None = None
N_RESIDUAL_BLOCKS: int | None = None
NORMALIZATION: str | None = None

# Walk-depth bins captured for both strategies. random_walk_depth uses
# the full range; v_star_stratified is restricted to d ≤ V_STAR_K.
DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
DEPTHS_V_STAR = tuple(d for d in DEPTHS if d <= V_STAR_K)
N_PER_DEPTH = 200
DEPTH_BUDGET_FACTOR = 2  # greedy: matches eval.py default (max_steps = 2*depth)
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
    if BODY_WIDTHS is None or N_RESIDUAL_BLOCKS is None or NORMALIZATION is None:
        raise NotImplementedError(
            "BODY_WIDTHS / N_RESIDUAL_BLOCKS / NORMALIZATION not set — "
            "fill these in P2b once the first 3x3 config is chosen. "
            "See plan §P2b."
        )
    net = ValueNet(
        CUBE_3X3,
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
    """Stable seed for a (run, strategy, depth) cell."""
    return SEED_BASE + 100_003 * run_idx + 1_009 * strategy_idx + depth


def sample_states_random_walk(
    *,
    depth: int,
    n: int,
    seed: int,
) -> torch.Tensor:
    """``random_walk_depth``: scramble = random walk of length ``d``."""
    spec = CUBE_3X3
    gen = torch.Generator(device="cpu").manual_seed(seed)
    states, _ = random_scrambles(
        spec, batch_size=n, depth=depth, generator=gen, prune_same_face=True
    )
    return states


def sample_states_v_star_stratified(
    *,
    depth: int,
    n: int,
    seed: int,
    v_star_states: np.ndarray,
    v_star_depths: np.ndarray,
) -> torch.Tensor:
    """``v_star_stratified``: state ~ Uniform({s : V*(s) == depth}).

    Only valid for depth ≤ V_STAR_K (bounded oracle's K from P1b). Caller
    must skip cells with depth > V_STAR_K.
    """
    if depth > V_STAR_K:
        raise ValueError(
            f"v_star_stratified at depth {depth} > K={V_STAR_K} — "
            "bounded oracle has no entries"
        )
    # The actual sampler primitive lives alongside the bounded oracle
    # module (P1b). Until that lands, this branch is unreachable because
    # main() short-circuits when V_STAR_PATH is absent.
    from rubik.oracle.v_star_bounded_3x3 import sample_states_at_v_star

    rng = np.random.default_rng(seed)
    states_np = sample_states_at_v_star(
        v_star_states, v_star_depths, target_v_star=depth, n=n, rng=rng, rotate=False
    )
    return torch.from_numpy(states_np.astype(np.int64))


def solve_with_method(
    net: torch.nn.Module,
    *,
    method: str,
    beam_width: int,
    states: torch.Tensor,
    depth: int,
) -> list[int]:
    """Run the named search method on ``states`` and return per-attempt solve lens.

    Greedy uses ``max_steps = 2 * depth`` (matches eval.py budget). Beam
    uses ``max_steps = BEAM_MAX_STEPS`` (matches the M6 acceptance config
    inherited from 2x2; P3+ may earn its own value for 3x3).
    """
    spec = CUBE_3X3
    if method == "greedy":
        solve_lens = greedy_solve_batch(
            net, spec, states, max_steps=DEPTH_BUDGET_FACTOR * depth
        )
        return solve_lens.cpu().tolist()
    if method == "beam":
        result = beam_solve_batch(
            net, spec, states, beam_width=beam_width, max_steps=BEAM_MAX_STEPS
        )
        return result.solve_lens.cpu().tolist()
    raise ValueError(f"unknown method: {method}")


def _summarize(lens: list[int], n: int) -> str:
    n_solved = sum(1 for x in lens if x >= 0)
    if n_solved == 0:
        return f"solved {n_solved}/{n}  avg_len=N/A"
    avg = sum(x for x in lens if x >= 0) / n_solved
    return f"solved {n_solved}/{n} ({100 * n_solved / n:5.1f}%)  avg_len={avg:.2f}"


def _write_stub(out: dict) -> None:
    """Write a placeholder solve_histograms.json so the renderer has input.

    Used when there are no runs yet (P1a scaffold), or when the bounded
    V* table hasn't been built (pre-P1b), or when the architecture
    constants haven't been chosen (pre-P2b). The renderer then produces
    an empty-state HTML rather than crashing.
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote stub {OUT_PATH.relative_to(REPO_ROOT)}")


def main() -> None:
    out: dict[str, dict] = {
        "config": {
            "n_per_depth": N_PER_DEPTH,
            "depth_budget_factor": DEPTH_BUDGET_FACTOR,
            "depths": list(DEPTHS),
            "depths_v_star": list(DEPTHS_V_STAR),
            "strategies": list(STRATEGIES),
            "methods": [name for name, _ in METHODS],
            "beam_width": dict(METHODS).get("beam"),
            "beam_max_steps": BEAM_MAX_STEPS,
            "seed_base": SEED_BASE,
            "v_star_k": V_STAR_K,
        },
        "runs": {},
    }

    if not RUNS:
        print(
            "no runs registered in RUNS — writing stub solve_histograms.json. "
            "Append (label, run_subdir) tuples once cycles land (P2b onward)."
        )
        _write_stub(out)
        return

    if BODY_WIDTHS is None or N_RESIDUAL_BLOCKS is None or NORMALIZATION is None:
        print(
            "architecture constants not set (BODY_WIDTHS / N_RESIDUAL_BLOCKS / "
            "NORMALIZATION) — writing stub solve_histograms.json. Earn these "
            "in P2a + fill in P2b before re-running this capture."
        )
        _write_stub(out)
        return

    device = torch.device("mps")

    # Load V* table once (shared across all runs × depths for the
    # v_star_stratified strategy). When the table doesn't exist yet
    # (pre-P1b), skip v_star_stratified — random_walk_depth still runs.
    v_star_states: np.ndarray | None = None
    v_star_depths: np.ndarray | None = None
    if V_STAR_PATH.exists():
        from rubik.oracle.v_star_bounded_3x3 import load_v_star_arrays

        v_star_states, v_star_depths = load_v_star_arrays(V_STAR_PATH)
        print(f"loaded bounded V* table: {len(v_star_states)} states (K={V_STAR_K})")
    else:
        print(
            f"bounded V* table not found at {V_STAR_PATH} — "
            "skipping v_star_stratified strategy (pre-P1b)"
        )

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
            if strategy == "v_star_stratified" and v_star_states is None:
                print(f"  -- skip strategy: {strategy} (no oracle)")
                continue
            print(f"  -- strategy: {strategy}")
            depths_for_strategy = (
                DEPTHS_V_STAR if strategy == "v_star_stratified" else DEPTHS
            )
            per_method: dict[str, dict[str, list[int]]] = {
                name: {} for name, _ in METHODS
            }
            for d in depths_for_strategy:
                seed = _cell_seed(run_idx, strategy_idx, d)
                if strategy == "random_walk_depth":
                    states = sample_states_random_walk(
                        depth=d, n=N_PER_DEPTH, seed=seed
                    )
                elif strategy == "v_star_stratified":
                    states = sample_states_v_star_stratified(
                        depth=d,
                        n=N_PER_DEPTH,
                        seed=seed,
                        v_star_states=v_star_states,
                        v_star_depths=v_star_depths,
                    )
                else:
                    raise ValueError(f"unknown strategy: {strategy}")

                for method_name, beam_width in METHODS:
                    lens = solve_with_method(
                        net,
                        method=method_name,
                        beam_width=beam_width,
                        states=states,
                        depth=d,
                    )
                    print(
                        f"     d={d:>2} [{method_name:>6}]: "
                        f"{_summarize(lens, N_PER_DEPTH)}"
                    )
                    per_method[method_name][str(d)] = lens

            run_entry[strategy] = {
                method_name: {"lens": per_method[method_name]}
                for method_name, _ in METHODS
            }

        out["runs"][run_subdir] = run_entry

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
