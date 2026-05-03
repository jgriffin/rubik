"""Mini-bench for `apply_moves` at a few representative batch sizes.

Used during the env.py CPU-sync cleanup block to capture before/after
numbers per fix so we can attribute speedup to each change. Generic
enough to reuse for any future hot-path fix in `apply_moves`: pass
`--label baseline | fix1 | fix2 | post-flag-X` and (optionally) `--out`
a JSON file under `runs/<ts>/<label>.json` for later comparison.

**Realistic call pattern.** `move_idxs` is built on CPU (matches
`random_scrambles` inner loop and the future beam-search children
expansion — both build move tags as CPU tensors and let `apply_moves`
migrate). This differs from `experiments/batch-sensitivity-2x2/run.py`,
which pre-migrates to MPS as a bench convenience and therefore measures
a different regime for the bounds-check syncs. For this cleanup loop,
the CPU-input pattern is the production reality.

Output: a clean per-batch table on stdout + optional JSON.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from rubik.cube.env import apply_moves
from rubik.cube.spec import CUBE_2X2
from rubik.perf import bootstrap_ci, time_op

DEFAULT_BATCH_SIZES = (1, 64, 8192, 2_097_152)
DEFAULT_TRIALS = 100
DEFAULT_WARMUP = 10

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_ROOT = REPO_ROOT / "experiments" / "mps-methodology" / "runs"


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def bench_one(batch: int, trials: int, warmup: int) -> dict:
    spec = CUBE_2X2
    device = torch.device("mps")

    states = (
        spec.solved_state.to(device=device, dtype=torch.int8)
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    # CPU move_idxs — production-realistic; this is what the bounds-check
    # relocation fix benefits.
    g = torch.Generator(device="cpu").manual_seed(0)
    move_idxs = torch.randint(0, spec.n_moves, (batch,), generator=g)

    def call() -> None:
        apply_moves(states, move_idxs, spec)

    timings = time_op(call, warmup=warmup, trials=trials, device="mps")
    median, lo, hi = bootstrap_ci(timings, seed=0)
    throughput = batch / median  # states per second
    return {
        "batch": batch,
        "median_s": median,
        "ci_lo_s": lo,
        "ci_hi_s": hi,
        "throughput_states_per_s": throughput,
        "n_trials": trials,
        "n_warmup": warmup,
    }


def fmt_us(seconds: float) -> str:
    return f"{seconds * 1e6:,.2f}"


def fmt_throughput(states_per_s: float) -> str:
    if states_per_s >= 1e9:
        return f"{states_per_s / 1e9:.2f} G"
    if states_per_s >= 1e6:
        return f"{states_per_s / 1e6:.2f} M"
    if states_per_s >= 1e3:
        return f"{states_per_s / 1e3:.2f} K"
    return f"{states_per_s:.2f}"


def print_table(label: str, rows: list[dict]) -> None:
    print(f"# {label} — apply_moves bench (2x2, MPS, CPU move_idxs)")
    print(
        f"{'B':>10}  {'median (μs)':>14}  {'95% CI (μs)':>26}  "
        f"{'throughput (st/s)':>20}"
    )
    print("-" * 76)
    for r in rows:
        ci = f"[{fmt_us(r['ci_lo_s'])}, {fmt_us(r['ci_hi_s'])}]"
        thr = fmt_throughput(r["throughput_states_per_s"])
        print(f"{r['batch']:>10}  {fmt_us(r['median_s']):>14}  {ci:>26}  {thr:>20}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to bench",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Tag for this run (e.g., 'baseline', 'fix1-bounds', 'fix2-perm-cache')",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Optional dir for JSON output. If set, writes to "
            "<out-dir>/<label>.json. If omitted, writes to "
            "experiments/mps-methodology/runs/<ts>/<label>.json."
        ),
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("ERROR: torch.backends.mps.is_available() is False", file=sys.stderr)
        return 1

    rows = [bench_one(b, args.trials, args.warmup) for b in args.batches]
    print_table(args.label, rows)

    if args.out_dir is None:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        out_dir = RUNS_ROOT / ts
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.json"

    payload = {
        "label": args.label,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "n_trials": args.trials,
        "n_warmup": args.warmup,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    abs_out = out_path.resolve()
    try:
        rel = abs_out.relative_to(REPO_ROOT)
        print(f"\nwrote: {rel}")
    except ValueError:
        print(f"\nwrote: {abs_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
