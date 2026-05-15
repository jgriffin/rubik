"""M11 B·P3: Python-side beam reference for the Node parity gate.

Generates a seeded 50-scramble depth-14 corpus and runs the Python beam
search (CPU) to produce per-row solve verdicts. The Node-side runner
(``web/scripts/parityNode.ts``) reads the resulting JSON and compares
its own TS-beam aggregate metrics against this reference.

We run on CPU (not MPS) so the precision regime matches onnxruntime's
CPU EP that Node will use — same FP32, same op-order ballpark. The
remaining drift between CPU PyTorch and onnxruntime CPU EP is exercised
by tests/onnx_parity_test_3x3.py (Block A), well under the parity gate's
aggregate tolerance.

Usage:
    uv run python scripts/parity_python_reference_3x3.py
    uv run python scripts/parity_python_reference_3x3.py --beam-width 256 --max-steps 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_3X3
from rubik.notation.facelet import state_to_facelet
from rubik.search.beam import beam_solve_batch
from rubik.server.inference import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "data" / "m11_parity_corpus_3x3.json"
DEFAULT_SEED = 0xBEEF


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=DEFAULT_SEED)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--scramble-depth", type=int, default=14)
    ap.add_argument("--beam-width", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=22)
    args = ap.parse_args()

    spec = CUBE_3X3
    device = torch.device("cpu")

    print(f"loading: {args.checkpoint}")
    loaded = load_model(args.checkpoint, device=device)
    net = loaded.net

    print(
        f"generating: {args.batch_size} scrambles, depth={args.scramble_depth},"
        f" seed=0x{args.seed:X}"
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    states, _move_seqs = random_scrambles(
        spec, batch_size=args.batch_size, depth=args.scramble_depth, generator=gen
    )

    print(
        f"solving: beam_width={args.beam_width}, max_steps={args.max_steps}"
        f" on CPU (single batched beam_solve_batch call)"
    )
    t0 = time.perf_counter()
    result = beam_solve_batch(
        net,
        spec,
        states,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
    )
    py_runtime_ms = int((time.perf_counter() - t0) * 1000)

    rows = []
    solve_lens = result.solve_lens.tolist()
    for i in range(args.batch_size):
        facelet = state_to_facelet(states[i], spec)
        solve_len = int(solve_lens[i])
        solved = solve_len >= 0
        moves = result.solve_paths[i] if solved else []
        rows.append(
            {
                "facelet": facelet,
                "solve_len": solve_len,
                "solved": solved,
                "solve_moves": [int(m) for m in moves],
            }
        )

    n_solved = sum(1 for r in rows if r["solved"])
    solved_lens = [r["solve_len"] for r in rows if r["solved"]]
    mean_len = sum(solved_lens) / len(solved_lens) if solved_lens else 0.0
    solve_rate = n_solved / args.batch_size if args.batch_size else 0.0

    payload = {
        "seed": f"0x{args.seed:X}",
        "scramble_depth": args.scramble_depth,
        "beam_width": args.beam_width,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "py_runtime_ms": py_runtime_ms,
        "py_solve_rate": solve_rate,
        "py_mean_solve_len": mean_len,
        "py_n_solved": n_solved,
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote: {args.output} ({args.batch_size} rows, "
        f"py solve_rate={n_solved}/{args.batch_size} ({100 * solve_rate:.1f}%), "
        f"mean_solve_len={mean_len:.1f} (over solved), "
        f"wall={py_runtime_ms / 1000:.1f}s)"
    )


if __name__ == "__main__":
    main()
