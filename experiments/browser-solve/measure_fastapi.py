"""Measure FastAPI/MPS solve latency over (width, row) — M11 Block D.

Loads the champion 3x3 ValueNet ONCE (the model is ~234M params and the
MPS warmup is several seconds). Iterates over widths × rows from
``corpus.json``, calling :func:`rubik.server.inference.solve_facelet`
directly (no HTTP round-trip — we want the model wall, not the network
wall). Appends a JSONL row per measurement to
``results/latencies.jsonl``.

Row shape (matches Playwright spec for cross-solver comparison)::

    {
      "solver": "fastapi",
      "width": 128,
      "row_idx": 0,
      "facelet": "UU...",
      "wall_ms": 1234,
      "solve_len": 14,
      "solved": true
    }

Usage::

    uv run python experiments/browser-solve/measure_fastapi.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rubik.server.inference import load_model, solve_facelet

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
CORPUS = EXPERIMENT_DIR / "corpus.json"
OUT = EXPERIMENT_DIR / "results" / "latencies.jsonl"
MODEL = (
    REPO_ROOT
    / "experiments"
    / "davi-3x3"
    / "runs"
    / "20260508T084940Z_ln_kmax30_100k"
    / "net_final.pt"
)

WIDTHS = [32, 64, 128, 256]
MAX_STEPS = 22


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS) as f:
        corpus = json.load(f)
    rows = corpus["rows"]

    print(f"loading model from {MODEL.relative_to(REPO_ROOT)}…")
    t0 = time.perf_counter()
    loaded = load_model(MODEL)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s on {next(loaded.net.parameters()).device}")

    # Warmup: one solve at each width so the first row of each width
    # doesn't carry the MPS kernel-compile spike. Throw the timings away.
    print("warming up MPS kernels at each width…")
    warmup_facelet = rows[0]["facelet"]
    for w in WIDTHS:
        _ = solve_facelet(loaded, warmup_facelet, beam_width=w, max_steps=MAX_STEPS)

    n_measurements = 0
    total_wall = 0.0
    with open(OUT, "a") as f:
        for width in WIDTHS:
            for row in rows:
                facelet = row["facelet"]
                row_idx = row["row_idx"]
                t_solve = time.perf_counter()
                r = solve_facelet(
                    loaded, facelet, beam_width=width, max_steps=MAX_STEPS
                )
                dt = time.perf_counter() - t_solve
                wall_ms = int(r["stats"]["time_ms"])  # inference wall, not Python overhead
                rec = {
                    "solver": "fastapi",
                    "width": width,
                    "row_idx": row_idx,
                    "facelet": facelet,
                    "wall_ms": wall_ms,
                    "solve_len": len(r["moves"]) if r["solved"] else -1,
                    "solved": bool(r["solved"]),
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_measurements += 1
                total_wall += dt
                print(
                    f"  width={width:4d} row={row_idx:2d} wall={wall_ms:5d}ms"
                    f" solved={rec['solved']} len={rec['solve_len']}"
                )
    print(f"fastapi: {n_measurements} measurements, total wall={total_wall:.1f}s")


if __name__ == "__main__":
    main()
