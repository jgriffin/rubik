"""Stage the first N rows of the m11 parity corpus into corpus.json.

The parity corpus at ``tests/data/m11_parity_corpus_3x3.json`` was
seeded 0xBEEF, depth-14 scrambles, beam_width=128, max_steps=22. Block
D reuses the same facelets so the perf comparison is anchored on the
same states that validated TS / Python parity in Block B.

Idempotent: reads source, writes destination. Re-running with the same
N yields the same bytes.

Usage::

    uv run python experiments/browser-solve/stage_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SOURCE = REPO_ROOT / "tests" / "data" / "m11_parity_corpus_3x3.json"
DEST = EXPERIMENT_DIR / "corpus.json"

# How many rows to stage. M11 D harness uses N=10 (corpus rows) × widths
# {32, 64, 128, 256} × 3 solvers = 120 measurements. WASM at width=256
# is ~minutes per solve, so N=10 is the practical ceiling for the first
# perf cut.
N_ROWS = 10


def main() -> None:
    with open(SOURCE) as f:
        src = json.load(f)
    rows_in = src["rows"]
    if len(rows_in) < N_ROWS:
        raise SystemExit(
            f"parity corpus has {len(rows_in)} rows, need at least {N_ROWS}"
        )
    rows_out = [
        {
            "row_idx": i,
            "facelet": r["facelet"],
            "py_solve_len": int(r["solve_len"]),
        }
        for i, r in enumerate(rows_in[:N_ROWS])
    ]
    out = {
        "source": str(SOURCE.relative_to(REPO_ROOT)),
        "source_seed": src.get("seed"),
        "source_scramble_depth": src.get("scramble_depth"),
        "n_rows": N_ROWS,
        "rows": rows_out,
    }
    DEST.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {DEST.relative_to(REPO_ROOT)} — {N_ROWS} rows")


if __name__ == "__main__":
    main()
