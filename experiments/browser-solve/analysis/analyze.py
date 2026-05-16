"""Analyze browser-solve latencies and emit results.md.

Reads ``results/latencies.jsonl`` (one row per measurement, see
``measure_fastapi.py`` for the row shape — Playwright spec emits the
same shape). Groups by ``(solver, width)`` and writes a numbers-first
markdown table to ``results/results.md``.

Per project convention, ``intuition.md`` (hand-written) is appended at
the end so re-running the analyzer doesn't clobber the hypotheses.

RUNS table at the top — a new cycle is "append a tuple, regenerate."
For this first measurement run there's only one (label, file) tuple.

Usage::

    uv run python experiments/browser-solve/analysis/analyze.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
INTUITION_PATH = EXPERIMENT_DIR / "intuition.md"
RESULTS_MD = RESULTS_DIR / "results.md"

# (label, jsonl-path-relative-to-experiment-dir). Each cycle appends.
RUNS: list[tuple[str, str]] = [
    ("first cut (N=10 × widths × 3 solvers)", "results/latencies.jsonl"),
]

SOLVER_ORDER = ["fastapi", "onnx-webgpu", "onnx-wasm"]
WIDTH_ORDER = [32, 64, 128, 256]


def _q(values: list[float], p: float) -> float:
    """Linear-interpolation quantile (matches numpy.quantile default)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    idx = p * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Group by (solver, width) and compute summary stats."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["solver"], int(r["width"]))].append(r)
    out: dict[tuple[str, int], dict] = {}
    for key, group in grouped.items():
        walls = [float(r["wall_ms"]) for r in group if r["wall_ms"] >= 0]
        solved = [r for r in group if r.get("solved")]
        solve_lens = [int(r["solve_len"]) for r in solved if r["solve_len"] >= 0]
        out[key] = {
            "n": len(group),
            "median_ms": statistics.median(walls) if walls else float("nan"),
            "p10_ms": _q(walls, 0.10) if walls else float("nan"),
            "p90_ms": _q(walls, 0.90) if walls else float("nan"),
            "solve_rate": len(solved) / len(group) if group else 0.0,
            "mean_solve_len_solved": (
                statistics.mean(solve_lens) if solve_lens else float("nan")
            ),
        }
    return out


def fmt_ms(v: float) -> str:
    if v != v:  # NaN
        return "—"
    if v >= 1000:
        return f"{v / 1000:.2f}s"
    return f"{v:.0f}"


def render(summary: dict[tuple[str, int], dict]) -> str:
    lines: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# Browser-solve perf — M11 Block D")
    lines.append("")
    lines.append(f"_Generated: {now}_")
    lines.append("")
    lines.append("## Run conditions")
    lines.append("")
    lines.append("- Hardware: Apple M4 Max (user's dev machine).")
    lines.append("- Model: `experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt` (champion 3x3 ValueNet, ~234M params).")
    lines.append("- ONNX export: `web/public/models/net_final.onnx` (M11 Block A export, parity-verified).")
    lines.append("- Corpus: first 10 rows of `tests/data/m11_parity_corpus_3x3.json` (depth-14 scrambles, seed 0xBEEF). Same facelets used in Block B parity gate.")
    lines.append("- Beam: `max_steps=22`; widths swept ∈ {32, 64, 128, 256}.")
    lines.append("- FastAPI path: direct Python (`solve_facelet`) — no HTTP round-trip. MPS warmed at every width before measurement.")
    lines.append("- Browser path: Playwright-driven Chromium, ONNX Runtime Web (WASM + WebGPU EPs). One page-load per (ep, width) so model-load + EP-warmup amortize across rows.")
    lines.append("")
    lines.append("## Per-(solver, width) summary")
    lines.append("")
    lines.append("| solver | width | n | median ms | p10 ms | p90 ms | solve_rate | mean solve_len |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for solver in SOLVER_ORDER:
        for width in WIDTH_ORDER:
            s = summary.get((solver, width))
            if s is None:
                lines.append(f"| {solver} | {width} | 0 | — | — | — | — | — |")
                continue
            slen = (
                f"{s['mean_solve_len_solved']:.1f}"
                if s["mean_solve_len_solved"] == s["mean_solve_len_solved"]
                else "—"
            )
            lines.append(
                f"| {solver} | {width} | {s['n']} | "
                f"{fmt_ms(s['median_ms'])} | {fmt_ms(s['p10_ms'])} | "
                f"{fmt_ms(s['p90_ms'])} | {s['solve_rate']:.0%} | {slen} |"
            )
    lines.append("")
    lines.append("## Speedups (median latency at each width)")
    lines.append("")
    lines.append("| width | FastAPI ms | WebGPU ms | WASM ms | WebGPU/FastAPI | WASM/FastAPI | WASM/WebGPU |")
    lines.append("|---|---|---|---|---|---|---|")
    for width in WIDTH_ORDER:
        f_ms = summary.get(("fastapi", width), {}).get("median_ms", float("nan"))
        g_ms = summary.get(("onnx-webgpu", width), {}).get("median_ms", float("nan"))
        w_ms = summary.get(("onnx-wasm", width), {}).get("median_ms", float("nan"))

        def ratio(a: float, b: float) -> str:
            if a != a or b != b or b == 0:
                return "—"
            return f"{a / b:.2f}x"

        lines.append(
            f"| {width} | {fmt_ms(f_ms)} | {fmt_ms(g_ms)} | {fmt_ms(w_ms)} | "
            f"{ratio(g_ms, f_ms)} | {ratio(w_ms, f_ms)} | {ratio(w_ms, g_ms)} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for _label, rel in RUNS:
        rows.extend(load_rows(EXPERIMENT_DIR / rel))
    summary = summarize(rows)
    md = render(summary)
    if INTUITION_PATH.exists():
        intuition = INTUITION_PATH.read_text().rstrip()
        md += "\n---\n\n" + intuition + "\n"
    RESULTS_MD.write_text(md)
    print(f"wrote {RESULTS_MD.relative_to(REPO_ROOT)} ({len(rows)} rows analyzed)")


if __name__ == "__main__":
    main()
