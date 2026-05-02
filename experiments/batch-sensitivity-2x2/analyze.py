"""M4 — Render the batch-sensitivity sweep into results.md.

Reads the newest runs/<ts>/data.json (or --run path), produces a
markdown throughput table per op, and embeds intuition observations
(dispatch-overhead region, saturation point, regime notes) plus an
optional steady-state-vs-bench correction factor.

Idempotent: rerunning regenerates results.md from the chosen run.

Streaming-output policy: this script does not stream — it writes one
file (results.md) and prints the path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch-sensitivity-2x2"
RUNS_ROOT = EXPERIMENT_DIR / "runs"
RESULTS_PATH = EXPERIMENT_DIR / "results.md"

# Scaling efficiency = (throughput_ratio / batch_ratio). Linear scaling
# = 1.0; full saturation = 0.0. We use efficiency rather than raw gain
# because the sweep grid uses non-uniform (8x) batch steps.
SATURATION_ONSET_EFFICIENCY = 0.80  # first transition below this = onset
HEAVY_SATURATION_EFFICIENCY = 0.40  # below this = heavily saturated

# Dispatch-bound heuristic: across the smallest few batch sizes, if per-call
# seconds vary by less than this relative spread, dispatch dominates work.
DISPATCH_FLAT_THRESHOLD = 0.20


def find_latest_run() -> Path | None:
    if not RUNS_ROOT.exists():
        return None
    candidates = sorted(RUNS_ROOT.glob("*/data.json"))
    if not candidates:
        return None
    return candidates[-1]


def fmt_seconds_ms(s: float) -> str:
    """Format seconds in milliseconds with 3 sig figs."""
    return f"{s * 1000:.3f}"


def fmt_throughput(t: float) -> str:
    """Format throughput per second with engineering suffix."""
    if t >= 1e9:
        return f"{t / 1e9:.3f} G"
    if t >= 1e6:
        return f"{t / 1e6:.3f} M"
    if t >= 1e3:
        return f"{t / 1e3:.3f} k"
    return f"{t:.3f} "


def render_op_table(op: str, cells: list[dict[str, Any]]) -> str:
    """Render a markdown table for one op."""
    rows = [c for c in cells if c["op"] == op]
    rows.sort(key=lambda c: c["batch_size"])

    if op == "apply_moves":
        unit = "states/s"
    elif op == "random_scrambles":
        unit = "scrambles/s"
    else:
        unit = "items/s"

    header = (
        "| batch | median (ms) | CI lo (ms) | CI hi (ms) "
        "| throughput (median) | throughput CI lo | throughput CI hi |"
    )
    lines = [
        f"### `{op}` ({unit})",
        "",
        header,
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {b} | {m} | {lo} | {hi} | {tm}{u} | {tl}{u} | {th}{u} |".format(
                b=r["batch_size"],
                m=fmt_seconds_ms(r["median_seconds"]),
                lo=fmt_seconds_ms(r["ci_lo_seconds"]),
                hi=fmt_seconds_ms(r["ci_hi_seconds"]),
                tm=fmt_throughput(r["throughput_per_sec_median"]),
                tl=fmt_throughput(r["throughput_per_sec_lo"]),
                th=fmt_throughput(r["throughput_per_sec_hi"]),
                u=unit,
            )
        )
    lines.append("")
    return "\n".join(lines)


def compute_observations(cells: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Per-op observation strings: dispatch region, saturation, anomalies."""
    obs: dict[str, list[str]] = {}
    ops = sorted({c["op"] for c in cells})
    for op in ops:
        rows = sorted(
            [c for c in cells if c["op"] == op], key=lambda c: c["batch_size"]
        )
        notes: list[str] = []
        if len(rows) < 2:
            notes.append(
                "Single batch size in this run — saturation/dispatch curve "
                "needs the full sweep (commit 7) to read."
            )
            obs[op] = notes
            continue

        # Dispatch-bound region: contiguous prefix where per-call seconds are
        # within DISPATCH_FLAT_THRESHOLD relative spread.
        small_rows = rows[: min(4, len(rows))]
        small_times = [r["median_seconds"] for r in small_rows]
        if len(small_times) >= 2:
            spread = (max(small_times) - min(small_times)) / max(small_times)
            if spread < DISPATCH_FLAT_THRESHOLD:
                notes.append(
                    f"Dispatch-bound region: B={small_rows[0]['batch_size']}.."
                    f"{small_rows[-1]['batch_size']} per-call seconds vary by "
                    f"only {spread * 100:.1f}% — call overhead dominates work."
                )

        # Scaling efficiency per transition = throughput_ratio / batch_ratio.
        # 1.0 = perfect linear scaling. 0.0 = fully saturated. We surface the
        # full curve and call out the saturation-onset cell.
        eff_curve: list[tuple[int, int, float]] = []  # (prev_B, cur_B, eff)
        onset_b: int | None = None
        heavy_b: int | None = None
        for prev, cur in zip(rows, rows[1:], strict=False):
            prev_t = prev["throughput_per_sec_median"]
            cur_t = cur["throughput_per_sec_median"]
            if prev_t <= 0:
                continue
            tput_ratio = cur_t / prev_t
            batch_ratio = cur["batch_size"] / prev["batch_size"]
            eff = tput_ratio / batch_ratio if batch_ratio > 0 else 0.0
            eff_curve.append((prev["batch_size"], cur["batch_size"], eff))
            if onset_b is None and eff < SATURATION_ONSET_EFFICIENCY:
                onset_b = prev["batch_size"]
            if heavy_b is None and eff < HEAVY_SATURATION_EFFICIENCY:
                heavy_b = prev["batch_size"]

        if eff_curve:
            curve_str = ", ".join(f"{p}→{c}: {e:.2f}" for p, c, e in eff_curve)
            notes.append(f"Scaling efficiency curve (per 8x batch step): {curve_str}.")

        if onset_b is not None:
            notes.append(
                f"Saturation onset: scaling efficiency drops below "
                f"{SATURATION_ONSET_EFFICIENCY:.0%} starting at B={onset_b} "
                f"(throughput grows but no longer linearly with batch)."
            )
        else:
            notes.append(
                "Linear-scaling regime across the full sweep — top batch may "
                "still have headroom."
            )

        if heavy_b is not None and heavy_b != onset_b:
            notes.append(
                f"Heavy saturation: efficiency below "
                f"{HEAVY_SATURATION_EFFICIENCY:.0%} starting at B={heavy_b} — "
                f"throughput is approaching a hard ceiling."
            )

        # Peak observed throughput.
        peak = max(rows, key=lambda r: r["throughput_per_sec_median"])
        peak_t = peak["throughput_per_sec_median"]
        peak_unit = "states/s" if op == "apply_moves" else "scrambles/s"
        notes.append(
            f"Peak observed throughput: {fmt_throughput(peak_t)}{peak_unit} "
            f"at B={peak['batch_size']}."
        )
        obs[op] = notes
    return obs


def render_observations(obs: dict[str, list[str]]) -> str:
    lines = []
    for op in sorted(obs):
        lines.append(f"### `{op}`")
        lines.append("")
        for note in obs[op]:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def render_steady_state_section(steady: dict[str, Any]) -> str:
    factor = steady.get("correction_factor")
    bench_per_call = steady.get("bench_bracket_per_call_s")
    steady_per_call = steady.get("steady_state_per_call_s")
    batch_size = steady.get("batch_size")
    n_calls = steady.get("n_calls")
    duration = steady.get("duration_s")

    return (
        "The sweep tables above report **bench-bracket** per-call seconds: "
        "`mps.synchronize()` before and after each timed call. That bracket "
        "pattern is the only reliable wall-time on MPS, but it bakes in pure "
        "sync stall on each side that pipelined workloads (training, beam "
        "search) don't pay.\n\n"
        f"At B={batch_size}, a steady-state pipelined loop "
        f"(sync every 64 calls, {duration:.1f}s wall-time, {n_calls} calls) "
        f"measured {steady_per_call * 1000:.3f} ms per call vs the "
        f"bench-bracket's {bench_per_call * 1000:.3f} ms — a correction "
        f"factor of **{factor:.3f}**. That is, real pipelined throughput is "
        f"~{1.0 / factor:.2f}x the bench-bracket throughput at this batch size.\n\n"
        "**Use the sweep tables for cross-cell comparisons** (the sync stall "
        "is a fixed per-call cost, so ratios and curve shapes are honest). "
        "**Apply the correction factor** when projecting absolute throughput "
        "for pipelined workloads. The macmon correlation in "
        "`experiments/mps-methodology/results.md` §5 is the calibration "
        "ground truth for this regime difference.\n"
    )


def render_results_md(
    data: dict[str, Any],
    data_path: Path,
    steady: dict[str, Any] | None,
) -> str:
    cells = data["cells"]
    machine = data.get("machine", {})
    git_sha = data.get("git_sha", "unknown")
    ran_at = data.get("ran_at", "unknown")
    config = data.get("config", {})

    sections: list[str] = []

    # 1. What this dir is.
    sections.append("# Batch-size sensitivity sweep (2x2)\n")
    sections.append(
        "## What this dir is\n\n"
        "Sweeps `apply_moves` and `random_scrambles` throughput across batch "
        "sizes on this M4 Max to find the dispatch-overhead floor, the "
        "saturation ceiling, and any allocator/memory cliffs. The curve "
        "shapes are inputs to M7 (beam search) and M8 (DAVI training) "
        "hyperparameter choices, and a deliberate exercise in building "
        "GPU-saturation intuition for this hardware.\n\n"
        "Sweep grid lives in `config.yaml`. Driver: `run.py`. Steady-state "
        "spot-check: `run_steady_state.py`. Renderer: `analyze.py`.\n"
    )

    # 2. Latest run.
    sections.append("## Latest run\n")
    sections.append(
        f"- **Timestamp:** `{ran_at}`\n"
        f"- **Machine:** `{machine.get('platform', 'unknown')}`\n"
        f"- **Torch:** `{machine.get('torch_version', 'unknown')}`\n"
        f"- **Git SHA:** `{git_sha}`\n"
        f"- **Data:** `{data_path.relative_to(REPO_ROOT)}` (gitignored)\n"
        f"- **Sweep ops:** `{config.get('ops', [])}`\n"
        f"- **Batch sizes:** `{config.get('batch_sizes', [])}`\n"
        f"- **Trials per cell:** {config.get('trials', '?')}; "
        f"warmup: {config.get('warmup', '?')}\n"
    )

    # 3. Methodology pointers.
    sections.append("## Methodology pointers\n")
    sections.append(
        "- `experiments/mps-methodology/results.md` — full triangulation "
        "strategy (bench-bracket → macmon → profiler), gotchas list, "
        "tools matrix. Read first if these numbers look surprising.\n"
        "- `src/rubik/perf/bench.py` — `time_op` (warmup → sync → "
        "perf_counter → fn → sync → perf_counter) and `bootstrap_ci` "
        "(median + 95% CI via numpy resampling).\n"
        "- `experiments/batch-sensitivity-2x2/config.yaml` — sweep grid.\n"
    )

    # 4. Throughput tables.
    sections.append("## Throughput tables\n")
    ops = sorted({c["op"] for c in cells})
    for op in ops:
        sections.append(render_op_table(op, cells))

    # 5. Observations.
    obs = compute_observations(cells)
    sections.append("## Observations\n")
    sections.append(
        "Reading these as GPU-saturation intuition for this M4 Max: "
        "dispatch overhead dominates at the smallest batches (per-call time "
        "is flat regardless of how many states we hand the GPU); the curve "
        "lifts when batch is large enough to amortize kernel launch; it "
        "plateaus when memory bandwidth or compute throughput is the wall.\n"
    )
    sections.append(render_observations(obs))

    # 6. Bench-bracket vs steady-state regime.
    sections.append("## Bench-bracket vs steady-state regime\n")
    if steady is not None:
        sections.append(render_steady_state_section(steady))
    else:
        sections.append(
            "Calibration pending — run `run_steady_state.py` and rerun "
            "`analyze.py --steady-state-run <path>` to embed the "
            "bench-vs-pipelined correction factor here. Until then: the "
            "sweep numbers are sync-bracket per-call seconds, which "
            "underestimate steady-state pipelined throughput by a "
            "regime-dependent factor (see "
            "`experiments/mps-methodology/results.md` §5).\n"
        )

    # 7. Intuition (hand-written, preserved across regenerations).
    intuition_path = EXPERIMENT_DIR / "intuition.md"
    if intuition_path.exists():
        sections.append(intuition_path.read_text().rstrip() + "\n")

    return "\n".join(sections)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Path to a runs/<ts>/data.json (default: newest in runs/).",
    )
    ap.add_argument(
        "--steady-state-run",
        type=Path,
        default=None,
        help="Path to a runs/<ts>/steady_state.json from run_steady_state.py.",
    )
    args = ap.parse_args()

    data_path = args.run if args.run is not None else find_latest_run()
    if data_path is None or not data_path.exists():
        print(
            "ERROR: no data.json found (looked under "
            f"{RUNS_ROOT.relative_to(REPO_ROOT)}). Run run.py first.",
            file=sys.stderr,
        )
        return 1

    with data_path.open("r") as f:
        data = json.load(f)

    steady: dict[str, Any] | None = None
    if args.steady_state_run is not None:
        if not args.steady_state_run.exists():
            print(
                f"ERROR: steady-state run not found at {args.steady_state_run}",
                file=sys.stderr,
            )
            return 1
        with args.steady_state_run.open("r") as f:
            steady = json.load(f)

    rendered = render_results_md(data, data_path, steady)
    RESULTS_PATH.write_text(rendered)
    print(f"results: {RESULTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
