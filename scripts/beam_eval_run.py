"""Run-dir wrapper for the beam-eval CLI.

Walks a training-run directory and invokes the same eval logic as
``beam_eval_model.py`` for each ``net_step_*.pt`` and ``net_final.pt``
checkpoint. All per-checkpoint payloads land **appended** as JSONL lines
into a single rollup file at
``<run-dir>/results/beam_eval_<config-name>.jsonl`` (one JSON object per
line, sorted ascending by step). Re-running an eval merges into the same
file, deduplicating by step (the new payload replaces the prior entry
for that step) — provided the existing records' eval-config signature
matches the current invocation.

Common workflow::

    uv run python scripts/beam_eval_run.py <run-dir> --config fast

Idempotent by default: if a record at step ``S`` already exists in the
target JSONL, that checkpoint is **skipped**. Pass ``--force`` to opt
back into "recompute everything" (the prior default — useful after
changing eval-config schema or debugging the eval itself). Different
``--config`` values map to different JSONL files (the config name is
in the filename), so fast and fast_kmax30 records never collide.

**Schema check.** Before merging, the runner extracts a config signature
``(max_depth, n_per_depth, beam_width, precision, seed)`` from each
existing record and compares against the current invocation. If any
existing record's signature differs (e.g. the YAML schedule was edited
mid-flight), the runner errors out with the offending record's signature
and aborts before any write. The user resolves either by picking a
different ``--config`` name (records land in a new JSONL) or by passing
``--force`` to back up and restart fresh.

**``--force`` behavior.** ``--force`` *always* backs up an existing
JSONL — whether or not its schema matches — to a sibling file named
``beam_eval_<config>.jsonl.bak.<UTC-timestamp>`` (timestamp format
``YYYYMMDDTHHMMSSZ``). After backup the primary file starts empty,
so every targeted checkpoint is re-evaluated rather than merged with
prior records. If no existing JSONL is present, ``--force`` is a no-op
for the backup step and proceeds with an empty merge state.

Filtering (mutually exclusive):

- ``--stride N`` — only checkpoints whose step is a multiple of N
  (plus net_final.pt always).
- ``--steps "10000,20000,final"`` — explicit step labels; literal
  ``"final"`` resolves to net_final.pt's step (read from metrics.jsonl
  or the training config's n_steps).

If neither is set, every checkpoint is fair game. Combined with
ensure-by-default, the natural workflow is coarse-to-fine: a first
pass with ``--stride 10000`` lays down the headline curve, a later
``--stride 5000`` pass fills in intermediate cells without recomputing
the existing 10k-multiple cells.

**Traversal order.** Checkpoints are processed **largest step first**
(net_final.pt first, then descending net_step_*.pt). If the user kills
mid-run, the most-trained — and usually most-interesting — checkpoints
are already on disk.

This script is the post-refactor sibling of ``beam_eval_model.py`` —
the same shared core (``evaluate_checkpoint``) runs for every
checkpoint; this layer just walks the directory and resolves filtering.

**Empty-checkpoint behavior.** If the run dir contains no checkpoints
matching ``net_step_*.pt`` or ``net_final.pt``, this script writes an
empty JSON object (``{}``) to ``<run-dir>/results/post_run_beam_eval.json``
and exits 0. Tests rely on this contract — a freshly-created run dir
is a normal pre-training state, not an error. (Empty-state file path
deliberately keeps the old name so test fixtures stay valid; the
post-rename per-checkpoint path is only used when checkpoints exist.)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "davi-3x3"
ORACLE_CACHE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"

# Make ``experiments/davi-3x3/eval.py`` and the scripts/ dir importable.
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_config import EvalConfig  # noqa: E402
from beam_eval_model import evaluate_checkpoint  # noqa: E402

from rubik.oracle.v_star_bounded_3x3 import (  # noqa: E402
    load_v_star_bounded_3x3_arrays,
)
from rubik.training.config import DAVIConfig  # noqa: E402

# Match ``net_step_<int>.pt``. ``net_final.pt`` is handled specially.
_STEP_RE = re.compile(r"^net_step_(\d+)\.pt$")

# Fields that define the eval-config "shape" of a JSONL record. If any of
# these differ between an existing record and the current invocation, the
# two payloads are not safely mergeable in the same file. ``config_name``
# is intentionally NOT here — it's a label, not a schema.
_SIGNATURE_FIELDS: tuple[str, ...] = (
    "max_depth",
    "n_per_depth",
    "beam_width",
    "precision",
    "seed",
)


def _signature_from_record(rec: dict) -> tuple:
    """Extract the (max_depth, n_per_depth, beam_width, precision, seed) tuple.

    ``n_per_depth`` is normalised to a tuple-of-ints so list-vs-tuple JSON
    round-tripping doesn't show up as a "mismatch".
    """
    return (
        int(rec["max_depth"]) if "max_depth" in rec else None,
        tuple(int(v) for v in rec["n_per_depth"]) if "n_per_depth" in rec else None,
        int(rec["beam_width"]) if "beam_width" in rec else None,
        str(rec["precision"]) if "precision" in rec else None,
        int(rec["seed"]) if "seed" in rec else None,
    )


def _signature_from_eval_cfg(eval_cfg) -> tuple:
    """Mirror of ``_signature_from_record`` for the live ``EvalConfig``."""
    return (
        int(eval_cfg.max_depth),
        tuple(int(v) for v in eval_cfg.n_per_depth),
        int(eval_cfg.beam_width),
        str(eval_cfg.precision),
        int(eval_cfg.seed),
    )


def _format_signature(sig: tuple) -> str:
    """Render a signature tuple as a human-readable one-liner."""
    max_depth, npd, bw, prec, seed = sig
    npd_str = str(tuple(npd)) if npd is not None else "None"
    return (
        f"max_depth={max_depth}, n_per_depth={npd_str}, "
        f"beam_width={bw}, precision={prec}, seed={seed}"
    )


def _backup_jsonl(jsonl_path: Path) -> Path:
    """Move ``jsonl_path`` to a ``.bak.<UTC-timestamp>`` sibling and return that path.

    Timestamp format ``YYYYMMDDTHHMMSSZ`` matches the run-dir convention.
    Caller is responsible for checking that ``jsonl_path`` exists first.
    """
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = jsonl_path.with_name(jsonl_path.name + f".bak.{ts}")
    shutil.move(str(jsonl_path), str(backup))
    return backup


def _final_step_from_metrics(run_dir: Path, fallback: int) -> int:
    """Return the highest step recorded in metrics.jsonl, or ``fallback``.

    With early-stop, the run can terminate before ``config.n_steps``; in
    that case ``net_final.pt`` corresponds to the early-stop step, not
    ``n_steps``. Reading metrics.jsonl gives the truthful step label.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return int(fallback)
    max_step = 0
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = rec.get("step")
        if isinstance(s, int) and s > max_step:
            max_step = s
    return max_step or int(fallback)


def _training_max_depth_from_run_dir(
    run_dir: Path, fallback_from_config: int
) -> tuple[int, str]:
    """Resolve the training-time ``max_scramble_depth`` for the run.

    Prefers the first ``run_start`` event in ``metrics.jsonl`` (the
    authoritative training-time provenance), falls back to the value
    parsed from the run's ``config.yaml`` (passed in as
    ``fallback_from_config``).

    Returns ``(max_depth, source)`` where ``source`` is one of
    ``"metrics.jsonl"`` or ``"config.yaml"`` for log-line attribution.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "run_start":
                v = rec.get("max_scramble_depth")
                if isinstance(v, int):
                    return int(v), "metrics.jsonl"
                # run_start exists but lacks the field → fall back.
                break
    return int(fallback_from_config), "config.yaml"


def _list_checkpoints(run_dir: Path, n_steps: int) -> list[tuple[int, Path]]:
    """Return ``[(step, path), ...]`` sorted **descending** by step.

    Largest-first traversal means partial kills leave the most-trained
    (and usually most-interesting) checkpoints already evaluated. If
    ``net_final.pt`` and a ``net_step_<N>.pt`` resolve to the same step,
    ``net_final.pt`` sorts first (it's the canonical end-of-training
    artifact even when its step matches a per-step snapshot).
    """
    out: list[tuple[int, Path]] = []
    final_step = _final_step_from_metrics(run_dir, fallback=n_steps)
    for p in sorted(run_dir.iterdir() if run_dir.exists() else []):
        if p.name == "net_final.pt":
            out.append((int(final_step), p))
            continue
        m = _STEP_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    # Sort descending by step; on ties, net_final.pt comes first.
    out.sort(key=lambda t: (-t[0], 0 if t[1].name == "net_final.pt" else 1))
    return out


def _filter_checkpoints(
    checkpoints: list[tuple[int, Path]],
    *,
    stride: int | None,
    steps_csv: str | None,
    final_step: int,
) -> list[tuple[int, Path]]:
    """Apply --stride or --steps filter. Mutually exclusive upstream."""
    if stride is not None:
        # Multiples of N, plus net_final.pt always.
        kept: list[tuple[int, Path]] = []
        for step, path in checkpoints:
            if path.name == "net_final.pt" or step % stride == 0:
                kept.append((step, path))
        return kept
    if steps_csv is not None:
        requested_raw = [s.strip() for s in steps_csv.split(",") if s.strip()]
        requested: set[int] = set()
        for s in requested_raw:
            if s == "final":
                requested.add(int(final_step))
            else:
                requested.add(int(s))
        filtered = [(step, path) for step, path in checkpoints if step in requested]
        missing = requested - {step for step, _ in filtered}
        if missing:
            raise ValueError(f"requested checkpoint steps not found: {sorted(missing)}")
        return filtered
    return checkpoints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training run directory (contains config.yaml + checkpoints).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help=(
            "Eval config name (resolves to scripts/eval-configs/<name>.yaml) "
            "or a literal filesystem path. Default: 'default'."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Slice n_per_depth[:N] from the config.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override config.beam_width.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=("fp32", "bf16", "fp16"),
        help="Override config.precision.",
    )
    parser.add_argument(
        "--include-v-star",
        action="store_true",
        help="Force include_v_star=True (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config.seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override training config.device.",
    )

    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Only evaluate checkpoints whose step is a multiple of this "
            "value (plus net_final.pt always). No default — when omitted, "
            "every checkpoint is fair game."
        ),
    )
    selector.add_argument(
        "--steps",
        type=str,
        default=None,
        help=(
            "Comma-separated step labels (e.g. '20000,30000,final'). "
            "'final' resolves to net_final.pt's step."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute every selected checkpoint even if a record at that "
            "step already exists in the target JSONL. Default behavior is "
            "to skip already-present steps (idempotent / ensure mode)."
        ),
    )
    parser.add_argument(
        "--render-html",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override config.render_html. When enabled, after all evals "
            "complete render a SINGLE overlay HTML at "
            "<run-dir>/results/eval_trajectory_<config>.html driven by the "
            "consolidated <run-dir>/results/beam_eval_<config>.jsonl. (Per-model "
            "renders are suppressed in run mode — one overlay is the point.) "
            "When omitted, the YAML config's render_html field decides."
        ),
    )

    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    config_path = run_dir / "config.yaml"
    results_dir = run_dir / "results"
    empty_state_path = results_dir / "post_run_beam_eval.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    # If config.yaml is absent, behavior depends on whether checkpoints
    # exist: a missing config + no checkpoints is "fresh run dir, nothing
    # to eval" → write empty JSON, exit 0. A missing config + present
    # checkpoints is a corrupt state → error out.
    training_config: DAVIConfig | None = None
    if config_path.exists():
        training_config = DAVIConfig.from_yaml(config_path)

    n_steps = training_config.n_steps if training_config is not None else 0
    checkpoints = _list_checkpoints(run_dir, n_steps)
    final_step = _final_step_from_metrics(run_dir, fallback=n_steps)

    if not checkpoints:
        empty_state_path.write_text(json.dumps({}, indent=2))
        out_display = (
            empty_state_path.relative_to(REPO_ROOT)
            if empty_state_path.is_relative_to(REPO_ROOT)
            else empty_state_path
        )
        print(
            f"no checkpoints in {run_dir}; wrote empty {out_display}",
            flush=True,
        )
        return 0

    if training_config is None:
        raise FileNotFoundError(
            f"{config_path} not found but checkpoints exist in {run_dir}; "
            "cannot reconstruct the value net without it"
        )

    checkpoints = _filter_checkpoints(
        checkpoints,
        stride=args.stride,
        steps_csv=args.steps,
        final_step=final_step,
    )

    # Resolve the eval YAML first so we can clamp against its schedule
    # length (the upper bound on what coverage the YAML can drive).
    eval_cfg = EvalConfig.resolve(args.config)
    yaml_max_depth = eval_cfg.max_depth

    # Auto-derive eval max_depth as min(yaml_max_depth, training_max_depth).
    # Source for training_max_depth: prefer metrics.jsonl's run_start event
    # (authoritative), fall back to config.yaml. This keeps eval coverage
    # within whichever ceiling is lower — never exceeds the YAML schedule
    # (since we have no sample counts past it) and never extends past
    # what the run actually trained on (deeper rows would be vacuous).
    train_kmax, train_source = _training_max_depth_from_run_dir(
        run_dir, fallback_from_config=int(training_config.max_scramble_depth)
    )
    if args.max_depth is None:
        effective_max_depth = min(yaml_max_depth, train_kmax)
        print(
            f"auto-derived max_depth={effective_max_depth} "
            f"(yaml={yaml_max_depth}, training={train_kmax}, "
            f"source={train_source})",
            flush=True,
        )
    else:
        effective_max_depth = int(args.max_depth)
        print(
            f"--max-depth={effective_max_depth} override "
            f"(yaml={yaml_max_depth}, training K_max={train_kmax})",
            flush=True,
        )

    # Apply CLI overrides + the resolved depth. ``with_overrides`` raises
    # if an explicit ``--max-depth N`` exceeds the YAML schedule length.
    eval_cfg = eval_cfg.with_overrides(
        beam_width=args.width,
        precision=args.precision,
        seed=args.seed,
        include_v_star=True if args.include_v_star else None,
        max_depth=effective_max_depth,
        render_html=args.render_html,
    )

    device = torch.device(args.device or training_config.device)

    # Pre-load oracle once if needed (avoids re-reading the .npz per checkpoint).
    oracle_arrays = None
    if eval_cfg.include_v_star:
        if not ORACLE_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"--include-v-star requires {ORACLE_CACHE_PATH} (run "
                "scripts/build_v_star_bounded_3x3.py first)."
            )
        oracle_arrays = load_v_star_bounded_3x3_arrays(ORACLE_CACHE_PATH)

    # Consolidated JSONL rollup: one line per checkpoint, sorted ascending
    # by step. Re-runs merge — new entries replace existing ones at the
    # same step, prior steps preserved.
    jsonl_path = results_dir / f"beam_eval_{eval_cfg.source_name}.jsonl"

    # --force always backs up an existing JSONL (regardless of whether
    # its records' schema matches the current invocation) and proceeds
    # from an empty merge state. This keeps prior runs recoverable while
    # making "recompute everything" an unambiguous reset.
    if args.force and jsonl_path.exists():
        backup_path = _backup_jsonl(jsonl_path)
        print(
            f"--force: backed up existing JSONL to {backup_path}",
            flush=True,
        )

    existing_by_step: dict[int, dict] = {}
    if jsonl_path.exists():
        # Load all existing records first, then validate their config
        # signatures against the current invocation. We only get here
        # when --force is OFF (force already moved the file aside above).
        loaded: list[dict] = []
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            loaded.append(rec)

        current_sig = _signature_from_eval_cfg(eval_cfg)
        mismatches: list[tuple[int, tuple]] = []
        first_mismatched_sig: tuple | None = None
        for rec in loaded:
            # Records that carry NONE of the signature fields are treated
            # as opaque (e.g. legacy/pre-schema records, or sentinels in
            # tests). Only records that carry the fields participate in
            # the mismatch check — partial-schema records (some fields
            # present, some not) still get flagged.
            if not any(field in rec for field in _SIGNATURE_FIELDS):
                continue
            sig = _signature_from_record(rec)
            if sig != current_sig:
                step_label = rec.get("step")
                mismatches.append(
                    (int(step_label) if isinstance(step_label, int) else -1, sig)
                )
                if first_mismatched_sig is None:
                    first_mismatched_sig = sig

        if mismatches:
            n_mismatch = len(mismatches)
            # Only count records that carry signature fields — opaque
            # legacy records aren't part of the comparison universe.
            n_eligible = sum(
                1 for rec in loaded if any(field in rec for field in _SIGNATURE_FIELDS)
            )
            count_clause = (
                f"{n_mismatch} record(s)"
                if n_mismatch == n_eligible
                else f"{n_mismatch} of {n_eligible} records"
            )
            existing_line = (
                _format_signature(first_mismatched_sig)
                if first_mismatched_sig is not None
                else "<no records>"
            )
            current_line = _format_signature(current_sig)
            parser.error(
                f"existing JSONL at {jsonl_path} has {count_clause} with a "
                "different config than this invocation.\n"
                f"  Existing: {existing_line}\n"
                f"  Current:  {current_line}\n"
                "Either:\n"
                "  - use a different --config name (records will land in a "
                "new JSONL file), or\n"
                "  - pass --force to back up the existing JSONL and start "
                "fresh."
            )

        for rec in loaded:
            s = rec.get("step")
            if isinstance(s, int):
                existing_by_step[s] = rec

    n_written = 0
    n_skipped = 0
    for filename_step, ckpt_path in checkpoints:
        # Ensure-by-default: skip if a record at this step is already in
        # the JSONL. ``filename_step`` is the best pre-load proxy for
        # the eventual record's step — for legacy bare state_dicts and
        # ordinary bundle checkpoints alike, the filename step matches
        # the resolved step. (The dict-vs-filename drift case from
        # ``test_dict_step_overrides_filename_step`` is exotic; on a
        # mismatch we'd evaluate redundantly and the dedup-by-step
        # write below absorbs it — preferable to silently skipping a
        # checkpoint whose recorded step doesn't match what we expected.)
        if (
            not args.force
            and ckpt_path.name != "net_final.pt"
            and filename_step in existing_by_step
        ):
            print(
                f"skipping step={filename_step} "
                "(already in JSONL — pass --force to recompute)",
                flush=True,
            )
            n_skipped += 1
            continue
        print(
            f"evaluating {ckpt_path.name} (filename_step={filename_step}) ...",
            flush=True,
        )
        payload = evaluate_checkpoint(
            model_path=ckpt_path,
            eval_cfg=eval_cfg,
            device=device,
            training_config=training_config,
            oracle_arrays=oracle_arrays,
        )
        # Prefer the bundle's authoritative step over the
        # filename-extracted one — the file can be renamed but the dict
        # carries the truth. Falls back to filename for legacy
        # bare-state-dict checkpoints (where checkpoint_step is None).
        ckpt_step = payload.get("checkpoint_step")
        step = int(ckpt_step) if ckpt_step is not None else int(filename_step)
        # net_final.pt always evaluates (its step isn't in the filename,
        # so we can't pre-check). The post-load dedup below catches the
        # "already-in-JSONL" case after the work is done — without
        # ``--force`` we drop the redundant payload to preserve the
        # existing record (consistent with the skip semantics for
        # net_step_*.pt above).
        if (
            not args.force
            and ckpt_path.name == "net_final.pt"
            and step in existing_by_step
        ):
            print(
                f"  step={step} already in JSONL — preserving existing "
                "record (pass --force to overwrite)",
                flush=True,
            )
            n_skipped += 1
            continue
        # Annotate with the resolved step so each line carries the
        # canonical training step it came from.
        payload["step"] = step
        existing_by_step[step] = payload
        n_written += 1
        # Persist after each successful eval so partial runs leave
        # usable data on disk — critical for the largest-first workflow
        # (kill anytime, the most-trained checkpoints are already saved).
        sorted_records = [existing_by_step[s] for s in sorted(existing_by_step.keys())]
        jsonl_path.write_text(
            "\n".join(json.dumps(rec) for rec in sorted_records) + "\n"
        )
        deep_rate = (
            payload["per_walk_depth"][-1]["solve_rate"]
            if payload["per_walk_depth"]
            else float("nan")
        )
        print(
            f"  step={step} "
            f"(wall={payload['wall_time_seconds']:.1f}s "
            f"solve_rate@d={eval_cfg.max_depth}={deep_rate:.3f}; "
            f"{len(sorted_records)} record(s) flushed)",
            flush=True,
        )

    sorted_records = [existing_by_step[s] for s in sorted(existing_by_step.keys())]
    if not sorted_records:
        # No checkpoint evaluated and none pre-existing — leave the file
        # absent rather than writing an empty one.
        pass
    elif n_written == 0:
        # All targeted checkpoints were skipped (ensure-by-default) — the
        # file already reflects existing state, no need to rewrite.
        pass
    else:
        # Already written incrementally above; this re-write is a no-op
        # but keeps the file consistent if anything mutated existing_by_step
        # outside the loop in the future.
        jsonl_path.write_text(
            "\n".join(json.dumps(rec) for rec in sorted_records) + "\n"
        )
    skip_note = f", {n_skipped} skipped" if n_skipped else ""
    print(
        f"\n{n_written} checkpoint(s) evaluated{skip_note}; rollup at "
        f"{jsonl_path} ({len(sorted_records)} total record(s))"
    )

    if eval_cfg.render_html and sorted_records:
        import subprocess

        renderer = REPO_ROOT / "scripts" / "render_beam_eval_report.py"
        html_path = results_dir / f"eval_trajectory_{eval_cfg.source_name}.html"
        subprocess.run(
            [
                sys.executable,
                str(renderer),
                "--input",
                str(jsonl_path),
                "--output",
                str(html_path),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        print(f"rendered {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
