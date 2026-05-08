"""DAVI training entry point for 3x3 — phase 2 onward.

Reads a YAML config (``DAVIConfig`` schema; every field required) and runs
a DAVI training loop on the 3x3 cube. Logs ``(step, loss, step_seconds)``
via ``MetricLogger`` to ``<out_dir>/metrics.jsonl``. Snapshots the
resolved config to ``<out_dir>/config.yaml`` so any run is reproducible
from its output directory alone.

Mirrors ``experiments/davi-2x2/run.py`` cell-for-cell, swapping
``CUBE_2X2`` → ``CUBE_3X3`` and adjusting paths to ``experiments/davi-3x3``.

Eval cadence: every ``config.eval_every`` steps, calls the lean
``value_eval`` (forward pass only, no search) against a deterministic
fresh random-walk eval set generated from the bounded-V* oracle's depth
range, and logs the resulting flat dict as one ``event="eval"`` record.

Early-stop is **disabled** in this script as of 2026-05-06: the
``macro_v_star_mae`` signal that previously drove it is wrong as a stop
trigger when ``max_scramble_depth >> K_oracle=6`` — the bounded oracle
only measures shallow calibration, while the network's natural
prediction range expands far beyond d=6 mid-training. See LOG.md "M8
full 3x3 training run" outcome for the full diagnosis (beam capability
at d=14 climbed +19pp through the macro's claimed plateau). The
``early_stop_*`` ``DAVIConfig`` fields are kept for backwards
compatibility with historical run configs but are no longer acted on
by this script.

Usage::

    uv run python experiments/davi-3x3/run.py \\
        --config experiments/davi-3x3/configs/<name>.yaml \\
        [--out-dir experiments/davi-3x3/runs/<custom>] \\
        [--warm-start <ckpt.pt> --from-step <int>]

If ``--out-dir`` is omitted, the run lands at
``experiments/davi-3x3/runs/<UTC-ts>_<config-stem>/`` — when the config
stem is ``warm_continue`` the resulting dir name carries ``warm_continue``
and is grep-distinguishable from fresh runs.

``--warm-start`` (with required ``--from-step``) is the minimal
warm-start mode: load model state_dict, target_net = copy of model,
optimizer built fresh (Adam moments NOT preserved). Step counter
starts at ``from-step``; first trained step is ``from-step + 1``.
``--warm-start`` and ``--resume`` are mutually exclusive — use
``--resume`` when the goal is to fully restore the
``(net, target_net, optimizer, step)`` triple.

Tier 0 (calibration) does *not* use this script — see
``calibrate_step_time_3x3.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.optim import Adam

from rubik.cube.spec import CUBE_3X3
from rubik.model.network import ValueNet
from rubik.oracle.v_star_bounded_3x3 import load_v_star_bounded_3x3
from rubik.training.config import DAVIConfig
from rubik.training.davi import davi_step, sync_target
from rubik.training.metric_logger import MetricLogger
from rubik.training.scrambles import generate_adi_batch
from rubik.training.wandb_sink import WandbAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "experiments" / "davi-3x3" / "runs"
ORACLE_PATH = REPO_ROOT / "data" / "v_star_bounded_3x3_k6.npz"

# Make the experiment directory importable so this script can `from eval
# import ...` regardless of where it's launched from.
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from eval import value_eval  # noqa: E402


def _resolve_out_dir(arg: Path | None, config_stem: str) -> Path:
    if arg is not None:
        return arg
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{ts}_{config_stem}"


def _load_oracle_dict() -> dict[bytes, int]:
    """Load the bounded V* oracle dict for ``value_eval``.

    Errors with a clear pointer to the build script if the cache is
    missing.
    """
    if not ORACLE_PATH.exists():
        raise FileNotFoundError(
            f"bounded V* oracle cache not found at {ORACLE_PATH}. "
            "Run `uv run python scripts/build_v_star_bounded_3x3.py` first."
        )
    return load_v_star_bounded_3x3(ORACLE_PATH)


def _load_checkpoint(
    path: Path,
    net: torch.nn.Module,
    target_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[str, int]:
    """Load a checkpoint into net / target_net / optimizer in place.

    Returns ``(mode, prior_step)`` where ``mode`` is ``"full"`` if the
    checkpoint carries optimizer + target_net state (new dict format) or
    ``"weights-only"`` if it's a bare ``state_dict`` (legacy format). In
    weights-only mode, ``target_net`` is synced from ``net`` and
    ``optimizer`` is left in its freshly-constructed state (Adam moments
    not restored).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "net_state" in ckpt:
        net.load_state_dict(ckpt["net_state"])
        target_net.load_state_dict(ckpt["target_net_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        return "full", int(ckpt.get("step", 0))
    net.load_state_dict(ckpt)
    sync_target(net, target_net)
    return "weights-only", 0


def _load_warm_start(
    path: Path,
    net: torch.nn.Module,
    target_net: torch.nn.Module,
    device: torch.device,
) -> None:
    """Load model weights from ``path`` into ``net`` and copy to ``target_net``.

    Minimal warm-start (vs. the heavier ``--resume``):
    - Loads ONLY the model state_dict — accepts both the new dict format
      (uses the ``net_state`` key) and the legacy bare-state-dict format.
    - Optimizer state is NOT restored (caller builds it fresh; Adam moments
      restart from zero — small ~500-step distortion acknowledged).
    - ``target_net`` is set to a clean copy of the loaded ``net`` weights
      via ``sync_target`` rather than re-loading from the checkpoint's
      ``target_net_state`` (if present). Rationale: warm-start is "start
      training from these weights with a fresh slate"; full resumability
      that restores the exact (net, target_net, optimizer) triple is the
      ``--resume`` path.

    Architecture mismatch surfaces as ``RuntimeError`` from
    ``load_state_dict`` — the caller config must match the checkpoint's
    arch dimensionally.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "net_state" in ckpt:
        net.load_state_dict(ckpt["net_state"])
    else:
        net.load_state_dict(ckpt)
    sync_target(net, target_net)


def _save_checkpoint(
    path: Path,
    net: torch.nn.Module,
    target_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    wall_time_seconds: float | None = None,
    wandb_run_id: str | None = None,
) -> None:
    """Persist a training-checkpoint bundle.

    The core fields (``net_state``, ``target_net_state``, ``optimizer_state``,
    ``step``) are always present. ``wall_time_seconds`` (training-run
    wall-clock) and ``wandb_run_id`` are optional provenance fields — when
    omitted by the caller, the corresponding key is OMITTED from the
    payload entirely (no ``None`` placeholders) so historical readers
    don't trip on it.
    """
    payload: dict = {
        "net_state": net.state_dict(),
        "target_net_state": target_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
    }
    if wall_time_seconds is not None:
        payload["wall_time_seconds"] = float(wall_time_seconds)
    if wandb_run_id is not None:
        payload["wandb_run_id"] = str(wandb_run_id)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to a prior checkpoint for full resume (net + target_net + "
            "optimizer + step). Accepts both the new dict format "
            "({net_state, target_net_state, optimizer_state, step}) and the "
            "legacy bare state_dict format. In the legacy case, target_net "
            "is synced from net and Adam moments restart. Use --warm-start "
            'for the lighter "load model weights, fresh optimizer" mode.'
        ),
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help=(
            "Path to a checkpoint whose model state_dict to load into the "
            "value net (and copy into the target net). Optimizer is built "
            "fresh (Adam moments NOT preserved). Requires --from-step. "
            "Architecture in the run config must match the checkpoint's "
            "dimensions or load_state_dict will raise. Mutually exclusive "
            "with --resume."
        ),
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=0,
        help=(
            "Initial training-step counter for the run loop. The first "
            "step trained is `from-step + 1`; the loop runs while step "
            "<= n_steps. Required (must be >= 1) when --warm-start is "
            "set; ignored otherwise. Default 0 = fresh run starting at "
            "step 1."
        ),
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help=(
            "Disable Weights & Biases logging for this run. By default, "
            "wandb is enabled and the run is also pushed there alongside "
            "the local JSONL. Use this flag for offline / smoke runs."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="rubik-3x3",
        help="W&B project name. Default: 'rubik-3x3'.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help=(
            "W&B entity (user or team). Default: None — uses the wandb "
            "user's default entity from `wandb login`."
        ),
    )
    args = parser.parse_args()

    # Wall-clock origin for the run — captured before any heavy work so
    # checkpoint ``wall_time_seconds`` reflects real training time elapsed
    # (init + loop), not just loop time. Bundled into every checkpoint
    # save so downstream readers can plot loss/eval against wall-clock
    # without having to cross-reference metrics.jsonl.
    run_start_perf = time.perf_counter()

    # Argument cross-checks for --warm-start / --from-step.
    if args.warm_start is not None and args.resume is not None:
        parser.error(
            "--warm-start and --resume are mutually exclusive. --warm-start "
            "loads model weights only with a fresh optimizer; --resume "
            "restores the full (net, target_net, optimizer, step) triple."
        )
    if args.warm_start is not None and args.from_step < 1:
        parser.error(
            "--warm-start requires --from-step >= 1 (the step counter the "
            f"run loop should resume from); got --from-step={args.from_step}."
        )
    if args.warm_start is None and args.from_step != 0:
        parser.error(
            "--from-step is only meaningful with --warm-start; got "
            f"--from-step={args.from_step} without --warm-start."
        )

    config = DAVIConfig.from_yaml(args.config)
    if args.warm_start is not None and args.from_step >= config.n_steps:
        parser.error(
            f"--from-step ({args.from_step}) must be < config.n_steps "
            f"({config.n_steps}); otherwise the run loop has nothing to do."
        )
    out_dir = _resolve_out_dir(args.out_dir, args.config.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the resolved config alongside the run — a downstream reader
    # of the run dir alone can reproduce it without needing the original
    # config path.
    config.to_yaml(out_dir / "config.yaml")

    # Optional W&B sink. wandb is opt-out (--no-wandb); auth/network
    # failures degrade gracefully to JSONL-only with a stderr warning.
    wandb_run: WandbAdapter | None = None
    wandb_run_id: str | None = None
    wandb_started = False
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=out_dir.name,
                dir=str(out_dir),
                config=config.to_dict(),
                tags=["cube=3x3", "phase=smoke"],
            )
            # Default x-axis on every panel = training step (not W&B's
            # internal `_step` log-call counter). With this set once at
            # init time, every key gets the right x-axis without per-key
            # configuration. See experiments/davi-3x3/wandb_workspace.md.
            wandb.define_metric("*", step_metric="step")
            wandb_run = WandbAdapter(wandb.run)
            # Capture the wandb run id once at init so checkpoint saves
            # don't need to reach back into the wandb module (and to
            # keep the value stable if wandb.run gets nulled out at
            # finish-time).
            wandb_run_id = wandb.run.id if wandb.run is not None else None
            wandb_started = True
        except Exception as e:
            print(
                f"wandb init failed: {e}; continuing with JSONL only — "
                "run `wandb login` to enable",
                file=sys.stderr,
            )
            wandb_run = None

    device = torch.device(config.device)
    spec = CUBE_3X3

    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    # Eval determinism is keyed on a fixed ``seed`` int (config.seed + 17),
    # passed into ``value_eval`` which constructs a fresh torch generator
    # internally per call. Same seed at every eval call → identical eval
    # set across training steps — what changes between calls is the network
    # weights, not the eval distribution. Offset from training-data seed so
    # the eval sample is independent of any given training batch. (This
    # replaces the smoke-run pattern of a single shared generator advanced
    # across calls; that was H4 — per-V* MAE bounced eval-to-eval because
    # the eval distribution drifted with the generator state.)
    eval_seed = config.seed + 17

    net = ValueNet(
        spec,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
        normalization=config.normalization,
    ).to(device)
    target_net = ValueNet(
        spec,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
        normalization=config.normalization,
    ).to(device)
    sync_target(net, target_net)
    optimizer = Adam(net.parameters(), lr=config.learning_rate)

    resume_mode: str | None = None
    resume_prior_step = 0
    if args.resume is not None:
        resume_mode, resume_prior_step = _load_checkpoint(
            args.resume, net, target_net, optimizer, device
        )

    # Warm-start: load model weights only; optimizer stays fresh; target_net
    # mirrors the loaded value_net via sync_target. Mutually exclusive with
    # --resume (validated at argparse time). Step counter starts at
    # args.from_step → first trained step is from_step + 1.
    warm_start_path: Path | None = None
    if args.warm_start is not None:
        _load_warm_start(args.warm_start, net, target_net, device)
        warm_start_path = args.warm_start
    initial_step = int(args.from_step) if args.warm_start is not None else 0

    # Bounded V* oracle: lookup table for ``value_eval`` per-V* MAE. The
    # dict-shaped form ``dict[bytes, int]`` is what ``value_eval``
    # consumes; ``lookup_v_star_bounded_3x3_batch`` does the bulk lookups.
    oracle_dict = _load_oracle_dict()

    # Early-stop wiring removed 2026-05-06: macro_v_star_mae is the wrong
    # signal at K_max >> K_oracle (=6). The bounded oracle only measures
    # shallow calibration, while the network's natural prediction range
    # expands far beyond d=6 mid-training, so macro rising at deep
    # training depths is by design, not regression. See LOG.md "M8 full
    # 3x3 training run" outcome and `tests/training/test_early_stop.py`
    # — the EarlyStopMonitor itself stays for future use; only its
    # wiring into this run loop is removed. DAVIConfig.early_stop_*
    # fields are kept for backwards compat with historical run configs.

    n_params = sum(p.numel() for p in net.parameters())
    try:
        run_dir_display = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        run_dir_display = out_dir
    print(f"run dir:                {run_dir_display}")
    print(f"device:                 {device}")
    print(f"params:                 {n_params / 1e6:.2f}M")
    print(f"body_widths:            {config.body_widths}")
    print(f"n_residual_blocks:      {config.n_residual_blocks}")
    print(f"normalization:          {config.normalization}")
    print(f"batch_size:             {config.batch_size}")
    print(f"max_scramble_depth:     {config.max_scramble_depth}")
    if config.max_scramble_depth_ramp_steps > 0:
        print(
            f"k_max curriculum:       {config.max_scramble_depth_initial}"
            f"→{config.max_scramble_depth} over "
            f"{config.max_scramble_depth_ramp_steps} steps"
        )
    print(f"target_sync_interval:   {config.target_sync_interval}")
    print(f"learning_rate:          {config.learning_rate}")
    print(f"n_steps:                {config.n_steps}")
    print(
        f"oracle:                 {ORACLE_PATH.relative_to(REPO_ROOT)} "
        f"({len(oracle_dict)} states)"
    )
    # Early-stop wiring removed 2026-05-06 (see comment above) — config
    # fields preserved but not acted on. Print a notice so it's obvious
    # at the run banner that legacy `early_stop_enabled: true` configs
    # are silently ignored by this script.
    if config.early_stop_enabled:
        print(
            "early_stop:             ignored (wiring removed 2026-05-06; "
            "macro_v_star_mae is wrong signal at K_max >> K_oracle)"
        )
    else:
        print("early_stop:             off")
    if resume_mode is not None:
        try:
            resume_display = args.resume.relative_to(REPO_ROOT)
        except ValueError:
            resume_display = args.resume
        print(
            f"resumed from:           {resume_display} "
            f"(mode={resume_mode}, prior_step={resume_prior_step})"
        )
    if warm_start_path is not None:
        try:
            warm_display = warm_start_path.relative_to(REPO_ROOT)
        except ValueError:
            warm_display = warm_start_path
        print(
            f"warm-start from:        {warm_display} "
            f"(from_step={initial_step}; fresh optimizer + target_net=copy "
            f"of net)"
        )
    print()

    # Track the most recent eval payload + the step it was logged at, so
    # run_end can surface the canonical "final" metrics regardless of
    # whether the loop exited via early-stop or hit the n_steps cap.
    last_eval_metrics: dict | None = None
    last_eval_step: int | None = None
    early_stopped = False

    try:
        with MetricLogger(out_dir / "metrics.jsonl", wandb_run=wandb_run) as logger:
            # Header record so a downstream analyzer can recover run-level
            # context without parsing the config yaml separately.
            run_start_fields: dict = {
                "n_params": n_params,
                "device": str(device),
                "body_widths": list(config.body_widths),
                "n_residual_blocks": config.n_residual_blocks,
                "normalization": config.normalization,
                "batch_size": config.batch_size,
                "max_scramble_depth": config.max_scramble_depth,
                "max_scramble_depth_initial": config.max_scramble_depth_initial,
                "max_scramble_depth_ramp_steps": config.max_scramble_depth_ramp_steps,
                "target_sync_interval": config.target_sync_interval,
                "learning_rate": config.learning_rate,
                "n_steps": config.n_steps,
                "oracle_size": len(oracle_dict),
                "early_stop_enabled": config.early_stop_enabled,
                "early_stop_metric": config.early_stop_metric,
                "early_stop_patience_evals": config.early_stop_patience_evals,
                "early_stop_min_evals": config.early_stop_min_evals,
                "early_stop_min_delta": config.early_stop_min_delta,
            }
            # Warm-start annotation: optional fields, absent on fresh runs
            # so historical readers don't trip on missing keys.
            if warm_start_path is not None:
                run_start_fields["warm_start_from"] = str(warm_start_path)
                run_start_fields["from_step"] = initial_step
            logger.log(event="run_start", **run_start_fields)

            if resume_mode is not None:
                logger.log(
                    event="resume",
                    source=str(args.resume),
                    mode=resume_mode,
                    prior_step=resume_prior_step,
                )

            # Loop starts at initial_step + 1 (=1 on a fresh run, =from_step+1
            # on a warm-start). Runs through config.n_steps inclusive.
            for step in range(initial_step + 1, config.n_steps + 1):
                t0 = time.perf_counter()
                current_k_max = config.current_k_max(step)
                states, _depths, _last_faces = generate_adi_batch(
                    spec,
                    batch_size=config.batch_size,
                    max_depth=current_k_max,
                    generator=generator,
                )
                states = states.to(device)
                loss = davi_step(net, target_net, optimizer, states, spec)
                step_seconds = time.perf_counter() - t0

                if (
                    config.target_sync_interval
                    and step % config.target_sync_interval == 0
                ):
                    sync_target(net, target_net)

                if config.log_every and step % config.log_every == 0:
                    logger.log(
                        event="step",
                        step=step,
                        loss=loss,
                        step_seconds=step_seconds,
                        k_max=current_k_max,
                    )
                    print(
                        f"step {step:>7d}/{config.n_steps}  "
                        f"loss {loss:.4f}  "
                        f"step {step_seconds * 1000:.1f} ms",
                        flush=True,
                    )

                if config.eval_every and step % config.eval_every == 0:
                    eval_metrics = value_eval(
                        net,
                        spec=spec,
                        oracle_dict=oracle_dict,
                        seed=eval_seed,
                    )
                    logger.log(
                        event="eval",
                        step=step,
                        **eval_metrics,
                    )
                    last_eval_metrics = eval_metrics
                    last_eval_step = step
                    macro = eval_metrics.get("macro_v_star_mae", float("nan"))
                    pred_mean = eval_metrics.get("pred_mean", float("nan"))
                    pred_std = eval_metrics.get("pred_std", float("nan"))
                    print(
                        f"  eval @ step {step:>7d}  "
                        f"macro_v_star_mae {macro:.4f}  "
                        f"pred_mean {pred_mean:.3f}  "
                        f"pred_std {pred_std:.3f}",
                        flush=True,
                    )

                if config.checkpoint_every and step % config.checkpoint_every == 0:
                    ckpt_path = out_dir / f"net_step_{step}.pt"
                    wall = time.perf_counter() - run_start_perf
                    _save_checkpoint(
                        ckpt_path,
                        net,
                        target_net,
                        optimizer,
                        step,
                        wall_time_seconds=wall,
                        wandb_run_id=wandb_run_id,
                    )
                    logger.log(event="checkpoint", step=step, path=str(ckpt_path.name))

            # Final checkpoint regardless of stop reason. Use the last step
            # actually trained (= ``step`` from the loop variable, which
            # holds either the early-stop step or n_steps).
            final_step = step  # noqa: F821 — bound by the loop above
            wall = time.perf_counter() - run_start_perf
            _save_checkpoint(
                out_dir / "net_final.pt",
                net,
                target_net,
                optimizer,
                final_step,
                wall_time_seconds=wall,
                wandb_run_id=wandb_run_id,
            )

            # run_end carries finals so downstream analysis doesn't need to
            # re-read every eval record. Surface the most recent eval if
            # one exists; otherwise emit a slim record.
            run_end_fields: dict = {
                "step": final_step,
                "early_stopped": early_stopped,
                "n_steps_planned": config.n_steps,
            }
            if last_eval_metrics is not None and last_eval_step is not None:
                run_end_fields["final_eval_step"] = last_eval_step
                run_end_fields["final_macro_v_star_mae"] = last_eval_metrics.get(
                    "macro_v_star_mae", float("nan")
                )
                run_end_fields["final_pred_mean"] = last_eval_metrics.get(
                    "pred_mean", float("nan")
                )
                run_end_fields["final_pred_std"] = last_eval_metrics.get(
                    "pred_std", float("nan")
                )
            logger.log(event="run_end", **run_end_fields)

        print()
        if last_eval_metrics is not None:
            print(
                f"final macro_v_star_mae: "
                f"{last_eval_metrics.get('macro_v_star_mae', float('nan')):.4f}"
            )
        # `early_stopped` always False here as of 2026-05-06 (wiring
        # removed), but the field is kept on run_end for downstream-
        # reader backwards compat.
        metrics_path = out_dir / "metrics.jsonl"
        try:
            metrics_display = metrics_path.relative_to(REPO_ROOT)
        except ValueError:
            metrics_display = metrics_path
        print(f"done. metrics:   {metrics_display}")
    finally:
        if wandb_started:
            try:
                import wandb

                wandb.finish()
            except Exception as e:
                print(f"wandb finish failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
