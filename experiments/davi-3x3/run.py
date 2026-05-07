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
``macro_v_star_mae`` drives the early-stop monitor.

Early-stop: when ``config.early_stop_enabled`` is True, the monitor
tracks ``macro_v_star_mae`` across evals and fires as a clean exit from
the training loop when the plateau criterion is met (``patience_evals``
consecutive non-improving evals after a warmup of ``min_evals``).

Usage::

    uv run python experiments/davi-3x3/run.py \\
        --config experiments/davi-3x3/configs/<name>.yaml \\
        [--out-dir experiments/davi-3x3/runs/<custom>]

If ``--out-dir`` is omitted, the run lands at
``experiments/davi-3x3/runs/<UTC-ts>_<config-stem>/``.

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
from rubik.training.early_stop import EarlyStopMonitor
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


def _save_checkpoint(
    path: Path,
    net: torch.nn.Module,
    target_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    torch.save(
        {
            "net_state": net.state_dict(),
            "target_net_state": target_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to a prior checkpoint to warm-start from. Accepts both the "
            "new dict format ({net_state, target_net_state, optimizer_state, "
            "step}) and the legacy bare state_dict format. In the legacy "
            "case, target_net is synced from net and Adam moments restart."
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

    config = DAVIConfig.from_yaml(args.config)
    out_dir = _resolve_out_dir(args.out_dir, args.config.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the resolved config alongside the run — a downstream reader
    # of the run dir alone can reproduce it without needing the original
    # config path.
    config.to_yaml(out_dir / "config.yaml")

    # Optional W&B sink. wandb is opt-out (--no-wandb); auth/network
    # failures degrade gracefully to JSONL-only with a stderr warning.
    wandb_run: WandbAdapter | None = None
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
            wandb_run = WandbAdapter(wandb.run)
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

    # Bounded V* oracle: lookup table for ``value_eval`` per-V* MAE. The
    # dict-shaped form ``dict[bytes, int]`` is what ``value_eval``
    # consumes; ``lookup_v_star_bounded_3x3_batch`` does the bulk lookups.
    oracle_dict = _load_oracle_dict()

    # Early-stop monitor — only used when config.early_stop_enabled.
    early_stop_monitor: EarlyStopMonitor | None = None
    if config.early_stop_enabled:
        if config.early_stop_metric != "macro_v_star_mae":
            raise ValueError(
                "early_stop_metric currently only supports 'macro_v_star_mae'; "
                f"got {config.early_stop_metric!r}"
            )
        early_stop_monitor = EarlyStopMonitor(
            patience_evals=config.early_stop_patience_evals,
            min_evals=config.early_stop_min_evals,
            min_delta=config.early_stop_min_delta,
        )

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
    if config.early_stop_enabled:
        print(
            f"early_stop:             on, metric={config.early_stop_metric}, "
            f"patience_evals={config.early_stop_patience_evals}, "
            f"min_evals={config.early_stop_min_evals}, "
            f"min_delta={config.early_stop_min_delta}"
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
            logger.log(
                event="run_start",
                n_params=n_params,
                device=str(device),
                body_widths=list(config.body_widths),
                n_residual_blocks=config.n_residual_blocks,
                normalization=config.normalization,
                batch_size=config.batch_size,
                max_scramble_depth=config.max_scramble_depth,
                max_scramble_depth_initial=config.max_scramble_depth_initial,
                max_scramble_depth_ramp_steps=config.max_scramble_depth_ramp_steps,
                target_sync_interval=config.target_sync_interval,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                oracle_size=len(oracle_dict),
                early_stop_enabled=config.early_stop_enabled,
                early_stop_metric=config.early_stop_metric,
                early_stop_patience_evals=config.early_stop_patience_evals,
                early_stop_min_evals=config.early_stop_min_evals,
                early_stop_min_delta=config.early_stop_min_delta,
            )

            if resume_mode is not None:
                logger.log(
                    event="resume",
                    source=str(args.resume),
                    mode=resume_mode,
                    prior_step=resume_prior_step,
                )

            for step in range(1, config.n_steps + 1):
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

                    # NaN safety: until the network's first prediction
                    # falls into a populated v_star_mae bucket the macro
                    # could be NaN. Skip the update in that case — the
                    # patience window doesn't advance until we have a
                    # real number.
                    if early_stop_monitor is not None and macro == macro:
                        should_stop = early_stop_monitor.update(macro)
                        if should_stop:
                            snap = early_stop_monitor.state()
                            logger.log(
                                event="early_stop",
                                step=step,
                                n_evals=snap.n_updates,
                                best_macro_v_star_mae=snap.best_value,
                                # convert 0-indexed history idx → the
                                # training step at which the best eval
                                # was logged.
                                best_eval_step=(
                                    ((snap.best_index or 0) + 1) * config.eval_every
                                ),
                                current_macro_v_star_mae=macro,
                                patience_evals=config.early_stop_patience_evals,
                                min_evals=config.early_stop_min_evals,
                                min_delta=config.early_stop_min_delta,
                            )
                            print(
                                f"  early-stop fired @ step {step}  "
                                f"best_macro_v_star_mae "
                                f"{snap.best_value:.4f} "
                                f"(eval idx {snap.best_index})  "
                                f"current {macro:.4f}",
                                flush=True,
                            )
                            early_stopped = True
                            break

                if config.checkpoint_every and step % config.checkpoint_every == 0:
                    ckpt_path = out_dir / f"net_step_{step}.pt"
                    _save_checkpoint(ckpt_path, net, target_net, optimizer, step)
                    logger.log(event="checkpoint", step=step, path=str(ckpt_path.name))

            # Final checkpoint regardless of stop reason. Use the last step
            # actually trained (= ``step`` from the loop variable, which
            # holds either the early-stop step or n_steps).
            final_step = step  # noqa: F821 — bound by the loop above
            _save_checkpoint(
                out_dir / "net_final.pt", net, target_net, optimizer, final_step
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
        if early_stopped:
            print(f"early-stopped at step {final_step}")
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
