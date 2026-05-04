"""DAVI training entry point — used from tier 1 onward.

Reads a YAML config (``DAVIConfig`` schema; every field required) and runs
a DAVI training loop. Logs ``(step, loss, step_seconds)`` via
``MetricLogger`` to ``<out_dir>/metrics.jsonl``. Snapshots the resolved
config to ``<out_dir>/config.yaml`` so any run is reproducible from its
output directory alone.

Eval cadence: every ``config.eval_every`` steps, evaluates against the
depth-stratified V* eval set (``experiments/davi-2x2/eval_set.npz`` —
built by ``scripts/build_eval_set_2x2.py``) and runs a greedy-policy
solve check across a fixed grid of test depths. Both metrics are logged
as one ``event="eval"`` record per cycle plus a ``event="run_end"``
record with the final values.

Usage::

    uv run python experiments/davi-2x2/run.py \\
        --config experiments/davi-2x2/configs/<name>.yaml \\
        [--out-dir experiments/davi-2x2/runs/<custom>]

If ``--out-dir`` is omitted, the run lands at
``experiments/davi-2x2/runs/<UTC-ts>_<config-stem>/``.

Tier 0 (calibration) does *not* use this script — see
``calibrate_step_time.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ValueNet
from rubik.training.config import DAVIConfig
from rubik.training.davi import davi_step, sync_target
from rubik.training.metric_logger import MetricLogger
from rubik.training.scrambles import generate_adi_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "experiments" / "davi-2x2" / "runs"
EVAL_SET_PATH = REPO_ROOT / "experiments" / "davi-2x2" / "eval_set.npz"

# Make the experiment directory importable so this script can `from eval
# import ...` regardless of where it's launched from.
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from eval import eval_against_v_star, greedy_solve  # noqa: E402


def _resolve_out_dir(arg: Path | None, config_stem: str) -> Path:
    if arg is not None:
        return arg
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{ts}_{config_stem}"


def _load_eval_set(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the depth-stratified V* eval set; return states-on-device + cpu depths."""
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"eval set not found at {EVAL_SET_PATH}. Run "
            "`uv run python scripts/build_eval_set_2x2.py` first."
        )
    with np.load(EVAL_SET_PATH) as data:
        states_np = data["states"].copy()
        depths_np = data["depths"].copy()
    eval_states_dev = torch.from_numpy(states_np).to(device)
    eval_depths_cpu = torch.from_numpy(depths_np)
    return eval_states_dev, eval_depths_cpu


def _format_solve_rates(metrics: dict) -> str:
    parts = []
    for key in sorted(k for k in metrics if k.startswith("solve_rate_d")):
        d = key[len("solve_rate_d") :]
        parts.append(f"d{d}={metrics[key]:.2f}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = DAVIConfig.from_yaml(args.config)
    out_dir = _resolve_out_dir(args.out_dir, args.config.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the resolved config alongside the run — a downstream reader
    # of the run dir alone can reproduce it without needing the original
    # config path.
    config.to_yaml(out_dir / "config.yaml")

    device = torch.device(config.device)
    spec = CUBE_2X2

    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    eval_solve_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)

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

    eval_states_dev, eval_depths_cpu = _load_eval_set(device)

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
    print(f"target_sync_interval:   {config.target_sync_interval}")
    print(f"learning_rate:          {config.learning_rate}")
    print(f"n_steps:                {config.n_steps}")
    print(
        f"eval_set:               {EVAL_SET_PATH.relative_to(REPO_ROOT)} "
        f"({eval_states_dev.shape[0]} states)"
    )
    print()

    with MetricLogger(out_dir / "metrics.jsonl") as logger:
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
            target_sync_interval=config.target_sync_interval,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            eval_set_size=int(eval_states_dev.shape[0]),
        )

        for step in range(1, config.n_steps + 1):
            t0 = time.perf_counter()
            states, _depths, _last_faces = generate_adi_batch(
                spec,
                batch_size=config.batch_size,
                max_depth=config.max_scramble_depth,
                generator=generator,
            )
            states = states.to(device)
            loss = davi_step(net, target_net, optimizer, states, spec)
            step_seconds = time.perf_counter() - t0

            if config.target_sync_interval and step % config.target_sync_interval == 0:
                sync_target(net, target_net)

            if config.log_every and step % config.log_every == 0:
                logger.log(
                    event="step",
                    step=step,
                    loss=loss,
                    step_seconds=step_seconds,
                )
                print(
                    f"step {step:>7d}/{config.n_steps}  "
                    f"loss {loss:.4f}  "
                    f"step {step_seconds * 1000:.1f} ms",
                    flush=True,
                )

            if config.eval_every and step % config.eval_every == 0:
                vstar_metrics = eval_against_v_star(
                    net, eval_states_dev, eval_depths_cpu
                )
                solve_metrics = greedy_solve(net, spec, generator=eval_solve_generator)
                logger.log(
                    event="eval",
                    step=step,
                    **vstar_metrics,
                    **solve_metrics,
                )
                print(
                    f"  eval @ step {step:>7d}  "
                    f"macro_mae {vstar_metrics['macro_mae']:.4f}  "
                    f"val_mae {vstar_metrics['val_mae']:.4f}  "
                    f"pred_std {vstar_metrics['pred_std']:.3f}  "
                    f"solve: {_format_solve_rates(solve_metrics)}",
                    flush=True,
                )

            if config.checkpoint_every and step % config.checkpoint_every == 0:
                ckpt_path = out_dir / f"net_step_{step}.pt"
                torch.save(net.state_dict(), ckpt_path)
                logger.log(event="checkpoint", step=step, path=str(ckpt_path.name))

        # Final eval before the last checkpoint, so run_end carries the
        # finals for downstream analysis.
        final_vstar = eval_against_v_star(net, eval_states_dev, eval_depths_cpu)
        final_solve = greedy_solve(net, spec, generator=eval_solve_generator)
        logger.log(
            event="eval",
            step=config.n_steps,
            final=True,
            **final_vstar,
            **final_solve,
        )
        torch.save(net.state_dict(), out_dir / "net_final.pt")
        logger.log(
            event="run_end",
            step=config.n_steps,
            final_macro_mae=final_vstar["macro_mae"],
            final_val_mae=final_vstar["val_mae"],
            final_pred_mean=final_vstar["pred_mean"],
            final_pred_std=final_vstar["pred_std"],
            **{f"final_{k}": v for k, v in final_solve.items()},
        )

    print()
    print(f"final macro_mae: {final_vstar['macro_mae']:.4f}")
    print(f"final val_mae:   {final_vstar['val_mae']:.4f}")
    print(f"final solve:     {_format_solve_rates(final_solve)}")
    metrics_path = out_dir / "metrics.jsonl"
    try:
        metrics_display = metrics_path.relative_to(REPO_ROOT)
    except ValueError:
        metrics_display = metrics_path
    print(f"done. metrics:   {metrics_display}")


if __name__ == "__main__":
    main()
