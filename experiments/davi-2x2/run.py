"""DAVI training entry point — used from tier 1 onward.

Reads a YAML config (``DAVIConfig`` schema; every field required) and runs
a DAVI training loop. Logs ``(step, loss, step_seconds)`` via
``MetricLogger`` to ``<out_dir>/metrics.jsonl``. Snapshots the resolved
config to ``<out_dir>/config.yaml`` so any run is reproducible from its
output directory alone.

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
import time
from datetime import UTC, datetime
from pathlib import Path

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


def _resolve_out_dir(arg: Path | None, config_stem: str) -> Path:
    if arg is not None:
        return arg
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{ts}_{config_stem}"


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

    net = ValueNet(
        spec,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
    ).to(device)
    target_net = ValueNet(
        spec,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
    ).to(device)
    sync_target(net, target_net)
    optimizer = Adam(net.parameters(), lr=config.learning_rate)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"run dir: {out_dir.relative_to(REPO_ROOT)}")
    print(f"device:  {device}")
    print(f"params:  {n_params / 1e6:.1f}M")
    print(f"steps:   {config.n_steps}")
    print()

    with MetricLogger(out_dir / "metrics.jsonl") as logger:
        # Header record so a downstream analyzer can recover run-level
        # context without parsing the config yaml separately.
        logger.log(
            event="run_start",
            n_params=n_params,
            device=str(device),
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

            if config.checkpoint_every and step % config.checkpoint_every == 0:
                ckpt_path = out_dir / f"net_step_{step}.pt"
                torch.save(net.state_dict(), ckpt_path)
                logger.log(event="checkpoint", step=step, path=str(ckpt_path.name))

        torch.save(net.state_dict(), out_dir / "net_final.pt")
        logger.log(event="run_end", step=config.n_steps)

    print()
    print(f"done. metrics: {(out_dir / 'metrics.jsonl').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
