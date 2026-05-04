"""T1 supervised V* regression trainer.

Tier 1 asks a capacity question: what's the smallest network that can
fit V* on the 2x2 to val-MAE < 0.5? It does **not** ask about DAVI
dynamics. So this trainer bypasses DAVI entirely — it loads the cached
V* table (`data/v_star_2x2.npz`, 3.67M canonical states with their
optimal cost-to-go), seed-deterministically splits 80/20 into train/val,
and trains a `ValueNet` against the V* labels with the loss configured
in YAML (``loss: mse`` or ``loss: l1``).

If a config can't fit V* with direct supervision, capacity is the
bottleneck — no training-dynamics fix would help. That separation is
the whole point of T1.

Usage::

    uv run python experiments/davi-2x2/t1-capacity/supervised.py \\
        --config experiments/davi-2x2/t1-capacity/configs/<name>.yaml \\
        [--out-dir experiments/davi-2x2/runs/<custom>]

Per-cell variants share the same train/val split (split_seed pinned in
the config) so val MAE numbers are directly comparable across cells.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam

from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ValueNet
from rubik.training.metric_logger import MetricLogger

REPO_ROOT = Path(__file__).resolve().parents[3]
V_STAR_CACHE = REPO_ROOT / "data" / "v_star_2x2.npz"
DEFAULT_RUNS_DIR = REPO_ROOT / "experiments" / "davi-2x2" / "runs"


_VALID_LOSSES = ("mse", "l1")
_VALID_NORMALIZATIONS = ("bn", "none", "ln")
_VALID_SAMPLERS = ("uniform", "depth_balanced")


@dataclass(frozen=True)
class SupervisedConfig:
    """Config for T1 supervised V* regression. Every field required."""

    # Network architecture (parameterized on CubeSpec at runtime)
    body_widths: tuple[int, int]
    n_residual_blocks: int
    normalization: str  # "bn" | "none" | "ln"; chosen at config time, no default

    # Optimizer
    batch_size: int
    n_steps: int
    learning_rate: float
    loss: str  # "mse" or "l1"; chosen at config time, no default
    sampler: str  # "uniform" or "depth_balanced"; chosen at config time

    # Reproducibility / device
    seed: int
    split_seed: int  # pins the 80/20 train/val partition across cells
    device: str

    # Eval
    eval_every: int  # eval val MAE every N steps (0 = end only)
    eval_n_states: int  # cap val eval set to this many states (random subset)
    log_every: int

    def __post_init__(self) -> None:
        if self.loss not in _VALID_LOSSES:
            raise ValueError(
                f"loss must be one of {_VALID_LOSSES!r}, got {self.loss!r}"
            )
        if self.normalization not in _VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {_VALID_NORMALIZATIONS!r}, "
                f"got {self.normalization!r}"
            )
        if self.sampler not in _VALID_SAMPLERS:
            raise ValueError(
                f"sampler must be one of {_VALID_SAMPLERS!r}, got {self.sampler!r}"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["body_widths"] = list(self.body_widths)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SupervisedConfig:
        d = dict(d)
        if "body_widths" in d:
            d["body_widths"] = tuple(d["body_widths"])
        return cls(**d)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> SupervisedConfig:
        return cls.from_dict(yaml.safe_load(path.read_text()))


def _load_v_star() -> tuple[np.ndarray, np.ndarray]:
    """Load V* cache as (states, depths) numpy arrays."""
    if not V_STAR_CACHE.exists():
        raise FileNotFoundError(
            f"V* cache not found at {V_STAR_CACHE}. Run "
            "`uv run python -m rubik.oracle.v_star_2x2` first."
        )
    with np.load(V_STAR_CACHE) as data:
        return data["states"].copy(), data["depths"].copy()


def _split_train_val(
    n: int, split_seed: int, val_frac: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Seed-deterministic 80/20 index split."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _resolve_out_dir(arg: Path | None, config_stem: str) -> Path:
    if arg is not None:
        return arg
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{ts}_{config_stem}"


def _eval_val(
    net: torch.nn.Module,
    val_states: torch.Tensor,
    val_depths: torch.Tensor,
    batch_size: int,
) -> dict:
    """Eval on the val set: val MAE, macro-MAE, per-depth MAE, pred mean/std.

    Returns a dict with keys ``val_mae`` (uniform mean over states),
    ``macro_mae`` (uniform mean across per-depth MAEs — depth-distribution
    invariant), ``per_depth_mae`` (dict[int, float]), ``pred_mean``,
    ``pred_std``. macro-MAE is the metric T1b chose: under depth-balanced
    sampling the val set is still bulk-dominated (V*'s peaked depth
    distribution), so uniform val_mae rewards predict-the-mean — macro-MAE
    treats each depth equally and can't be gamed that way.
    """
    net.eval()
    pred_chunks: list[torch.Tensor] = []
    n = val_states.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            preds = net(val_states[start:end])
            pred_chunks.append(preds.detach().cpu())
    net.train()

    all_preds = torch.cat(pred_chunks)
    val_depths_cpu = val_depths.detach().cpu()
    abs_errors = (all_preds - val_depths_cpu).abs()

    val_mae = float(abs_errors.mean().item())
    pred_mean = float(all_preds.mean().item())
    pred_std = float(all_preds.std(unbiased=False).item())

    # Per-depth MAE — group abs errors by depth label, then average.
    depth_ints = val_depths_cpu.to(torch.int64)
    per_depth_mae: dict[int, float] = {}
    for d in torch.unique(depth_ints).tolist():
        mask = depth_ints == d
        per_depth_mae[int(d)] = float(abs_errors[mask].mean().item())
    macro_mae = float(sum(per_depth_mae.values()) / len(per_depth_mae))

    return {
        "val_mae": val_mae,
        "macro_mae": macro_mae,
        "per_depth_mae": per_depth_mae,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = SupervisedConfig.from_yaml(args.config)
    out_dir = _resolve_out_dir(args.out_dir, args.config.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_dir / "config.yaml")

    device = torch.device(config.device)
    spec = CUBE_2X2

    torch.manual_seed(config.seed)

    # Load V* cache + split
    states_np, depths_np = _load_v_star()
    n_total = states_np.shape[0]
    train_idx, val_idx = _split_train_val(n_total, config.split_seed)

    # Subsample val for eval speed (fixed by split_seed for comparability)
    val_rng = np.random.default_rng(config.split_seed + 1)
    if val_idx.size > config.eval_n_states:
        val_eval_idx = val_rng.choice(val_idx, size=config.eval_n_states, replace=False)
    else:
        val_eval_idx = val_idx

    # Move data to device (3.67M × 24 int8 = 88MB; depths int8 = 3.67MB; trivial)
    states_dev = torch.from_numpy(states_np).to(device)
    depths_dev = torch.from_numpy(depths_np).to(device).float()
    train_idx_dev = torch.from_numpy(train_idx.astype(np.int64)).to(device)

    val_states = states_dev[val_eval_idx]
    val_depths = depths_dev[val_eval_idx]

    # If using depth-balanced sampling, pre-bucket the train indices by depth
    # so per-step batch construction can sample equally from each bucket.
    train_depth_buckets: list[torch.Tensor] | None = None
    bucket_take: int = 0
    bucket_extras: int = 0
    if config.sampler == "depth_balanced":
        train_depths_np = depths_np[train_idx]
        unique_depths = np.unique(train_depths_np)
        train_depth_buckets = []
        for d in unique_depths:
            in_bucket = np.flatnonzero(train_depths_np == d)
            # store as device-side LongTensor of train_idx values for the bucket
            bucket_train_idx = train_idx[in_bucket].astype(np.int64)
            train_depth_buckets.append(torch.from_numpy(bucket_train_idx).to(device))
        n_buckets = len(train_depth_buckets)
        # batch_size split equally per bucket (with replacement) — extras
        # distributed to the first `extras` buckets so total == batch_size.
        bucket_take = config.batch_size // n_buckets
        bucket_extras = config.batch_size - bucket_take * n_buckets
        bucket_counts = [int(b.numel()) for b in train_depth_buckets]
        print(
            f"sampler:    depth_balanced, {n_buckets} buckets, "
            f"~{bucket_take}/bucket (+{bucket_extras} extras)"
        )
        bucket_summary = ", ".join(
            f"d{int(d)}={c}" for d, c in zip(unique_depths, bucket_counts, strict=True)
        )
        print(f"  bucket sizes: {bucket_summary}")
    else:
        print(f"sampler:    {config.sampler}")

    # Build net
    net = ValueNet(
        spec,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
        normalization=config.normalization,
    ).to(device)
    optimizer = Adam(net.parameters(), lr=config.learning_rate)

    n_params = sum(p.numel() for p in net.parameters())
    try:
        run_dir_display = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        run_dir_display = out_dir
    print(f"run dir:    {run_dir_display}")
    print(f"device:     {device}")
    print(f"n_params:   {n_params / 1e6:.2f}M")
    print(f"body:       {config.body_widths}, n_res={config.n_residual_blocks}")
    print(f"norm:       {config.normalization}")
    print(f"loss:       {config.loss}")
    print(
        f"train/val:  {train_idx.size:,} / {val_idx.size:,}  "
        f"(val eval: {val_eval_idx.size:,})"
    )
    print(f"steps:      {config.n_steps}")
    print()

    loss_fn = F.mse_loss if config.loss == "mse" else F.l1_loss

    # Sampler RNG — seed pinned by config.seed (independent of split_seed)
    sample_gen = torch.Generator(device="cpu").manual_seed(config.seed)

    with MetricLogger(out_dir / "metrics.jsonl") as logger:
        logger.log(
            event="run_start",
            n_params=n_params,
            device=str(device),
            n_train=int(train_idx.size),
            n_val=int(val_idx.size),
            n_val_eval=int(val_eval_idx.size),
            body_widths=list(config.body_widths),
            n_residual_blocks=config.n_residual_blocks,
            normalization=config.normalization,
            loss=config.loss,
            sampler=config.sampler,
        )

        for step in range(1, config.n_steps + 1):
            t0 = time.perf_counter()

            if config.sampler == "uniform":
                # Sample batch_size training indices uniformly with replacement.
                # CPU sampling -> device gather; matches DAVI's batch-build pattern.
                sample_offsets = torch.randint(
                    0, train_idx.size, (config.batch_size,), generator=sample_gen
                )
                batch_idx = train_idx_dev[sample_offsets.to(device)]
            else:
                # depth_balanced: ~bucket_take per depth (with replacement so
                # tiny buckets like depth 0 can still contribute), distribute
                # `bucket_extras` to the first buckets to total batch_size.
                assert train_depth_buckets is not None
                pieces: list[torch.Tensor] = []
                for i, bucket in enumerate(train_depth_buckets):
                    take = bucket_take + (1 if i < bucket_extras else 0)
                    if take == 0:
                        continue
                    offsets_cpu = torch.randint(
                        0, int(bucket.numel()), (take,), generator=sample_gen
                    )
                    pieces.append(bucket[offsets_cpu.to(device)])
                batch_idx = torch.cat(pieces)
            batch_states = states_dev[batch_idx]
            batch_targets = depths_dev[batch_idx]

            preds = net(batch_states)
            loss = loss_fn(preds, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_seconds = time.perf_counter() - t0

            if config.log_every and step % config.log_every == 0:
                logger.log(
                    event="step",
                    step=step,
                    loss=loss.item(),
                    step_seconds=step_seconds,
                )

            if config.eval_every and step % config.eval_every == 0:
                metrics = _eval_val(
                    net, val_states, val_depths, config.batch_size
                )
                logger.log(event="eval", step=step, **metrics)
                print(
                    f"step {step:>5d}/{config.n_steps}  "
                    f"loss {loss.item():.4f}  "
                    f"val_mae {metrics['val_mae']:.4f}  "
                    f"macro_mae {metrics['macro_mae']:.4f}  "
                    f"pred_mean {metrics['pred_mean']:.3f}  "
                    f"pred_std {metrics['pred_std']:.3f}  "
                    f"step_ms {step_seconds * 1000:.1f}",
                    flush=True,
                )

        # Final eval (always, even if eval_every is 0 or doesn't divide evenly)
        final_metrics = _eval_val(
            net, val_states, val_depths, config.batch_size
        )
        logger.log(
            event="eval", step=config.n_steps, final=True, **final_metrics
        )
        torch.save(net.state_dict(), out_dir / "net_final.pt")
        logger.log(
            event="run_end",
            step=config.n_steps,
            final_val_mae=final_metrics["val_mae"],
            final_macro_mae=final_metrics["macro_mae"],
            final_pred_mean=final_metrics["pred_mean"],
            final_pred_std=final_metrics["pred_std"],
        )

    print()
    print(f"final val_mae:   {final_metrics['val_mae']:.4f}")
    print(f"final macro_mae: {final_metrics['macro_mae']:.4f}")
    print(f"final pred_mean: {final_metrics['pred_mean']:.4f}")
    print(f"final pred_std:  {final_metrics['pred_std']:.4f}")
    metrics_path = out_dir / "metrics.jsonl"
    try:
        metrics_display = metrics_path.relative_to(REPO_ROOT)
    except ValueError:
        metrics_display = metrics_path
    print(f"metrics:       {metrics_display}")


if __name__ == "__main__":
    main()
