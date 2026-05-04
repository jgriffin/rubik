"""T1 sanity check: memorize a fixed 1k-state V* subset.

Phase A's failure mode (val_mae ~0.90 across 23x param scaling) was
diagnostic-confirmed loss-distribution-shaped, not capacity-shaped. Before
re-architecting T1 around loss/normalization choice, we want to **rule out
machinery bugs**: if a network with way-more-than-enough capacity can't
drive train MSE to ~0 on a tiny fixed dataset, the bug is in the training
loop / data path / model — not the data distribution.

Approach: depth-stratified-sampling 1000 states from the V* corpus
(proportional to V*'s depth distribution, deterministic by ``subset_seed``),
train **full-batch** with MSE for ``n_steps`` and watch train_mse. Pass
criterion: ``train_mse <= 1e-3 within n_steps``.

Mirrors ``supervised.py`` conventions: ``SupervisedConfig``-shaped dataclass,
``MetricLogger``, run-dir layout with ``config.yaml`` + ``net_final.pt`` +
``metrics.jsonl``. Adds ``subset_idx.npy`` for reproducibility.

Usage::

    uv run python experiments/davi-2x2/t1-capacity/memorize_1k.py \\
        --config experiments/davi-2x2/t1-capacity/configs/<name>.yaml \\
        [--out-dir experiments/davi-2x2/runs/<custom>]
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


@dataclass(frozen=True)
class MemorizeConfig:
    """Config for the memorize-1k sanity check. Every field required."""

    # Network architecture
    body_widths: tuple[int, int]
    n_residual_blocks: int

    # Optimizer / training
    batch_size: int  # set equal to subset_size for full-batch
    n_steps: int
    learning_rate: float
    loss: str  # "mse" or "l1"; chosen at config time, no default

    # Subset construction
    subset_size: int  # how many states to memorize (default 1000)
    subset_seed: int  # pins the depth-stratified subset draw

    # Reproducibility / device
    seed: int
    device: str

    # Eval / logging
    log_every: int  # log train_mse + train_mae every N steps
    val_eval_every: int  # courtesy: eval val every N steps (0 = off)
    val_eval_n_states: int  # how many V* states (excluding the 1k subset) to eval on
    val_eval_seed: int  # pins the val slice

    # Pass criterion
    pass_train_mse: float  # train MSE at or below this counts as memorized

    def __post_init__(self) -> None:
        if self.loss not in _VALID_LOSSES:
            raise ValueError(
                f"loss must be one of {_VALID_LOSSES!r}, got {self.loss!r}"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["body_widths"] = list(self.body_widths)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> MemorizeConfig:
        d = dict(d)
        if "body_widths" in d:
            d["body_widths"] = tuple(d["body_widths"])
        return cls(**d)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> MemorizeConfig:
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


def _depth_stratified_subset(
    depths: np.ndarray, subset_size: int, subset_seed: int
) -> np.ndarray:
    """Pick `subset_size` indices proportional to V*'s depth distribution.

    Quotas per depth use largest-remainder rounding so they sum to exactly
    ``subset_size``. Within each depth, indices are sampled uniformly
    without replacement. Deterministic in ``subset_seed``.
    """
    rng = np.random.default_rng(subset_seed)
    n_total = depths.shape[0]
    unique_depths, counts = np.unique(depths, return_counts=True)

    # Largest-remainder allocation: floor(p_i * subset_size), then distribute
    # leftover to the largest remainders.
    raw = counts.astype(np.float64) / n_total * subset_size
    quotas = np.floor(raw).astype(np.int64)
    leftover = subset_size - int(quotas.sum())
    if leftover > 0:
        # Highest fractional parts get the extra slots; ties broken by depth order.
        remainders = raw - quotas
        order = np.argsort(-remainders, kind="stable")
        quotas[order[:leftover]] += 1
    assert int(quotas.sum()) == subset_size

    chosen = []
    for d, q in zip(unique_depths, quotas, strict=True):
        if q == 0:
            continue
        depth_idx = np.flatnonzero(depths == d)
        # If quota > available (shouldn't happen at subset_size=1000 vs 3.67M),
        # cap to available — caller's deterministic-rounding upstream handles it.
        take = min(int(q), depth_idx.size)
        picked = rng.choice(depth_idx, size=take, replace=False)
        chosen.append(picked)

    out = np.concatenate(chosen)
    rng.shuffle(out)  # break depth-grouping so full-batch isn't depth-ordered
    return out


def _resolve_out_dir(arg: Path | None, config_stem: str) -> Path:
    if arg is not None:
        return arg
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{ts}_{config_stem}"


def _eval_val_mae(
    net: torch.nn.Module,
    val_states: torch.Tensor,
    val_depths: torch.Tensor,
    chunk: int,
) -> float:
    """Mean absolute error of net predictions vs V* on the val slice."""
    net.eval()
    total_abs = 0.0
    n = val_states.shape[0]
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            preds = net(val_states[start:end])
            total_abs += (preds - val_depths[start:end]).abs().sum().item()
    net.train()
    return total_abs / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = MemorizeConfig.from_yaml(args.config)
    out_dir = _resolve_out_dir(args.out_dir, args.config.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_dir / "config.yaml")

    if config.batch_size != config.subset_size:
        # Full-batch is the design — warn loudly if not.
        print(
            f"WARN: batch_size ({config.batch_size}) != subset_size "
            f"({config.subset_size}); script is intended for full-batch."
        )

    device = torch.device(config.device)
    spec = CUBE_2X2

    torch.manual_seed(config.seed)

    # Load V* cache + pick stratified subset.
    states_np, depths_np = _load_v_star()
    subset_idx = _depth_stratified_subset(
        depths_np, config.subset_size, config.subset_seed
    )
    np.save(out_dir / "subset_idx.npy", subset_idx)

    subset_states_np = states_np[subset_idx]
    subset_depths_np = depths_np[subset_idx]

    # Move subset to device — tiny (1k * 24 int8 = 24KB).
    subset_states = torch.from_numpy(subset_states_np).to(device)
    subset_targets = torch.from_numpy(subset_depths_np).to(device).float()

    # Optional: build a small val slice from non-subset indices for courtesy
    # generalization probe. (Not the pass criterion.)
    val_states = val_targets = None
    if config.val_eval_every and config.val_eval_n_states:
        val_rng = np.random.default_rng(config.val_eval_seed)
        n_total = states_np.shape[0]
        in_subset = np.zeros(n_total, dtype=bool)
        in_subset[subset_idx] = True
        candidate_idx = np.flatnonzero(~in_subset)
        take = min(config.val_eval_n_states, candidate_idx.size)
        val_idx = val_rng.choice(candidate_idx, size=take, replace=False)
        val_states = torch.from_numpy(states_np[val_idx]).to(device)
        val_targets = torch.from_numpy(depths_np[val_idx]).to(device).float()

    # Build net.
    net = ValueNet(
        spec,
        body_widths=config.body_widths,
        n_residual_blocks=config.n_residual_blocks,
    ).to(device)
    optimizer = Adam(net.parameters(), lr=config.learning_rate)

    n_params = sum(p.numel() for p in net.parameters())
    try:
        run_dir_display = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        run_dir_display = out_dir

    # Per-depth count summary for sanity.
    sub_unique, sub_counts = np.unique(subset_depths_np, return_counts=True)
    quota_str = ", ".join(
        f"d{int(d)}={int(c)}" for d, c in zip(sub_unique, sub_counts, strict=True)
    )

    print(f"run dir:    {run_dir_display}")
    print(f"device:     {device}")
    print(f"n_params:   {n_params / 1e6:.2f}M")
    print(f"body:       {config.body_widths}, n_res={config.n_residual_blocks}")
    print(f"loss:       {config.loss}")
    print(f"subset:     {config.subset_size} states, seed {config.subset_seed}")
    print(f"  quotas:   {quota_str}")
    print(f"steps:      {config.n_steps}, lr {config.learning_rate}, full-batch")
    print(f"pass when:  train_mse <= {config.pass_train_mse}")
    print()

    loss_fn = F.mse_loss if config.loss == "mse" else F.l1_loss

    pass_step: int | None = None
    final_train_mse = float("nan")
    final_train_mae = float("nan")

    with MetricLogger(out_dir / "metrics.jsonl") as logger:
        logger.log(
            event="run_start",
            n_params=n_params,
            device=str(device),
            subset_size=config.subset_size,
            subset_seed=config.subset_seed,
            n_steps=config.n_steps,
            learning_rate=config.learning_rate,
            body_widths=list(config.body_widths),
            n_residual_blocks=config.n_residual_blocks,
            loss=config.loss,
            pass_train_mse=config.pass_train_mse,
        )

        # Step 0: log untrained loss for the curve's left edge.
        with torch.no_grad():
            preds0 = net(subset_states)
            init_mse = F.mse_loss(preds0, subset_targets).item()
            init_mae = (preds0 - subset_targets).abs().mean().item()
        logger.log(
            event="step",
            step=0,
            train_mse=init_mse,
            train_mae=init_mae,
            step_seconds=0.0,
        )
        print(
            f"step {0:>5d}/{config.n_steps}  "
            f"train_mse {init_mse:.4f}  train_mae {init_mae:.4f}",
            flush=True,
        )

        for step in range(1, config.n_steps + 1):
            t0 = time.perf_counter()

            preds = net(subset_states)
            loss = loss_fn(preds, subset_targets)
            # Always compute MSE for the pass criterion, regardless of which
            # loss the gradient followed.
            with torch.no_grad():
                err = preds - subset_targets
                train_mse_t = (err * err).mean()
                train_mae_t = err.abs().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_seconds = time.perf_counter() - t0

            train_mse = train_mse_t.item()
            train_mae = train_mae_t.item()

            if pass_step is None and train_mse <= config.pass_train_mse:
                pass_step = step
                logger.log(
                    event="pass",
                    step=step,
                    train_mse=train_mse,
                    train_mae=train_mae,
                )
                print(
                    f"  -> PASS: train_mse {train_mse:.6f} <= "
                    f"{config.pass_train_mse} at step {step}",
                    flush=True,
                )

            if config.log_every and step % config.log_every == 0:
                logger.log(
                    event="step",
                    step=step,
                    train_mse=train_mse,
                    train_mae=train_mae,
                    step_seconds=step_seconds,
                )

            if (
                config.val_eval_every
                and val_states is not None
                and step % config.val_eval_every == 0
            ):
                val_mae = _eval_val_mae(net, val_states, val_targets, chunk=4096)
                logger.log(event="val_eval", step=step, val_mae=val_mae)
                print(
                    f"step {step:>5d}/{config.n_steps}  "
                    f"train_mse {train_mse:.4f}  train_mae {train_mae:.4f}  "
                    f"val_mae {val_mae:.4f}  step_ms {step_seconds * 1000:.1f}",
                    flush=True,
                )
            elif (
                config.log_every
                and step % config.log_every == 0
                and step % max(1, config.log_every * 10) == 0
            ):
                # Print every 10*log_every for terminal readability.
                print(
                    f"step {step:>5d}/{config.n_steps}  "
                    f"train_mse {train_mse:.4f}  train_mae {train_mae:.4f}  "
                    f"step_ms {step_seconds * 1000:.1f}",
                    flush=True,
                )

            final_train_mse = train_mse
            final_train_mae = train_mae

        verdict = "PASS" if pass_step is not None else "FAIL"
        logger.log(
            event="run_end",
            step=config.n_steps,
            final_train_mse=final_train_mse,
            final_train_mae=final_train_mae,
            pass_step=pass_step,
            verdict=verdict,
        )
        torch.save(net.state_dict(), out_dir / "net_final.pt")

    print()
    print(f"final train_mse: {final_train_mse:.6f}")
    print(f"final train_mae: {final_train_mae:.6f}")
    if pass_step is not None:
        print(f"verdict:         PASS at step {pass_step}")
    else:
        print(
            f"verdict:         FAIL "
            f"(train_mse > {config.pass_train_mse} at step {config.n_steps})"
        )
    metrics_path = out_dir / "metrics.jsonl"
    try:
        metrics_display = metrics_path.relative_to(REPO_ROOT)
    except ValueError:
        metrics_display = metrics_path
    print(f"metrics:         {metrics_display}")


if __name__ == "__main__":
    main()
