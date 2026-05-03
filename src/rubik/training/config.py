"""DAVI training configuration.

Frozen dataclass; YAML-serialized for `experiments/davi-2x2/config.yaml`.
Hyperparameter defaults are DeepCubeA's published values (the M7 sweep
retunes for 2x2). Body widths and residual count match `ValueNet`'s
defaults so a fresh `DAVIConfig()` reproduces the SPEC baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DAVIConfig:
    # Curriculum
    max_scramble_depth: int = 14

    # Optimizer
    batch_size: int = 1000
    n_steps: int = 100_000
    learning_rate: float = 1e-4
    target_sync_interval: int = 5000

    # Network architecture (mirrors ValueNet defaults — kept here so
    # config files can override without touching code).
    body_widths: tuple[int, int] = (5000, 1000)
    n_residual_blocks: int = 4

    # Logging / checkpointing
    log_every: int = 100
    eval_every: int = 5000
    checkpoint_every: int = 10_000

    # Reproducibility / device
    seed: int = 0
    device: str = "mps"

    def to_dict(self) -> dict:
        d = asdict(self)
        # YAML can't round-trip tuples; store as list.
        d["body_widths"] = list(self.body_widths)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DAVIConfig:
        d = dict(d)
        if "body_widths" in d:
            d["body_widths"] = tuple(d["body_widths"])
        return cls(**d)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> DAVIConfig:
        return cls.from_dict(yaml.safe_load(path.read_text()))
