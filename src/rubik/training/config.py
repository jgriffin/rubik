"""DAVI training configuration.

Frozen dataclass; YAML-serialized for ``experiments/davi-2x2/...``.

**Every field is required.** No defaults. Two reasons:

1. *Forces every YAML explicit.* You cannot accidentally inherit a
   hyperparameter from DeepCubeA (or anywhere else) — the YAML carries
   exactly the values the run actually used. The "story" we tell about
   any run reads off the YAML, not off code defaults that may have
   shifted under us.

2. *Honest comparison.* M5's tiered-experimentation methodology earns
   its configuration from tier 0 calibration → tier 1 sanity → tier 2
   sweeps → tier 3 champion. Borrowing DeepCubeA values would skip the
   methodology. DeepCubeA's published defaults live as a *baseline* in
   ``experiments/davi-2x2/baselines/deepcubea_defaults.yaml`` for
   comparison, never as the constructor default.

If a yaml file is missing any field, ``from_yaml`` will raise
``TypeError`` from the dataclass constructor — the "missing key" surfaces
loudly rather than silently defaulting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DAVIConfig:
    # Curriculum
    max_scramble_depth: int

    # Optimizer
    batch_size: int
    n_steps: int
    learning_rate: float
    target_sync_interval: int

    # Network architecture
    body_widths: tuple[int, int]
    n_residual_blocks: int

    # Logging / checkpointing (use 0 to disable any of these)
    log_every: int
    eval_every: int
    checkpoint_every: int

    # Reproducibility / device
    seed: int
    device: str

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
