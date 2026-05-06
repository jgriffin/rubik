"""DAVI training configuration.

Frozen dataclass; YAML-serialized for ``experiments/davi-2x2/...``.

**Every field is required. No defaults.** The reason: the right values
are an empirical question — pick them by running tier 1+ experiments
yourself, not by inheriting from anywhere. With no defaults, every YAML
on disk carries the exact hyperparameters that produced its run, and
the "story" of any run reads off the YAML alone. There is no implicit
fallback to drift behind your back.

If a yaml file is missing any field, ``from_yaml`` will raise
``TypeError`` from the dataclass constructor — the "missing key" surfaces
loudly rather than silently defaulting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

_VALID_NORMALIZATIONS = ("bn", "none", "ln")


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
    normalization: str  # "bn" | "none" | "ln"; chosen at config time, no default

    # Logging / checkpointing (use 0 to disable any of these)
    log_every: int
    eval_every: int
    checkpoint_every: int

    # Reproducibility / device
    seed: int
    device: str

    def __post_init__(self) -> None:
        if self.normalization not in _VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {_VALID_NORMALIZATIONS!r}, "
                f"got {self.normalization!r}"
            )

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
