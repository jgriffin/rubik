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
    max_scramble_depth: int  # final / steady-state K_max
    # Curriculum schedule. When ramp_steps == 0: flat — K_max = max_scramble_depth
    # always (initial is ignored). When ramp_steps > 0: K_max linearly grows from
    # max_scramble_depth_initial at step 1 up to max_scramble_depth at step
    # max_scramble_depth_ramp_steps; clamped at max_scramble_depth thereafter.
    max_scramble_depth_initial: int
    max_scramble_depth_ramp_steps: int

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
        if self.max_scramble_depth_ramp_steps < 0:
            raise ValueError(
                "max_scramble_depth_ramp_steps must be >= 0; "
                f"got {self.max_scramble_depth_ramp_steps}"
            )
        if self.max_scramble_depth_ramp_steps > 0:
            if self.max_scramble_depth_initial < 1:
                raise ValueError(
                    "max_scramble_depth_initial must be >= 1 when ramp_steps > 0; "
                    f"got {self.max_scramble_depth_initial}"
                )
            if self.max_scramble_depth_initial > self.max_scramble_depth:
                raise ValueError(
                    "max_scramble_depth_initial must be <= max_scramble_depth; "
                    f"got initial={self.max_scramble_depth_initial}, "
                    f"final={self.max_scramble_depth}"
                )

    def current_k_max(self, step: int) -> int:
        """K_max active at training step ``step`` (1-indexed).

        Flat (ramp_steps == 0): always ``max_scramble_depth``.
        Ramp: linear interpolation from ``max_scramble_depth_initial`` at
        step 1 to ``max_scramble_depth`` at step ``max_scramble_depth_ramp_steps``,
        clamped at ``max_scramble_depth`` thereafter. Floored to int.
        """
        if step < 1:
            raise ValueError(f"step must be >= 1; got {step}")
        if self.max_scramble_depth_ramp_steps == 0:
            return self.max_scramble_depth
        if step >= self.max_scramble_depth_ramp_steps:
            return self.max_scramble_depth
        span = self.max_scramble_depth - self.max_scramble_depth_initial
        frac = (step - 1) / max(self.max_scramble_depth_ramp_steps - 1, 1)
        return int(self.max_scramble_depth_initial + frac * span)

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
