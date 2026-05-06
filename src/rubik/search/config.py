"""Beam-search evaluation configuration.

Frozen dataclass; YAML-serialized for ``experiments/beam-search-2x2/...``.

**Every field is required. No defaults.** Same convention as ``DAVIConfig``:
the right values are an empirical question, every YAML on disk carries the
exact hyperparameters that produced its run, and missing fields surface
loudly as ``TypeError`` from the dataclass constructor rather than
silently defaulting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

_VALID_NORMALIZATIONS = ("bn", "none", "ln")


@dataclass(frozen=True)
class BeamEvalConfig:
    # Source checkpoint to evaluate. Repo-relative path string.
    checkpoint_path: str

    # Network architecture (must match the trained checkpoint).
    body_widths: tuple[int, int]
    n_residual_blocks: int
    normalization: str  # "bn" | "none" | "ln"

    # Eval grid.
    depths: tuple[int, ...]
    beam_widths: tuple[int, ...]
    n_per_depth: int
    max_steps: int

    # Reproducibility / device.
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
        # YAML can't round-trip tuples; store as list. All three tuple fields
        # need this conversion (and the inverse on load).
        d["body_widths"] = list(self.body_widths)
        d["depths"] = list(self.depths)
        d["beam_widths"] = list(self.beam_widths)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BeamEvalConfig:
        d = dict(d)
        if "body_widths" in d:
            d["body_widths"] = tuple(d["body_widths"])
        if "depths" in d:
            d["depths"] = tuple(d["depths"])
        if "beam_widths" in d:
            d["beam_widths"] = tuple(d["beam_widths"])
        return cls(**d)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> BeamEvalConfig:
        return cls.from_dict(yaml.safe_load(path.read_text()))
