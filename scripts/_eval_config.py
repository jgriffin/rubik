"""Shared loader for beam-eval YAML config files.

Bundles the per-depth sample schedule + beam width + precision + seed for
the post-training beam-eval scripts (``beam_eval_model.py``,
``beam_eval_run.py``, ``beam_eval_sweep.py``). The underscore prefix flags
this as a script-internal helper, not a public package API.

Two named configs ship with the repo at ``scripts/eval-configs/``:

- ``fast`` — cycle-screening eval, deep-tail SE ~6.5pp, ~25-30s wall.
- ``default`` — production cycle-decision eval, deep-tail SE ~4.6pp.

The schema mirrors ``DAVIConfig.from_yaml`` at
``src/rubik/training/config.py``: a small dataclass with a
``from_yaml`` classmethod and explicit per-field validation. Unlike
DAVIConfig this lives at script scope (not under ``src/rubik/``) since
it's only consumed by the eval scripts.

Usage::

    cfg = EvalConfig.resolve("fast")          # named lookup
    cfg = EvalConfig.resolve("/tmp/x.yaml")    # path
    cfg.beam_width        # 256
    cfg.n_per_depth       # [12, 24, ...]
    cfg.precision         # "bf16"

The ``resolve`` helper handles both name lookups (relative to
``scripts/eval-configs/<name>.yaml`` from the repo root) and literal
filesystem paths (anything containing ``/`` or ending in ``.yaml`` /
``.yml``). FileNotFoundError surfaces both the requested name and the
exact search path so the failure mode is greppable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "scripts" / "eval-configs"

_VALID_PRECISIONS = ("fp32", "bf16", "fp16")


@dataclass(frozen=True)
class EvalConfig:
    """Beam-eval YAML config bundle."""

    description: str
    n_per_depth: tuple[int, ...]
    beam_width: int
    precision: str
    seed: int
    include_v_star: bool = False
    n_per_v_star_layer: int = 200
    # Whether to also produce HTML alongside the JSON output. CLI flag
    # --render-html / --no-render-html overrides the YAML choice. For
    # beam_eval_run.py, this drives a single overlay HTML across all
    # checkpoints rather than per-checkpoint files.
    render_html: bool = False
    # Track which file (or named config) this came from — for output
    # filenames + JSON metadata.
    source_path: Path = field(default_factory=Path)
    source_name: str = ""

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def resolve(cls, name_or_path: str) -> EvalConfig:
        """Load by named lookup or filesystem path.

        - If ``name_or_path`` contains ``/`` or ends in ``.yaml`` /
          ``.yml``: treat as a literal filesystem path.
        - Otherwise: resolve as
          ``scripts/eval-configs/<name_or_path>.yaml`` (relative to repo
          root).

        Raises ``FileNotFoundError`` with the requested name + search
        path when the resolved file doesn't exist.
        """
        is_path = (
            "/" in name_or_path
            or name_or_path.endswith(".yaml")
            or name_or_path.endswith(".yml")
        )
        if is_path:
            path = Path(name_or_path).resolve()
            source_name = path.stem
        else:
            path = (CONFIG_DIR / f"{name_or_path}.yaml").resolve()
            source_name = name_or_path
        if not path.exists():
            raise FileNotFoundError(
                f"eval config '{name_or_path}' not found (searched: {path})"
            )
        return cls.from_yaml(path, source_name=source_name)

    @classmethod
    def from_yaml(cls, path: Path, *, source_name: str = "") -> EvalConfig:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"eval config {path} must be a YAML mapping; got {type(raw).__name__}"
            )
        return cls.from_dict(
            raw,
            source_path=path.resolve(),
            source_name=source_name or path.stem,
        )

    @classmethod
    def from_dict(
        cls,
        d: dict,
        *,
        source_path: Path = Path(),
        source_name: str = "",
    ) -> EvalConfig:
        # Validate required fields up front — clearer error than KeyError
        # from the dataclass constructor.
        required = ("description", "n_per_depth", "beam_width", "precision", "seed")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(
                f"eval config missing required fields {missing} "
                f"(source: {source_path or source_name})"
            )

        # n_per_depth: non-empty list of positive ints.
        npd_raw = d["n_per_depth"]
        if not isinstance(npd_raw, list) or not npd_raw:
            raise ValueError(f"n_per_depth must be a non-empty list; got {npd_raw!r}")
        npd: list[int] = []
        for i, v in enumerate(npd_raw):
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"n_per_depth[{i}] must be a positive int; got {v!r}")
            npd.append(int(v))

        # beam_width: positive int.
        bw = d["beam_width"]
        if not isinstance(bw, int) or bw <= 0:
            raise ValueError(f"beam_width must be a positive int; got {bw!r}")

        # precision.
        prec = d["precision"]
        if prec not in _VALID_PRECISIONS:
            raise ValueError(
                f"precision must be one of {_VALID_PRECISIONS}; got {prec!r}"
            )

        # seed.
        seed = d["seed"]
        if not isinstance(seed, int):
            raise ValueError(f"seed must be an int; got {seed!r}")

        # Optional fields with defaults.
        include_v_star = bool(d.get("include_v_star", False))
        n_per_v_star_layer = int(d.get("n_per_v_star_layer", 200))
        if n_per_v_star_layer <= 0:
            raise ValueError(
                f"n_per_v_star_layer must be positive; got {n_per_v_star_layer!r}"
            )
        render_html = bool(d.get("render_html", False))

        return cls(
            description=str(d["description"]),
            n_per_depth=tuple(npd),
            beam_width=int(bw),
            precision=str(prec),
            seed=int(seed),
            include_v_star=include_v_star,
            n_per_v_star_layer=n_per_v_star_layer,
            render_html=render_html,
            source_path=source_path,
            source_name=source_name,
        )

    # ------------------------------------------------------------------
    # Override application
    # ------------------------------------------------------------------

    def with_overrides(
        self,
        *,
        beam_width: int | None = None,
        precision: str | None = None,
        seed: int | None = None,
        include_v_star: bool | None = None,
        max_depth: int | None = None,
        render_html: bool | None = None,
    ) -> EvalConfig:
        """Return a new EvalConfig with the given fields overridden.

        ``max_depth`` slices ``n_per_depth[:max_depth]`` and errors if it
        exceeds the underlying schedule length.
        """
        npd = self.n_per_depth
        if max_depth is not None:
            if max_depth < 1:
                raise ValueError(f"max_depth must be >= 1; got {max_depth}")
            if max_depth > len(self.n_per_depth):
                raise ValueError(
                    f"max_depth={max_depth} exceeds config n_per_depth length "
                    f"{len(self.n_per_depth)}"
                )
            npd = self.n_per_depth[:max_depth]

        new_precision = precision if precision is not None else self.precision
        if new_precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"precision must be one of {_VALID_PRECISIONS}; got {new_precision!r}"
            )

        return EvalConfig(
            description=self.description,
            n_per_depth=npd,
            beam_width=beam_width if beam_width is not None else self.beam_width,
            precision=new_precision,
            seed=seed if seed is not None else self.seed,
            include_v_star=(
                include_v_star if include_v_star is not None else self.include_v_star
            ),
            n_per_v_star_layer=self.n_per_v_star_layer,
            render_html=(
                render_html if render_html is not None else self.render_html
            ),
            source_path=self.source_path,
            source_name=self.source_name,
        )

    @property
    def max_depth(self) -> int:
        return len(self.n_per_depth)

    @property
    def walk_depths(self) -> tuple[int, ...]:
        return tuple(range(1, len(self.n_per_depth) + 1))
