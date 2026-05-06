"""Weights & Biases sink for the MetricLogger duck-typed passthrough.

`MetricLogger` accepts any object that implements ``log(dict) -> Any``
(see `_RunLike` Protocol in ``metric_logger.py``). This module provides a
concrete adapter, ``WandbAdapter``, that wraps a ``wandb.Run`` and
satisfies that shape — but transforms the fields on the way through so
that what lands in W&B is **namespaced and chart-friendly**, not the raw
JSONL field shape.

Why the transform lives here, not in MetricLogger:

- MetricLogger stays a *pure* JSONL writer with a duck-typed forwarding
  hook — no awareness of W&B-specific concerns (panel grouping,
  flattened nested dicts, regex-regrouped per-depth fields).
- W&B sinks differ by experiment: a future sink for a different
  experiment may want different namespaces. Keeping the policy in the
  adapter (not the logger) makes that swap trivial.
- Tests for the adapter can run **without** wandb installed — we accept
  any object exposing ``log(dict)`` (including a ``MagicMock``). The
  ``wandb.Run`` type only appears under ``TYPE_CHECKING``.

Field-routing policy (per ``event``):

- ``event="step"``  → ``train/<field>`` (e.g. ``train/loss``,
  ``train/step_seconds``, ``train/k_max``)
- ``event="eval"``  → ``eval/<field>`` for scalars; nested
  ``per_depth_mae`` flattened slash-separated; ``solve_rate_d<N>`` and
  ``avg_solve_len_d<N>`` regex-regrouped to ``eval/solve_rate/d<N>`` and
  ``eval/avg_solve_len/d<N>`` so W&B groups them as multi-line panels
- ``event="checkpoint"`` → ``checkpoint/<field>``
- ``event="run_start"`` / ``event="run_end"`` → ``run/<field>``
- No ``event`` → fields pass through unchanged

``step`` and ``event`` are **always** kept top-level (never namespaced):
W&B uses ``step`` as the x-axis for line plots, and ``event`` is useful
as a filterable scalar in panel queries.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb  # noqa: F401  (only used for type hint below)


_PER_DEPTH_RE = re.compile(r"^(.+)_d(\d+)$")


def _flatten_dicts(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested dicts using slash-separated keys.

    ``{"per_depth_mae": {"1": 0.99, "2": 0.67}}`` becomes
    ``{"per_depth_mae/1": 0.99, "per_depth_mae/2": 0.67}`` (with optional
    ``prefix`` prepended via slash).
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dicts(v, key))
        else:
            out[key] = v
    return out


def _regroup_per_depth(
    d: dict[str, Any], prefix: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``d`` into (regrouped, leftover).

    Keys matching ``r"^(.+)_d(\\d+)$"`` move to ``regrouped`` under
    ``f"{prefix}/<base>/d<N>"``; non-matching keys stay in ``leftover``
    untouched.
    """
    regrouped: dict[str, Any] = {}
    leftover: dict[str, Any] = {}
    for k, v in d.items():
        m = _PER_DEPTH_RE.match(k)
        if m:
            base, n = m.group(1), m.group(2)
            regrouped[f"{prefix}/{base}/d{n}"] = v
        else:
            leftover[k] = v
    return regrouped, leftover


def _namespace_under(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Prepend ``f"{prefix}/"`` to every key in ``d``."""
    return {f"{prefix}/{k}": v for k, v in d.items()}


def _transform_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Route a JSONL-shaped record into the W&B namespace policy.

    See module docstring for the routing rules. The input is **not**
    mutated; a new dict is returned.
    """
    fields = dict(fields)  # defensive copy
    event = fields.pop("event", None)

    # `step` is always top-level (W&B x-axis); pull it aside.
    top: dict[str, Any] = {}
    if "step" in fields:
        top["step"] = fields.pop("step")
    if event is not None:
        top["event"] = event

    if event == "step":
        return {**top, **_namespace_under(fields, "train")}

    if event == "eval":
        # 1) per-depth nested dicts get flattened slash-separated
        nested: dict[str, Any] = {}
        flat: dict[str, Any] = {}
        for k, v in fields.items():
            if isinstance(v, dict):
                nested[k] = v
            else:
                flat[k] = v
        flattened_nested = _flatten_dicts(nested)
        # 2) per-depth scalar suffix `_d<N>` patterns regrouped
        regrouped, leftover = _regroup_per_depth(flat, prefix="eval")
        # 3) Remaining flat scalars (val_mae, macro_mae, pred_mean, ...)
        scalars = _namespace_under(leftover, "eval")
        # Nested-flattened keys also belong under eval/ namespace.
        nested_namespaced = _namespace_under(flattened_nested, "eval")
        return {**top, **scalars, **nested_namespaced, **regrouped}

    if event == "checkpoint":
        return {**top, **_namespace_under(fields, "checkpoint")}

    if event in ("run_start", "run_end"):
        # Nested values (e.g. body_widths list) get passed through as-is;
        # only top-level keys get the `run/` prefix. W&B handles list
        # values on a single key fine (stored as config-shaped JSON).
        return {**top, **_namespace_under(fields, "run")}

    # No event, or unknown event: pass through unchanged.
    if event is not None:
        # Unknown event — restore it and pass everything through.
        passthrough: dict[str, Any] = {"event": event, **fields}
        if "step" in top:
            passthrough["step"] = top["step"]
        return passthrough
    return fields


class WandbAdapter:
    """Wrap a ``wandb.Run`` so it satisfies MetricLogger's ``_RunLike`` shape.

    Every ``log(fields)`` call routes ``fields`` through
    :func:`_transform_fields` (event-driven namespacing + flattening +
    regrouping) and forwards the result to ``run.log(...)``.

    The ``run`` parameter is duck-typed (any object exposing ``log(dict)``)
    so this class is testable without wandb installed.
    """

    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, fields: dict[str, Any]) -> Any:
        return self._run.log(_transform_fields(fields))
