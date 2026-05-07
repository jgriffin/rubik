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

- ``event="step"``  → ``train/<field>``
- ``event="eval"``  → ``value/<field>`` (forward-pass value-net eval —
  records emitted by ``value_eval``; this is the "value" panel group,
  separate from beam-search capability evals)
- ``event="beam_eval_walk"`` → ``beam_walk/<field>``
- ``event="beam_eval_v_star"`` → ``beam_v_star/<field>``
- ``event="checkpoint"`` → ``checkpoint/<field>``
- ``event="run_start"`` / ``event="run_end"`` → ``run/<field>``
- No ``event`` → fields pass through unchanged

Within each eval-shaped record, the same per-depth handling applies:
nested dicts get flattened slash-separated; per-depth scalar suffixes
``_d<N>`` get regex-regrouped to ``<base>/d<N>`` paths.

**Zero-padding of d-keys.** All ``d{N}`` segments inside the resulting
W&B keys are zero-padded to two digits (``d1`` → ``d01``, ``d14`` stays
``d14``). This is purely for W&B's natural-sort behavior in panel
legends — without padding the panels go ``d1, d10, d11, d12, d13, d14,
d2, d3, ..., d9``. The on-disk JSONL records keep the unpadded keys
(unchanged shape) so existing analyzers don't break.

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

# Match a `d{digits}` token that appears as a path segment (preceded by
# `/`) and is followed by `/` or end of string. Used to zero-pad the
# digit run for natural sort in W&B.
_D_KEY_PAD_RE = re.compile(r"(?<=/)d(\d+)(?=/|$)")


def _pad_d_keys(key: str) -> str:
    """Zero-pad ``d{N}`` path segments in ``key`` to two digits.

    ``"value/v_star_mae/d1"`` → ``"value/v_star_mae/d01"``
    ``"value/per_walk_depth/d1/pred_mean"`` → ``"value/per_walk_depth/d01/pred_mean"``
    ``"value/per_walk_depth/d14/pred_mean"`` → unchanged (already 2+ digits)
    """
    return _D_KEY_PAD_RE.sub(lambda m: f"d{int(m.group(1)):02d}", key)


def _pad_dict_d_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Apply ``_pad_d_keys`` to every key in ``d``."""
    return {_pad_d_keys(k): v for k, v in d.items()}


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


def _route_evalshape(
    fields: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Route an eval-shaped record under ``prefix/`` with depth handling.

    Used for ``event="eval"`` (→ ``value/``), ``event="beam_eval_walk"``
    (→ ``beam_walk/``), and ``event="beam_eval_v_star"`` (→
    ``beam_v_star/``). All three share the per-depth shape: nested dicts
    flattened, ``_d{N}`` suffixes regex-regrouped.
    """
    nested: dict[str, Any] = {}
    flat: dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, dict):
            nested[k] = v
        else:
            flat[k] = v
    flattened_nested = _flatten_dicts(nested)
    regrouped, leftover = _regroup_per_depth(flat, prefix=prefix)
    scalars = _namespace_under(leftover, prefix)
    nested_namespaced = _namespace_under(flattened_nested, prefix)
    return {**scalars, **nested_namespaced, **regrouped}


def _transform_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Route a JSONL-shaped record into the W&B namespace policy.

    See module docstring for the routing rules. The input is **not**
    mutated; a new dict is returned. ``d{N}`` path segments in the final
    keys are zero-padded to two digits for natural-sort in W&B.
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
        body = _namespace_under(fields, "train")
        return {**top, **_pad_dict_d_keys(body)}

    if event == "eval":
        # Forward-pass value-net eval (`value_eval` records).
        body = _route_evalshape(fields, prefix="value")
        return {**top, **_pad_dict_d_keys(body)}

    if event == "beam_eval_walk":
        # Beam-search capability eval on random-walk states.
        body = _route_evalshape(fields, prefix="beam_walk")
        return {**top, **_pad_dict_d_keys(body)}

    if event == "beam_eval_v_star":
        # Beam-search capability eval on V*-stratified states.
        body = _route_evalshape(fields, prefix="beam_v_star")
        return {**top, **_pad_dict_d_keys(body)}

    if event == "checkpoint":
        body = _namespace_under(fields, "checkpoint")
        return {**top, **_pad_dict_d_keys(body)}

    if event in ("run_start", "run_end"):
        # Nested values (e.g. body_widths list) get passed through as-is;
        # only top-level keys get the `run/` prefix. W&B handles list
        # values on a single key fine (stored as config-shaped JSON).
        body = _namespace_under(fields, "run")
        return {**top, **_pad_dict_d_keys(body)}

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
