"""Append-only JSONL metric logger with optional W&B passthrough.

The primary backend is line-delimited JSON written to disk: one record per
``log()`` call, each automatically tagged with a UTC ISO timestamp. A
secondary backend is an optional ``wandb_run``-shaped object — any object
exposing a ``log(dict)`` method. When provided, the same fields go to both
backends; when absent (the default), only JSONL is written.

The W&B passthrough is duck-typed by design: this module never imports
``wandb``. That keeps W&B opt-in, deferred, and trivial to swap (it lets
tests pass a ``Mock`` and the production code pass a real ``wandb.Run``).
The pattern matches the M5 Plan B decision (W&B for logging only — and
deferred until the tier 1 → tier 2 boundary when ~10 runs have accumulated;
JSONL is sufficient until then).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol


class _RunLike(Protocol):
    """Duck-typed shape for the optional W&B passthrough."""

    def log(self, fields: dict[str, Any]) -> Any: ...


class MetricLogger:
    """Append-only JSONL logger; optional W&B passthrough.

    Each ``log()`` call writes one JSON object to the JSONL file (parent
    directory created on demand) and, if a ``wandb_run`` was provided,
    forwards the same fields to ``wandb_run.log``. The on-disk record
    additionally carries a ``ts`` field (UTC ISO 8601, microsecond
    precision); the W&B forward does *not* include ``ts`` since W&B
    auto-stamps its own.

    Use as a context manager (``with MetricLogger(path) as log: ...``) for
    deterministic file-handle cleanup.
    """

    def __init__(
        self,
        jsonl_path: Path,
        *,
        wandb_run: _RunLike | None = None,
    ) -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = jsonl_path
        self._fp = jsonl_path.open("a")
        self._wandb_run = wandb_run

    @property
    def path(self) -> Path:
        """Resolved path of the underlying JSONL file."""
        return self._jsonl_path

    def log(self, **fields: Any) -> None:
        """Append one JSON record with the given fields plus a UTC timestamp.

        Flushes after every write so a mid-run crash leaves every prior
        record on disk. If a W&B run is attached, the same ``fields``
        (without the timestamp) are forwarded to ``wandb_run.log``.
        """
        record = {"ts": datetime.now(UTC).isoformat(), **fields}
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()
        if self._wandb_run is not None:
            self._wandb_run.log(dict(fields))

    def close(self) -> None:
        """Close the underlying file handle. Idempotent."""
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> MetricLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
