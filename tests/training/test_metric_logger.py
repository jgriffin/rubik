"""Tests for `rubik.training.metric_logger.MetricLogger`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rubik.training.metric_logger import MetricLogger


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_log_writes_one_jsonl_record_per_call(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricLogger(path) as logger:
        logger.log(step=1, loss=0.5)
        logger.log(step=2, loss=0.4)
        logger.log(step=3, loss=0.3)

    records = _read_records(path)
    assert len(records) == 3
    assert [r["step"] for r in records] == [1, 2, 3]
    assert [r["loss"] for r in records] == [0.5, 0.4, 0.3]


def test_log_includes_utc_iso_timestamp(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricLogger(path) as logger:
        logger.log(step=1)

    record = _read_records(path)[0]
    assert "ts" in record
    parsed = datetime.fromisoformat(record["ts"])
    # tzinfo present and UTC
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_log_serializes_arbitrary_field_types(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricLogger(path) as logger:
        logger.log(
            cell_idx=0,
            batch_size=1000,
            body_widths=[5000, 1000],
            n_residual_blocks=4,
            step_ms_median=12.34,
            samples=[1.1, 2.2, 3.3],
            extra={"nested": True, "depth": 14},
        )

    record = _read_records(path)[0]
    assert record["batch_size"] == 1000
    assert record["body_widths"] == [5000, 1000]
    assert record["step_ms_median"] == pytest.approx(12.34)
    assert record["samples"] == pytest.approx([1.1, 2.2, 3.3])
    assert record["extra"] == {"nested": True, "depth": 14}


def test_creates_parent_directory(tmp_path):
    path = tmp_path / "deep" / "nested" / "metrics.jsonl"
    assert not path.parent.exists()
    with MetricLogger(path) as logger:
        logger.log(step=1)
    assert path.exists()


def test_flush_per_record_survives_open_handle_reread(tmp_path):
    """Open the file with no .close() and read it back — flush ran per record."""
    path = tmp_path / "metrics.jsonl"
    logger = MetricLogger(path)
    logger.log(step=1, loss=0.9)
    # Read while logger is still open — flush guarantees this works.
    records = _read_records(path)
    assert len(records) == 1
    assert records[0]["loss"] == 0.9
    logger.close()


def test_wandb_passthrough_called_with_fields_only(tmp_path):
    path = tmp_path / "metrics.jsonl"
    fake_run = MagicMock()
    with MetricLogger(path, wandb_run=fake_run) as logger:
        logger.log(step=42, loss=0.123)

    fake_run.log.assert_called_once_with({"step": 42, "loss": 0.123})
    # On disk the record additionally carries `ts`.
    record = _read_records(path)[0]
    assert record["step"] == 42
    assert "ts" in record


def test_wandb_passthrough_not_called_when_none(tmp_path):
    """Default: no wandb_run, no .log() invocation, no wandb import attempted."""
    path = tmp_path / "metrics.jsonl"
    with MetricLogger(path) as logger:
        logger.log(step=1)
    # Just assert no crash; the absence of any wandb dependency is the contract.


def test_close_is_idempotent(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = MetricLogger(path)
    logger.log(step=1)
    logger.close()
    logger.close()  # second call is a no-op


def test_path_property_returns_jsonl_path(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricLogger(path) as logger:
        assert logger.path == path


def test_log_appends_to_existing_file(tmp_path):
    """Re-opening the same path appends rather than truncating."""
    path = tmp_path / "metrics.jsonl"
    with MetricLogger(path) as logger:
        logger.log(step=1)
    with MetricLogger(path) as logger:
        logger.log(step=2)
    records = _read_records(path)
    assert [r["step"] for r in records] == [1, 2]
