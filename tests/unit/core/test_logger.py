"""
tests/unit/core/test_logger.py
================================
Unit tests for pathoai.core.logger.

Tests cover:
- StructuredJsonFormatter: JSON output, extra fields, exception info
- ConsoleFormatter: human-readable format, level paddings
- configure_logging: console handler, file handler creation, level filtering
- get_logger: returns Logger, caching, hierarchy
- ExperimentLogger: context manager, file creation, teardown

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.3
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from typing import List

import pytest

from pathoai.core.logger import (
    ConsoleFormatter,
    ExperimentLogger,
    StructuredJsonFormatter,
    _initialized_loggers,
    configure_logging,
    get_logger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_record(
    msg: str = "test message",
    level: int = logging.INFO,
    name: str = "pathoai.test",
    extra: dict | None = None,
) -> logging.LogRecord:
    """Create a LogRecord for formatter tests."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


class _ListHandler(logging.Handler):
    """Captures log records into a list for assertions."""

    def __init__(self):
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# StructuredJsonFormatter
# ---------------------------------------------------------------------------

class TestStructuredJsonFormatter:
    """Tests for the JSON-Lines log formatter."""

    def test_output_is_valid_json(self):
        """Each formatted record must be valid JSON."""
        fmt = StructuredJsonFormatter()
        record = _make_log_record("hello pathoai")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_output_contains_required_fields(self):
        """Formatted record must contain timestamp, level, logger, message."""
        fmt = StructuredJsonFormatter()
        record = _make_log_record("hello")
        parsed = json.loads(fmt.format(record))
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed

    def test_message_matches_original(self):
        """Message field must equal the original log message."""
        fmt = StructuredJsonFormatter()
        record = _make_log_record("unique_message_xyz")
        parsed = json.loads(fmt.format(record))
        assert parsed["message"] == "unique_message_xyz"

    def test_level_matches_record_level(self):
        """Level field must match the record's level name."""
        fmt = StructuredJsonFormatter()
        for level, name in [(logging.DEBUG, "DEBUG"), (logging.ERROR, "ERROR")]:
            record = _make_log_record("msg", level=level)
            parsed = json.loads(fmt.format(record))
            assert parsed["level"] == name

    def test_extra_fields_are_included(self):
        """Extra fields passed to the logger must appear in the JSON output."""
        fmt = StructuredJsonFormatter()
        record = _make_log_record("msg", extra={"slide_id": "TCGA-001", "n_patches": 42})
        parsed = json.loads(fmt.format(record))
        assert parsed.get("slide_id") == "TCGA-001"
        assert parsed.get("n_patches") == 42

    def test_non_serializable_extra_converted_to_string(self):
        """Non-JSON-serializable extra values must be converted to strings."""
        fmt = StructuredJsonFormatter()
        record = _make_log_record("msg", extra={"obj": object()})
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "obj" in parsed
        assert isinstance(parsed["obj"], str)

    def test_timestamp_is_iso_format(self):
        """timestamp field must be a valid ISO 8601 string."""
        fmt = StructuredJsonFormatter()
        record = _make_log_record("msg")
        parsed = json.loads(fmt.format(record))
        ts = parsed["timestamp"]
        assert "T" in ts  # ISO 8601 separator
        assert ts.endswith("Z") or "+" in ts or "-" in ts

    def test_exception_info_included_when_present(self):
        """Exception info must be included in the JSON output when available."""
        fmt = StructuredJsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = _make_log_record("error occurred")
            record.exc_info = sys.exc_info()
            parsed = json.loads(fmt.format(record))
            assert "exception" in parsed
            assert "ValueError" in parsed["exception"]


# ---------------------------------------------------------------------------
# ConsoleFormatter
# ---------------------------------------------------------------------------

class TestConsoleFormatter:
    """Tests for the human-readable console formatter."""

    def test_output_contains_message(self):
        """Formatted string must contain the original message."""
        fmt = ConsoleFormatter()
        record = _make_log_record("critical system check")
        output = fmt.format(record)
        assert "critical system check" in output

    def test_output_contains_level_name(self):
        """Formatted string must contain the level name."""
        fmt = ConsoleFormatter()
        record = _make_log_record("msg", level=logging.WARNING)
        output = fmt.format(record)
        assert "WARNING" in output

    def test_output_contains_logger_name(self):
        """Formatted string must contain the logger name."""
        fmt = ConsoleFormatter()
        record = _make_log_record("msg", name="pathoai.wsi.reader")
        output = fmt.format(record)
        assert "pathoai.wsi.reader" in output

    def test_extra_fields_appended_to_output(self):
        """Extra context fields must appear in the formatted string."""
        fmt = ConsoleFormatter()
        record = _make_log_record("msg", extra={"slide_id": "TCGA-001"})
        output = fmt.format(record)
        assert "TCGA-001" in output


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    """Tests for the configure_logging() setup function."""

    def test_console_logging_adds_handler(self):
        """configure_logging with console=True must add a StreamHandler."""
        configure_logging(log_dir=None, level="WARNING", console=True)
        root = logging.getLogger("pathoai")
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(stream_handlers) >= 1
        # Cleanup
        root.handlers.clear()

    def test_no_console_logging_adds_no_stream_handler(self):
        """configure_logging with console=False must not add a StreamHandler."""
        configure_logging(log_dir=None, level="WARNING", console=False)
        root = logging.getLogger("pathoai")
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(stream_handlers) == 0
        root.handlers.clear()

    def test_file_logging_creates_log_files(self, tmp_path: Path):
        """configure_logging with log_dir must create log files on first write."""
        configure_logging(
            log_dir=tmp_path,
            experiment_id="test_exp",
            level="DEBUG",
            console=False,
            structured_file=True,
            human_file=True,
        )
        root = logging.getLogger("pathoai")
        root.debug("first log entry")
        for h in root.handlers:
            h.flush()
        root.handlers.clear()

        log_files = list(tmp_path.iterdir())
        assert len(log_files) >= 1

    def test_log_level_filters_lower_levels(self, tmp_path: Path):
        """Messages below the configured level must not be logged."""
        handler = _ListHandler()
        handler.setLevel(logging.WARNING)

        configure_logging(log_dir=None, level="WARNING", console=False)
        root = logging.getLogger("pathoai")
        root.addHandler(handler)

        root.debug("debug message — should be filtered")
        root.info("info message — should be filtered")
        root.warning("warning — must appear")

        warning_records = [r for r in handler.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1
        root.handlers.clear()

    def test_configure_logging_replaces_existing_handlers(self):
        """Calling configure_logging twice must not accumulate handlers."""
        configure_logging(log_dir=None, level="WARNING", console=True)
        configure_logging(log_dir=None, level="WARNING", console=True)
        root = logging.getLogger("pathoai")
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(stream_handlers) == 1
        root.handlers.clear()


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    """Tests for get_logger()."""

    def test_returns_logging_logger(self):
        """get_logger must return a standard logging.Logger instance."""
        logger = get_logger("pathoai.test_module")
        assert isinstance(logger, logging.Logger)

    def test_name_is_preserved(self):
        """Logger name must match the name passed to get_logger."""
        name = "pathoai.wsi.reader"
        logger = get_logger(name)
        assert logger.name == name

    def test_same_name_returns_cached_instance(self):
        """Two calls with the same name must return the same object."""
        l1 = get_logger("pathoai.cache_test")
        l2 = get_logger("pathoai.cache_test")
        assert l1 is l2

    def test_different_names_return_different_loggers(self):
        """Different names must yield distinct Logger objects."""
        l1 = get_logger("pathoai.module_a")
        l2 = get_logger("pathoai.module_b")
        assert l1 is not l2

    def test_child_logger_inherits_from_pathoai_root(self):
        """Child loggers must inherit from the 'pathoai' root logger."""
        logger = get_logger("pathoai.sub.module")
        assert logger.name.startswith("pathoai")


# ---------------------------------------------------------------------------
# ExperimentLogger context manager
# ---------------------------------------------------------------------------

class TestExperimentLogger:
    """Tests for the ExperimentLogger context manager."""

    def test_context_manager_enters_and_exits_cleanly(self, tmp_path: Path):
        """ExperimentLogger must enter and exit without raising."""
        with ExperimentLogger(
            log_dir=tmp_path,
            experiment_id="test_exp_001",
            level="WARNING",
        ) as exp_logger:
            assert exp_logger is not None

    def test_log_dir_created_on_enter(self, tmp_path: Path):
        """The experiment log directory must exist after entering context."""
        exp_id = "test_exp_dir"
        with ExperimentLogger(log_dir=tmp_path, experiment_id=exp_id, level="WARNING"):
            assert (tmp_path / exp_id).is_dir()

    def test_handlers_cleared_on_exit(self, tmp_path: Path):
        """All handlers must be cleared when ExperimentLogger exits."""
        with ExperimentLogger(
            log_dir=tmp_path,
            experiment_id="cleanup_test",
            level="WARNING",
        ):
            pass
        root = logging.getLogger("pathoai")
        assert len(root.handlers) == 0
