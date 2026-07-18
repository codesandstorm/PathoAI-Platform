"""
pathoai/core/logger.py
======================
Structured logging factory for PathoAI-Platform.

Provides a consistent, structured logging interface across all engines.
Features:
- Structured log records with experiment context injection
- Simultaneous console + rotating file output
- JSON-structured log format for automated parsing
- Per-experiment log directory with timestamp
- Module-level logger acquisition via get_logger(__name__)

Usage:
    from pathoai.core.logger import get_logger
    logger = get_logger(__name__)

    # Basic logging
    logger.info("Stage complete")

    # Structured logging with context dict
    logger.info("Patch extracted", extra={"slide_id": sid, "n_patches": n})

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# STRUCTURED JSON FORMATTER
# ---------------------------------------------------------------------------

class StructuredJsonFormatter(logging.Formatter):
    """Log formatter that outputs JSON Lines format for machine parsing.

    Each log record is emitted as a single JSON object on one line.
    Compatible with log aggregation systems (Elasticsearch, Splunk, etc.)

    Format:
        {"timestamp": "...", "level": "INFO", "logger": "pathoai.wsi", "message": "...", ...extra}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject any extra fields passed via the `extra` parameter
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            }:
                try:
                    json.dumps(value)  # Only include JSON-serializable values
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


# ---------------------------------------------------------------------------
# HUMAN-READABLE CONSOLE FORMATTER
# ---------------------------------------------------------------------------

class ConsoleFormatter(logging.Formatter):
    """Colored, human-readable log formatter for console output.

    Format:
        2026-07-18T17:09:45+05:30 [INFO ] [pathoai.wsi     ] Message
    """

    # ANSI color codes for different log levels
    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",    # Cyan
        "INFO":     "\033[32m",    # Green
        "WARNING":  "\033[33m",    # Yellow
        "ERROR":    "\033[31m",    # Red
        "CRITICAL": "\033[35m",    # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        level_str = f"{record.levelname:<8}"  # Pad to 8 chars
        logger_str = f"{record.name:<25}"     # Pad to 25 chars

        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).astimezone().isoformat(timespec="seconds")

        base = (
            f"{timestamp} "
            f"{color}[{level_str}]{self.RESET} "
            f"[{logger_str}] "
            f"{record.getMessage()}"
        )

        # Append extra context fields if present (non-standard fields)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            }
        }
        if extras:
            extras_str = " | ".join(f"{k}={v}" for k, v in extras.items())
            base = f"{base} [{extras_str}]"

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


# ---------------------------------------------------------------------------
# LOGGER REGISTRY — PREVENTS DUPLICATE HANDLERS
# ---------------------------------------------------------------------------

_initialized_loggers: Dict[str, logging.Logger] = {}
_root_log_dir: Optional[Path] = None
_experiment_id: Optional[str] = None


def configure_logging(
    log_dir: Optional[Path] = None,
    experiment_id: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
    structured_file: bool = True,
    human_file: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB per file
    backup_count: int = 5,
) -> None:
    """Configure the PathoAI logging system.

    Must be called once at application startup before any loggers are created.
    Subsequent calls to get_logger() will use this configuration.

    Parameters
    ----------
    log_dir : Path, optional
        Directory to write log files. If None, file logging is disabled.
    experiment_id : str, optional
        Experiment identifier injected into every structured log record.
        If None, generates a timestamp-based ID.
    level : str
        Minimum log level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        Default is "INFO".
    console : bool
        Whether to output logs to console (stdout). Default True.
    structured_file : bool
        Whether to write JSON-structured logs to file. Default True.
    human_file : bool
        Whether to write human-readable logs to file. Default True.
    max_bytes : int
        Maximum size of each log file before rotation. Default 50 MB.
    backup_count : int
        Number of rotated backup files to retain. Default 5.
    """
    global _root_log_dir, _experiment_id

    _root_log_dir = log_dir
    _experiment_id = experiment_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure the root "pathoai" logger
    root_logger = logging.getLogger("pathoai")
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()  # Remove any existing handlers

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(ConsoleFormatter())
        root_logger.addHandler(console_handler)

    # File handlers
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if human_file:
            human_path = log_dir / f"{_experiment_id}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                human_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(ConsoleFormatter())
            root_logger.addHandler(file_handler)

        if structured_file:
            json_path = log_dir / f"{_experiment_id}_structured.jsonl"
            json_handler = logging.handlers.RotatingFileHandler(
                json_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            json_handler.setLevel(numeric_level)
            json_handler.setFormatter(StructuredJsonFormatter())
            root_logger.addHandler(json_handler)

    # Prevent propagation to Python root logger (avoids duplicate messages)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    This is the primary entry point for acquiring loggers throughout
    the PathoAI codebase. Should be called at module level:

        logger = get_logger(__name__)

    Parameters
    ----------
    name : str
        Module name, typically __name__ (e.g., "pathoai.wsi.reader").

    Returns
    -------
    logging.Logger
        Configured logger instance. Handlers are inherited from the
        "pathoai" root logger configured via configure_logging().
    """
    if name in _initialized_loggers:
        return _initialized_loggers[name]

    logger = logging.getLogger(name)
    _initialized_loggers[name] = logger
    return logger


class ExperimentLogger:
    """Context manager that configures logging for a single experiment run.

    Ensures log files are created in the correct experiment directory
    and configuration is torn down cleanly after the experiment completes.

    Usage:
        with ExperimentLogger(log_dir=Path("logs"), experiment_id="exp_001"):
            run_pipeline(config)
    """

    def __init__(
        self,
        log_dir: Path,
        experiment_id: str,
        level: str = "INFO",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.experiment_id = experiment_id
        self.level = level

    def __enter__(self) -> "ExperimentLogger":
        configure_logging(
            log_dir=self.log_dir / self.experiment_id,
            experiment_id=self.experiment_id,
            level=self.level,
        )
        logger = get_logger("pathoai.core.logger")
        logger.info(
            "Experiment logging initialized",
            extra={
                "experiment_id": self.experiment_id,
                "log_dir": str(self.log_dir / self.experiment_id),
            },
        )
        return self

    def __exit__(self, *args: Any) -> None:
        logger = get_logger("pathoai.core.logger")
        logger.info("Experiment complete — closing log handlers")
        root_logger = logging.getLogger("pathoai")
        for handler in root_logger.handlers:
            handler.flush()
            handler.close()
        root_logger.handlers.clear()
        _initialized_loggers.clear()
