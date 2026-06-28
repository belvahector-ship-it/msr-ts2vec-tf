"""
src/utils/logging_utils.py — Standardized Logging Setup
========================================================

Purpose
-------
Provide a single, consistent logging configuration used by every module
in the project. All experiment-relevant events are logged to:

  1. A rotating file handler in logs/ (INFO level and above)
  2. A stream handler to stdout (configurable level)

Design
------
- print() is never used for experiment output.
- Every module acquires its logger via: logger = get_logger(__name__)
- The root logger is configured once at application startup via
  setup_logging(). Subsequent calls are no-ops unless force=True.
- Log format: timestamp | level | module_name | message
  This format is parseable for later analysis of training runs.

Usage
-----
    from src.utils.logging_utils import get_logger, setup_logging

    # At application entry point (scripts/, experiment runner):
    setup_logging(log_dir=Paths.logs(), level="INFO")

    # In every module:
    logger = get_logger(__name__)
    logger.info("Data loaded: %d rows", len(df))
    logger.warning("Gap detected in %s: %d missing candles", tf, n_gaps)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_LOGGING_CONFIGURED: bool = False

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    log_dir: Path | None = None,
    level: str = "INFO",
    log_filename: str | None = None,
    force: bool = False,
) -> None:
    """
    Configure the root logging system for the project.

    Should be called exactly once at the start of every script or
    experiment runner. Subsequent calls are no-ops unless force=True.

    Parameters
    ----------
    log_dir : Path or None
        Directory for log files. If None, file logging is disabled and
        only stream logging is set up. Directory is created if absent.
    level : str
        Logging level name: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        Applied to both the root logger and the file handler.
    log_filename : str or None
        Name of the log file. If None, defaults to
        'run_{YYYYMMDD_HHMMSS}.log'.
    force : bool
        If True, reconfigure logging even if it was already set up.
        Use with caution; existing handlers will be removed first.
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED and not force:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to prevent duplicate output on re-configuration
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATE_FORMAT,
    )

    # Stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            log_filename = f"run_{timestamp}.log"

        log_path = log_dir / log_filename
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Log the log file path so it is self-referential
        root_logger.info("Logging to file: %s", log_path)

    # Suppress overly verbose third-party loggers
    for noisy_logger in ("urllib3", "ccxt", "asyncio", "matplotlib"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for a module.

    This is the standard way every source module acquires its logger.
    The returned logger inherits its level and handlers from the root
    logger configured by setup_logging().

    Parameters
    ----------
    name : str
        Logger name. Convention: always pass __name__ from the calling module.
        Example: logger = get_logger(__name__)

    Returns
    -------
    logging.Logger
        Named logger instance.

    Notes
    -----
    If setup_logging() has not been called, log output defaults to Python's
    basicConfig behavior (WARNING level, stderr only).
    """
    return logging.getLogger(name)
