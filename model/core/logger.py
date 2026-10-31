"""Consistent logging setup for every pipeline stage.

The pipeline is long-running and mostly unattended, so each stage logs to the
console *and* to a per-stage file. ``configure_logging`` is idempotent: calling
it twice in the same interpreter replaces the handlers it created instead of
duplicating every line.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from model.core.paths import resolve_path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Marker attribute used to recognise (and later remove) handlers we own.
_HANDLER_MARKER = "_recommender_handler"


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """Configure the root logger for a pipeline stage.

    Args:
        level: Standard logging level name (``"INFO"``, ``"DEBUG"``, ...).
        log_file: Optional file that receives the same records. Parent folders
            are created automatically.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove handlers from a previous call so repeated stage runs do not
    # duplicate every log line.
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    setattr(console, _HANDLER_MARKER, True)
    root.addHandler(console)

    if log_file is not None:
        target = resolve_path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, True)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper that documents intent)."""
    return logging.getLogger(name)
