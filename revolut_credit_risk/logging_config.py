"""Logging setup — console (INFO) + file (DEBUG).

[Assumption] The paper does not discuss logging. This is our engineering
decision for pipeline visibility.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from revolut_credit_risk import config


def setup_logging() -> None:
    """Configure root logger with console and file handlers."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers (avoids duplicates on re-import)
    root.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler — INFO
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # File handler — DEBUG
    fh = logging.FileHandler(config.LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    # Quieten noisy third-party loggers
    for name in ("featuretools", "woodwork", "urllib3", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
