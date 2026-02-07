"""Markdown report writer.

[Assumption] The paper does not prescribe a report format. This is our
design choice to produce a human-readable summary of pipeline results.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from revolut_credit_risk import config

logger = logging.getLogger(__name__)


class ReportWriter:
    """Accumulates sections and writes ``outputs/pipeline_report.md``."""

    def __init__(self) -> None:
        self._sections: list[str] = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._sections.append(
            f"# Credit Risk Model Development Report\n_Generated: {ts}_\n"
        )

    def add_section(self, title: str, body: str) -> None:
        """Append a markdown section to the report."""
        self._sections.append(f"## {title}\n\n{body}\n")

    def write(self, path: Path | None = None) -> Path:
        """Write the accumulated report to disk and return the path."""
        path = path or config.REPORT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self._sections)
        path.write_text(content, encoding="utf-8")
        logger.info("Report written to %s (%d sections)", path, len(self._sections) - 1)
        return path
