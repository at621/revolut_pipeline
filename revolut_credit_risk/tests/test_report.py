"""Tests for the report writer."""
from __future__ import annotations

from revolut_credit_risk.report import ReportWriter


def test_report_writer(tmp_path):
    rw = ReportWriter()
    rw.add_section("Test Section", "Some content here.")
    rw.add_section("Another Section", "More content.")

    path = rw.write(tmp_path / "test_report.md")
    assert path.exists()

    content = path.read_text(encoding="utf-8")
    assert "Credit Risk Model Development Report" in content
    assert "Test Section" in content
    assert "Another Section" in content
    assert "Some content here." in content
