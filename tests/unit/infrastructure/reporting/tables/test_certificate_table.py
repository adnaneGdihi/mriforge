"""Unit tests for infrastructure/reporting/tables/certificate_table.py.

Covers: build_certificate_rows, render_markdown, render_latex,
        write_certificate_table.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spectramr.infrastructure.reporting.tables.certificate_table import (
    build_certificate_rows,
    render_latex,
    render_markdown,
    write_certificate_table,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD = {
    "upstream_certificates": {
        "A": {
            "certified": True,
            "bound_value": 0.8534,
            "target": 0.85,
            "n_calibration": 200,
            "passed": True,
        },
        "B": {
            "certified": False,
            "bound_value": 0.7200,
            "target": 0.75,
            "n_calibration": 150,
            "passed": False,
        },
        "C": {
            "certified": None,
            "bound_value": None,
            "target": None,
            "n_calibration": None,
            "passed": None,
        },
    }
}


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_certificate_rows_canary():
    """Rows are built from a valid badge payload."""
    rows = build_certificate_rows(SAMPLE_PAYLOAD)
    assert len(rows) == 3
    letters = {r["letter"] for r in rows}
    assert letters == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "letter,expect_certified,expect_passed",
    [
        ("A", "✓", "✓"),
        ("B", "✗", "✗"),
        ("C", "—", "—"),
    ],
)
def test_row_symbol_mapping(letter, expect_certified, expect_passed):
    """Certified / passed symbols match the spec."""
    rows = build_certificate_rows(SAMPLE_PAYLOAD)
    row = next(r for r in rows if r["letter"] == letter)
    assert row["certified"] == expect_certified
    assert row["passed"] == expect_passed


@pytest.mark.unit
@pytest.mark.parametrize("render_fn", [render_markdown, render_latex])
def test_render_empty_rows(render_fn):
    """Rendering with zero rows should not crash and return a non-empty string."""
    output = render_fn([])
    assert isinstance(output, str)
    assert len(output) > 0


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_markdown_contains_header_row():
    """Markdown output has a header row with expected column names."""
    rows = build_certificate_rows(SAMPLE_PAYLOAD)
    md = render_markdown(rows)
    assert "letter" in md
    assert "bound_value" in md


@pytest.mark.unit
def test_latex_begins_with_tabular():
    """LaTeX output starts a tabular environment."""
    rows = build_certificate_rows(SAMPLE_PAYLOAD)
    tex = render_latex(rows)
    assert r"\begin{tabular}" in tex
    assert r"\end{tabular}" in tex


@pytest.mark.unit
def test_write_certificate_table_creates_files():
    """write_certificate_table produces both .md and .tex files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_certificate_table(SAMPLE_PAYLOAD, tmpdir)
    assert "markdown" in paths
    assert "latex" in paths


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_rows_empty_payload():
    """Payload with no upstream_certificates returns empty list (not error)."""
    rows = build_certificate_rows({})
    assert rows == []
