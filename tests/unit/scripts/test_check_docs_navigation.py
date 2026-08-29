"""Planted violations for the docs-navigation gate's orphan predicate.

Non-negotiable 15: a gate is only a gate for the violation shape you have
watched it fail on. The orphan predicate has TWO shapes because Sphinx accepts
two spellings -- the reStructuredText ``:orphan:`` field and MyST's
``orphan: true`` front matter -- and it recognised only the first, so a
Markdown orphan read as a navigation failure. Both shapes are pinned here, in
both polarities, so widening one never silently drops the other.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_GATE = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_docs_navigation.py"


def _load():
    spec = importlib.util.spec_from_file_location("_check_docs_navigation", _GATE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load()


@pytest.mark.parametrize(
    ("name", "body", "expected"),
    [
        # -- opted out: the gate must NOT report these as orphans -------------
        ("rst_field.rst", ":orphan:\n\nTitle\n=====\n", True),
        ("md_front_matter.md", "---\norphan: true\n---\n\n# Title\n", True),
        ("md_front_matter_yes.md", "---\norphan: yes\n---\n\n# Title\n", True),
        ("md_front_matter_mixed.md", "---\nmyst:\n  x: 1\norphan: true\n---\n\n# T\n", True),
        # -- NOT opted out: the gate MUST still report these ------------------
        ("plain.rst", "Title\n=====\n\nBody.\n", False),
        ("plain.md", "# Title\n\nBody.\n", False),
        ("md_front_matter_false.md", "---\norphan: false\n---\n\n# Title\n", False),
        ("md_no_front_matter.md", "orphan: true\n\n# Title\n", False),
    ],
)
def test_orphan_marker_is_recognised_in_both_spellings(gate, tmp_path, name, body, expected):
    page = tmp_path / name
    page.write_text(body)
    assert gate.is_orphan_marked(page) is expected


def test_the_markdown_shape_is_the_one_that_regressed(gate, tmp_path):
    """The exact page that exposed the gap: docs/reference/workflow_backlog.md.

    Pinned separately from the table so a future edit to the parametrisation
    cannot quietly drop the case that motivated the change.
    """
    page = tmp_path / "workflow_backlog.md"
    page.write_text("---\norphan: true\n---\n\n# Workflow backlog\n\nBody.\n")
    assert gate.is_orphan_marked(page) is True
