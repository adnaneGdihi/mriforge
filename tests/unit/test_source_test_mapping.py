"""Meta-test: every substantial source module has at least one referencing test.

For every non-trivial module under src/spectramr/ (>=40 non-blank non-comment
lines, not __init__.py), asserts that SOME test file under tests/ references
the module stem — either by importing it or by being named test_<stem>.py.

Uses pytest.xfail (not hard fail) for unreferenced modules so the existing
coverage backlog does not break collection on day one.  As waves of new tests
land, modules flip from xfail to pass automatically.  Convert to a hard assert
once the backlog is cleared.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Compute paths robustly from __file__ so this file can live anywhere under
# tests/ without hard-coding depth assumptions.
_HERE = Path(__file__).resolve()
# Walk up until we find the directory that contains both src/ and tests/
_REPO = _HERE.parent
while _REPO != _REPO.parent:
    if (_REPO / "src").is_dir() and (_REPO / "tests").is_dir():
        break
    _REPO = _REPO.parent

SRC = _REPO / "src" / "spectramr"
TESTS = _REPO / "tests"


def _substantial_modules() -> list[Path]:
    """Return source modules with >= 40 non-blank, non-comment lines."""
    out: list[Path] = []
    for p in sorted(SRC.rglob("*.py")):
        if p.name == "__init__.py":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        loc = sum(
            1
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )
        if loc >= 40:
            out.append(p)
    return out


def _all_test_text() -> str:
    """Concatenate all test-file contents into one searchable string."""
    parts: list[str] = []
    for p in TESTS.rglob("test_*.py"):
        try:
            parts.append(p.read_text(encoding="utf-8"))
        except OSError:
            pass
    return "\n".join(parts)


@pytest.fixture(scope="module")
def test_corpus() -> str:
    return _all_test_text()


@pytest.mark.parametrize(
    "mod",
    _substantial_modules(),
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_module_is_referenced_by_some_test(mod: Path, test_corpus: str) -> None:
    """Assert that at least one test file mentions the module stem."""
    stem = mod.stem  # e.g. "fft_ops"
    # A reference counts when the stem appears as a whole word — this matches
    # both `import fft_ops`, `from ... import fft_ops`, and test files named
    # test_fft_ops.py (whose own content will include `fft_ops` by definition).
    referenced = re.search(rf"\b{re.escape(stem)}\b", test_corpus) is not None
    if not referenced:
        pytest.xfail(
            f"{mod.relative_to(SRC)} has no referencing test yet "
            "(coverage backlog — add tests under tests/ and this xfail becomes a pass)"
        )
    assert referenced
