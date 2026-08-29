"""Ratchet test: no NEW stray ``print()`` calls in ``src/``.

Beautify backlog item 1.1
(``TODO/backlog_beautify_logging_and_reporting.md``) calls for every
non-CLI ``print(...)`` to route through ``logging.getLogger(__name__)``
so log filtering, JSON sinks, and the rich ConsoleFormatter all work
consistently. The full audit is large (~93 sites today); this test
locks in the count so the trend is monotonically downward — new
``print()`` introductions fail the suite, while existing ones can be
migrated piecemeal.

Allowed exemptions (these directories never route through the logger
by design):

- ``src/cli/`` and ``src/main.py`` — CLI entry points printing user-
  facing help text / tables / results.
- ``src/tools/`` — standalone CLI scripts.

To migrate a site:

1. Replace ``print(msg)`` with ``logger.info(msg)`` (or appropriate level).
2. Drop the file's count from ``BUDGET`` below by 1 (or remove the entry).
3. CI stays green.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

# Subtrees that legitimately use ``print()`` for user-facing output.
ALLOWED_DIRS: tuple[str, ...] = (
    "cli",
    "tools",
)

ALLOWED_FILES: tuple[str, ...] = (
    "main.py",  # CLI top-level entry; prints help/results
)

# Maximum permitted stray-print count. Today's baseline is 33
# (down from 78 after the ``src/models/analysis/analyze_gradient_logs.py``
# migration). The budget is the ceiling — drops are welcome, growths
# fail the test. When a site is migrated to ``logger.*``, decrement
# this budget so regressions in the same file fail loudly.
BUDGET: int = 33


_PRINT_RE = re.compile(r"^[ \t]+print\(")


def _is_excluded(rel_path: str) -> bool:
    """Return True if the file is under an allowed exemption."""
    if rel_path in ALLOWED_FILES:
        return True
    for allowed_dir in ALLOWED_DIRS:
        if rel_path == allowed_dir or rel_path.startswith(allowed_dir + "/"):
            return True
    return False


def _count_stray_prints() -> int:
    """Count indented ``print(...)`` calls under ``src/`` minus exemptions."""
    total = 0
    for py_file in SRC.rglob("*.py"):
        rel = str(py_file.relative_to(SRC))
        if _is_excluded(rel):
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if _PRINT_RE.match(line):
                total += 1
    return total


def test_stray_print_count_does_not_grow() -> None:
    """Stray ``print()`` calls in non-CLI ``src/`` code must not increase.

    If this fails: either you added a new ``print()`` (replace with
    ``logger.info(...)``) or you migrated some and need to drop ``BUDGET``.
    """
    actual = _count_stray_prints()
    assert actual <= BUDGET, (
        f"Stray print() count grew from {BUDGET} to {actual}. "
        f"New print() calls must route through logging.getLogger(__name__) "
        f"instead — see TODO/backlog_beautify_logging_and_reporting.md item 1.1."
    )


def test_budget_matches_reality_within_tolerance() -> None:
    """If migrations land and the count drops, force the BUDGET to follow.

    Catches stale ratchet entries — a contributor who migrates 10 prints
    but forgets to drop ``BUDGET`` shouldn't accidentally widen the
    permitted regression window. We allow some slack (5) because tiny
    drifts shouldn't require this test to be edited every PR.
    """
    actual = _count_stray_prints()
    slack = 5
    assert actual >= BUDGET - slack, (
        f"Stray print() count dropped to {actual} but BUDGET is still {BUDGET}. "
        f"Tighten the ratchet: set BUDGET = {actual} in this file."
    )
