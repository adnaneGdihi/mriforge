"""
Phase 4 logging-hygiene fitness function.

Library code under ``src/spectramr/`` must emit through the ``logging``
module, not ``print()``. User-facing CLI tools and the package
entrypoint legitimately write to stdout; everywhere else, a ``print()``
defeats log routing (file sinks, structured fields, log levels) and
fights ``infrastructure.logging.banner`` / ``phase``.

Legitimate stdout subtrees (blanket-exempt):
  * ``spectramr/cli/`` — interactive CLI tools.
  * ``spectramr/tools/`` — diagnostic / inspection scripts.
  * ``spectramr/main.py`` — top-level entrypoint banner.

Everywhere else is **library code**: the gate fails on any ``print(``
call that is not in ``_known_violations.json["stray_prints"]``. The
debt list is the paydown target; the slow-lane test fails until it is
empty.

History: 2026-05-28 punch-list closure. Backlog item
``backlog_beautify_logging_and_reporting.md`` Phase 1.1 ("eliminate
stray prints") was uncovered with no enforcement; this gate locks the
policy in so a future contributor cannot reintroduce silent stdout.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "spectramr"
VIOLATIONS_FILE = Path(__file__).parent / "_known_violations.json"

# Blanket-exempt subtrees: legitimate stdout for human-facing tooling.
BLANKET_EXEMPT_PREFIXES: tuple[str, ...] = (
    "spectramr/cli/",
    "spectramr/tools/",
    "spectramr/main.py",
)


def _is_blanket_exempt(rel: str) -> bool:
    return any(rel.startswith(p) or rel == p.rstrip("/") for p in BLANKET_EXEMPT_PREFIXES)


def _is_main_guard(node: ast.If) -> bool:
    """True if ``node`` is an ``if __name__ == "__main__":`` guard.

    Prints inside a module's ``__main__`` block are script-entrypoint
    output (equivalent to a CLI tool), so they are exempt -- the gate
    only polices prints reachable through library imports.
    """
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    left = test.left
    if not (isinstance(left, ast.Name) and left.id == "__name__"):
        return False
    return any(
        isinstance(c, ast.Constant) and c.value == "__main__"
        for c in test.comparators
    )


class _PrintFinder(ast.NodeVisitor):
    """Collect line numbers of bare ``print(...)`` calls outside ``__main__``."""

    def __init__(self) -> None:
        self.lines: list[int] = []

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        # Skip the body of ``if __name__ == "__main__":`` -- script
        # entrypoint output is exempt. Still visit the ``orelse`` and
        # nested non-main code.
        if _is_main_guard(node):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Match the global builtin ``print`` only (not ``self.print``,
        # ``foo.print``, etc. -- those are method calls on objects).
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.lines.append(node.lineno)
        self.generic_visit(node)


def _scan_files_with_prints() -> dict[str, int]:
    """Return ``{relative_path: print_count}`` for every library file with prints."""
    found: dict[str, int] = {}
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(REPO_ROOT / "src"))
        if _is_blanket_exempt(rel):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        finder = _PrintFinder()
        finder.visit(tree)
        if finder.lines:
            found[rel] = len(finder.lines)
    return found


def _load_known_print_debt() -> dict[str, int]:
    """Return ``{relative_path: tolerated_print_count}`` from the debt baseline."""
    if not VIOLATIONS_FILE.exists():
        return {}
    data = json.loads(VIOLATIONS_FILE.read_text())
    return {v["file"]: int(v["count"]) for v in data.get("stray_prints", [])}


@pytest.mark.architecture
def test_no_new_stray_prints_outside_cli_tools_main() -> None:
    """Gate: fail on any new ``print(`` in library code.

    The fail message tells the user which file regressed and by how
    many prints. Two acceptable fixes:

      * Replace the ``print(`` with the file's module logger
        (``logging.getLogger(__name__).<level>(...)``).
      * If the print is genuinely user-facing CLI output, move the
        function to ``spectramr/cli/`` or ``spectramr/tools/``.

    Temporarily adding to the debt list is only OK when the migration
    is genuinely deferred -- the slow-lane test fails until the list is
    empty.
    """
    current = _scan_files_with_prints()
    debt = _load_known_print_debt()

    new_violations: list[str] = []
    increased: list[str] = []
    for rel, count in current.items():
        tolerated = debt.get(rel)
        if tolerated is None:
            new_violations.append(f"  {rel}  ({count} print call(s))")
        elif count > tolerated:
            increased.append(
                f"  {rel}  ({count} prints; tolerated {tolerated})"
            )

    if new_violations or increased:
        messages: list[str] = []
        if new_violations:
            messages.append(
                "NEW stray print() calls in library code "
                "(not in _known_violations.json[stray_prints]):\n"
                + "\n".join(new_violations)
            )
        if increased:
            messages.append(
                "EXISTING files now have MORE stray prints than the debt "
                "baseline tolerates:\n" + "\n".join(increased)
            )
        messages.append(
            "\nFix: replace print(...) with the file's module logger "
            "(``logging.getLogger(__name__).info/warning/error(...)``). "
            "If the print is genuinely user-facing CLI output, move the "
            "function to spectramr/cli/ or spectramr/tools/."
        )
        pytest.fail("\n\n".join(messages))


@pytest.mark.architecture
@pytest.mark.slow
def test_known_print_debt_is_being_paid_down() -> None:
    """Cleanup tracker: fails until ``stray_prints`` debt is empty.

    Run with ``pytest -m slow`` to assess paydown progress. Passes only
    when the debt list is empty AND no file in the list still emits a
    ``print(`` -- so removing a stale entry is detected.
    """
    current = _scan_files_with_prints()
    debt = _load_known_print_debt()

    still_in_debt = sorted(f for f in debt if f in current)
    stale_debt = sorted(f for f in debt if f not in current)
    messages: list[str] = []
    if stale_debt:
        messages.append(
            "Stale entries in _known_violations.json[stray_prints] "
            "(file no longer has print() -- safe to remove):\n"
            + "\n".join(f"  {f}" for f in stale_debt)
        )
    if still_in_debt:
        total = sum(debt[f] for f in still_in_debt)
        messages.append(
            f"{len(still_in_debt)} library files still emit print() "
            f"({total} total calls). Migrate each to the module logger:\n"
            + "\n".join(f"  {f}: {debt[f]}" for f in still_in_debt)
        )
    assert not (still_in_debt or stale_debt), "\n\n".join(messages)
