"""Every ``pkgutil.walk_packages`` in ``src/`` must be handed an ``onerror``.

``walk_packages`` recurses into a sub-package by **importing** it, and that
import runs inside pkgutil -- not in the caller's loop body. So the ``try /
except ImportError`` wrapped around ``importlib.import_module(module_name)``
below the loop, which every discovery walk in this repo has, cannot see it.
With the default ``onerror=None`` pkgutil discards the exception and abandons
the whole sub-tree. Verbatim, from CPython's ``pkgutil.walk_packages``::

    except ImportError:
        if onerror is not None:
            onerror(info.name)

Measured 2026-08-28 against ``spectramr.core.metrics``: planting ``raise
ImportError`` in the ``connectivity`` sub-package took the registry from 211
metrics to 210, with no error, no warning, and exit 0. Nothing downstream could
distinguish that from an empty sub-package.

This is a fitness function rather than three separate per-module tests because
the defect is a *shape*, not a site: each walk is written independently, each
one's own tests pass, and the next one added will be written the same way. Two
of the three walks in the tree were flat when this landed, which made their hole
latent rather than absent -- adding one nested package would have re-opened it
silently. CLAUDE.md non-negotiable 15: a detector defect outranks an equal
code defect, because it multiplies every future finding.

There is no allowlist and no baseline on purpose. A walk that genuinely wants
to tolerate a broken sub-package can pass ``onerror`` a callback that says so;
what it may not do is leave the decision to a default that reports nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "spectramr"

pytestmark = pytest.mark.architecture


def _is_walk_packages(node: ast.Call) -> bool:
    """``pkgutil.walk_packages(...)`` or a bare ``walk_packages(...)``."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "walk_packages"
    return isinstance(func, ast.Name) and func.id == "walk_packages"


def find_walks_without_onerror(root: Path) -> list[str]:
    """Return ``path:line`` for every walk_packages call missing ``onerror``.

    AST, not grep: a call-site regex under-counts exactly where the code is
    tidiest, because a multi-line call puts the keyword on a different line
    from the callee.
    """
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover - a broken file is a louder failure
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_walk_packages(node):
                continue
            names = {kw.arg for kw in node.keywords}
            # ``**kwargs`` gives arg=None; treat an opaque splat as compliant
            # rather than guessing at its contents.
            if "onerror" in names or None in names:
                continue
            # Relative to REPO_ROOT when the scan runs over src/, but the
            # plant test below scans a tmp_path -- and a detector that cannot
            # be pointed at a planted tree is a detector nobody can falsify.
            try:
                shown = path.relative_to(REPO_ROOT)
            except ValueError:
                shown = path.relative_to(root)
            offenders.append(f"{shown}:{node.lineno}")
    return offenders


def test_no_discovery_walk_swallows_a_subpackage_failure() -> None:
    offenders = find_walks_without_onerror(SRC_ROOT)
    assert not offenders, (
        "pkgutil.walk_packages called without onerror=; a sub-package that fails "
        "to import is silently discarded along with everything beneath it, and "
        "the caller's own except-ImportError never runs because pkgutil does that "
        "import itself:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_sees_a_planted_violation(tmp_path: Path) -> None:
    """The detector is only a detector for a shape it has been watched failing on.

    Both spellings, and both the compliant and non-compliant form of each, so a
    scanner that matched on the callee text alone (and therefore never on the
    keyword) could not score this green.
    """
    (tmp_path / "attr_bad.py").write_text(
        "import pkgutil\nfor x in pkgutil.walk_packages(P, 'p.'):\n    pass\n"
    )
    (tmp_path / "name_bad.py").write_text(
        "from pkgutil import walk_packages\n"
        "for x in walk_packages(\n    P,\n    'p.',\n):\n    pass\n"
    )
    (tmp_path / "attr_ok.py").write_text(
        "import pkgutil\nfor x in pkgutil.walk_packages(P, 'p.', onerror=boom):\n    pass\n"
    )
    (tmp_path / "name_ok.py").write_text(
        "from pkgutil import walk_packages\n"
        "for x in walk_packages(\n    P,\n    'p.',\n    onerror=boom,\n):\n    pass\n"
    )

    found = {Path(entry.split(":")[0]).name for entry in find_walks_without_onerror(tmp_path)}
    assert found == {"attr_bad.py", "name_bad.py"}, (
        f"the scan reported {sorted(found)}; it must flag exactly the two calls "
        "with no onerror= and neither of the two that have it"
    )
