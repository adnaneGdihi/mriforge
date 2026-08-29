"""Two test files that import under the same module name abort the whole session.

This has now bitten twice on the same basename. ``tests/smoke/test_strategies.py``
first collided with ``tests/unit/infrastructure/distributed/test_strategies.py``;
a package marker fixed that leaf, and the identical collision reappeared at
``tests/unit/data/collation/test_strategies.py``.

The failure is maximally hostile to diagnose:

* it is a **collection** error, so pytest reports ``Interrupted: N errors during
  collection`` and runs **zero** tests — a full-suite run looks catastrophically
  broken rather than mildly misconfigured;
* pytest stops at the first one, so fixing it reveals the next (there were three
  when this guard was written, and only one was visible);
* each file collects perfectly **in isolation**, so the obvious reproduction
  step reports success.

Mechanism: under pytest's default ``prepend`` import mode a test module's name
comes from walking UP from the file while each parent has ``__init__.py``. The
first directory without one becomes the sys.path root, and everything below it
imports under its **bare basename**. Two bare ``test_registry`` modules anywhere
in the tree therefore collide — and a marker on the leaf does not help if any
parent above it lacks one, which is exactly why
``tests/unit/models/diffusion/samplers/__init__.py`` did not prevent its own
collision.

This guard fails fast, names both paths, and says which directories need a
marker — instead of the session dying with ``import file mismatch``.
"""

from __future__ import annotations

import collections
import pathlib

TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _imports_bare(test_file: pathlib.Path) -> bool:
    """Does this file import under its bare basename rather than a dotted path?

    True when any directory from the file's parent up to (and including)
    ``tests/`` lacks ``__init__.py`` — that gap is where pytest roots sys.path.
    """
    directory = test_file.parent
    while True:
        if not (directory / "__init__.py").exists():
            return True
        if directory == TESTS_ROOT:
            return False
        directory = directory.parent


def _missing_markers(test_file: pathlib.Path) -> list[pathlib.Path]:
    """Directories from the file's parent up to ``tests/`` that lack a marker."""
    gaps: list[pathlib.Path] = []
    directory = test_file.parent
    while True:
        if not (directory / "__init__.py").exists():
            gaps.append(directory)
        if directory == TESTS_ROOT:
            break
        directory = directory.parent
    return gaps


def _bare_named_modules() -> dict[str, list[pathlib.Path]]:
    by_name: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for test_file in TESTS_ROOT.rglob("test_*.py"):
        if "__pycache__" in test_file.parts:
            continue
        if _imports_bare(test_file):
            by_name[test_file.name].append(test_file)
    return by_name


def test_no_two_test_modules_share_a_bare_basename():
    collisions = {
        name: sorted(paths) for name, paths in _bare_named_modules().items() if len(paths) > 1
    }
    if not collisions:
        return

    lines = ["Test modules colliding on a bare import name:", ""]
    for name, paths in sorted(collisions.items()):
        lines.append(f"  {name}")
        for path in paths:
            lines.append(f"      {path.relative_to(TESTS_ROOT.parent)}")
        # Qualifying ONE side is enough to break the tie. Recommend whichever
        # chain needs the fewest new markers -- in practice the tests/unit side,
        # whose parents already carry them, rather than tests/smoke/, which
        # carries none and would need the whole chain.
        # Tie-break toward tests/unit/: both existing fixes for this failure
        # live there (tests/unit/infrastructure/distributed/, then
        # tests/unit/data/collation/), and tests/smoke/ deliberately carries no
        # markers at all.
        cheapest = min(
            paths,
            key=lambda p: (len(_missing_markers(p)), "unit" not in p.parts),
        )
        gaps = _missing_markers(cheapest)
        rel = ", ".join(str(g.relative_to(TESTS_ROOT.parent)) + "/" for g in gaps)
        lines.append(
            f"      fix: add __init__.py to {rel} "
            f"({len(gaps)} marker(s); qualifying either side is enough)"
        )
        lines.append("")
    lines.append(
        "Left unfixed, pytest aborts the WHOLE session with 'import file "
        "mismatch' and runs zero tests. Each file still collects fine on its "
        "own, so this only reproduces on a multi-directory run."
    )
    raise AssertionError("\n".join(lines))


def test_the_guard_can_actually_detect_a_collision():
    """A guard that cannot fail is not a guard (pitfall #16).

    Pins the detector against the real tree rather than a fixture: this basename
    genuinely imports bare (``tests/smoke/`` has no marker), so the machinery is
    exercised on live data and this test starts failing if ``_imports_bare``
    is ever reduced to a constant ``False``.
    """
    bare = _bare_named_modules()
    assert "test_strategies.py" in bare, (
        "tests/smoke/test_strategies.py should still import under its bare "
        "basename; if tests/smoke/ gained an __init__.py, repoint this probe."
    )
    assert any(p.parts[-2] == "smoke" for p in bare["test_strategies.py"])
