"""Collection-time invariants for the test suite itself.

Both invariants here are *session-fatal* when broken: pytest reports them as
collection errors and aborts with "Interrupted: N errors during collection", so a
single stray file takes all ~45k tests down rather than failing in isolation. They
are cheap to assert and expensive to discover on a 48-hour cluster job.

1. Every ``@pytest.mark.<name>`` is registered. Under ``--strict-markers`` an
   unregistered mark is an error, not a warning.
2. No two test files collapse onto the same pytest module name. With the default
   ``prepend`` import mode the module name is the dotted path back to the first
   ancestor lacking ``__init__.py``, so two same-named files in two non-package
   directories both import as the bare basename and the second one dies with
   "import file mismatch".
"""

from __future__ import annotations

import re
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_ROOT = _REPO_ROOT / "tests"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Marks pytest ships itself; they are never listed in `markers`.
_BUILTIN_MARKS = frozenset(
    {
        "filterwarnings",
        "parametrize",
        "skip",
        "skipif",
        "tryfirst",
        "trylast",
        "usefixtures",
        "xfail",
    }
)

# Marks a plugin or conftest registers at runtime. Each needs a live registrar --
# hand-declaring one in `markers` would silence the error while leaving the
# behaviour dead (the reason pytest-timeout is a dependency rather than a line in
# the markers list).
_RUNTIME_REGISTERED_MARKS = {
    "timeout": "pytest-timeout (declared in the [test] extra)",
    "asyncio": "conftest.py::pytest_configure",
}

_MARK_RE = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")


def _declared_marks() -> set[str]:
    """Mark names registered in ``[tool.pytest.ini_options] markers``."""
    ini = tomllib.loads(_PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]
    return {line.split(":", 1)[0].split("(", 1)[0].strip() for line in ini["markers"]}


def _test_files() -> list[Path]:
    return [p for p in _TESTS_ROOT.rglob("test_*.py") if "__pycache__" not in p.parts]


def _pytest_module_name(path: Path) -> str:
    """Reproduce pytest's ``prepend``-mode module name for ``path``."""
    parts = [path.stem]
    parent = path.parent
    while (parent / "__init__.py").is_file():
        parts.append(parent.name)
        parent = parent.parent
    return ".".join(reversed(parts))


def test_every_used_mark_is_registered() -> None:
    """An unregistered mark is a collection error under --strict-markers."""
    allowed = _declared_marks() | _BUILTIN_MARKS | set(_RUNTIME_REGISTERED_MARKS)

    offenders: dict[str, set[str]] = defaultdict(set)
    for path in _test_files():
        for name in _MARK_RE.findall(path.read_text(encoding="utf-8", errors="ignore")):
            if name not in allowed:
                offenders[name].add(str(path.relative_to(_REPO_ROOT)))

    assert not offenders, (
        "Unregistered pytest marks -- these abort collection for the WHOLE suite "
        "under --strict-markers. Add each to [tool.pytest.ini_options] markers in "
        "pyproject.toml (or install a plugin that registers it):\n"
        + "\n".join(
            f"  {name}: {sorted(files)[0]}"
            + (f" (+{len(files) - 1} more)" if len(files) > 1 else "")
            for name, files in sorted(offenders.items())
        )
    )


def test_no_two_test_files_share_a_pytest_module_name() -> None:
    """Duplicate module names abort collection with "import file mismatch"."""
    by_module: dict[str, list[Path]] = defaultdict(list)
    for path in _test_files():
        by_module[_pytest_module_name(path)].append(path)

    collisions = {name: paths for name, paths in by_module.items() if len(paths) > 1}

    assert not collisions, (
        "Test files collapse onto one pytest module name, which aborts collection "
        "with 'import file mismatch'. Add an __init__.py to the directories below "
        "so the module name carries its package path, or rename one file:\n"
        + "\n".join(
            f"  {name}: " + ", ".join(str(p.relative_to(_REPO_ROOT)) for p in sorted(paths))
            for name, paths in sorted(collisions.items())
        )
    )


@pytest.mark.parametrize(
    "package_dir",
    [
        "tests/unit/builders",
        "tests/unit/infrastructure/training/builders",
    ],
)
def test_colliding_builder_dirs_stay_packages(package_dir: str) -> None:
    """Regression guard: both builder dirs hold test_loss_builder.py.

    They only coexist because each is a package, so the module names differ. Losing
    either __init__.py silently reintroduces the collision.
    """
    init = _REPO_ROOT / package_dir / "__init__.py"
    assert init.is_file(), (
        f"{package_dir}/__init__.py is missing. Its test_loss_builder.py and "
        f"test_optimization_builder.py then import under their bare basenames and "
        f"collide with the sibling builders/ directory."
    )
