"""Import-smoke + regression tests for ``mriforge.tools.tune``.

WS-8 (Outer-shell) fix: ``src/mriforge/tools/tune.py`` previously carried a
``sys.path.append(...)`` hack (plus the now-unused ``sys`` and ``Path``
imports) so the script could be run as a loose file. With the package
installed (``pip install -e``) and the ``mriforge.tools`` namespace in place,
that manipulation is dead weight and risks shadowing the real install path.

These tests pin the fix:

1. The module imports cleanly (no ``sys.path`` manipulation at import time)
   and exposes a callable ``main`` entry point.
2. The module source contains no ``sys.path.append`` and no leftover
   ``sys`` / ``Path`` imports, so the hack cannot silently creep back in.

Kept deliberately as an import-smoke test — the heavy Optuna / training
machinery is only reached inside ``main()``, which we never invoke here.
"""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest

pytestmark = pytest.mark.unit

MODULE_NAME = "mriforge.tools.tune"


def test_tune_imports_cleanly_and_exposes_main() -> None:
    """Importing the module succeeds and yields a callable ``main``."""
    module = importlib.import_module(MODULE_NAME)

    assert hasattr(module, "main"), "mriforge.tools.tune must expose `main`"
    assert callable(module.main), "`main` must be callable"


def test_tune_source_has_no_syspath_hack_or_unused_imports() -> None:
    """Regression: no ``sys.path.append`` and no ``sys`` / ``Path`` imports.

    Reads the *worktree* source (where the fix lives) and asserts the
    removed lines are gone for good.
    """
    module = importlib.import_module(MODULE_NAME)
    source = inspect.getsource(module)

    # The literal hack must be gone.
    assert "sys.path.append" not in source, "sys.path.append hack must be removed"

    # The now-unused imports must be gone. Parse the AST so a substring like
    # "Path" inside a docstring/comment can't trip the assertion.
    tree = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    assert "sys" not in imported_names, "unused `sys` import must be removed"
    assert "Path" not in imported_names, "unused `Path` import must be removed"
