"""Shared loader for the ``scripts/release/`` modules under test.

These modules import each other by **bare name** -- they are scripts, not a
package -- so whoever loads them has to put ``scripts/release`` on the path
first. ``assert_public_repo_settings.py`` does that for itself at runtime; a
test that loads a module by file path gets no package context and must do the
same, which is what this file owns for every test in this directory.

Loading by *path* rather than ``from scripts.release import ...`` is not a
style choice: ``tests/unit/scripts`` is a regular package and shadows the root
``scripts`` directory, so the dotted form resolves to the wrong tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_DIR = REPO_ROOT / "scripts" / "release"

if str(RELEASE_DIR) not in sys.path:
    sys.path.insert(0, str(RELEASE_DIR))


def load_release_module(stem: str, alias: str | None = None) -> ModuleType:
    """Load ``scripts/release/<stem>.py`` by path.

    The module is registered in ``sys.modules`` **before** it is executed:
    ``@dataclass`` resolves ``cls.__module__`` through ``sys.modules`` to decide
    whether an annotation is ``KW_ONLY``, so an unregistered path-loaded module
    raises AttributeError on the decorator rather than on any use of it.
    """
    name = alias or f"{stem}_under_test"
    spec = importlib.util.spec_from_file_location(name, RELEASE_DIR / f"{stem}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
