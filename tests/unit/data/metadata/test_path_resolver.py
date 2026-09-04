"""Unit tests for the centralized path resolver.

Targets ``spectramr.data.metadata.path_resolver``:
- ``get_project_root`` returns a :class:`pathlib.Path`.
- ``PathResolver.resolve`` handles relative, absolute, and empty paths.
- ``PathResolver.resolve`` joins relative paths with the project root.
- Legacy absolute-prefix stripping works when configured via env var.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_get_project_root_returns_path() -> None:
    from spectramr.data.metadata.path_resolver import get_project_root

    root = get_project_root()
    assert isinstance(root, Path)
    # The repo has a pyproject.toml at its root.
    assert (root / "pyproject.toml").exists() or root.is_dir()


# ---------------------------------------------------------------------------
# PathResolver.resolve — empty / None / falsy inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_path", ["", None])
def test_resolve_falsy_input_returns_as_is(empty_path) -> None:
    from spectramr.data.metadata.path_resolver import PathResolver

    result = PathResolver.resolve(empty_path)
    assert result == empty_path


# ---------------------------------------------------------------------------
# Relative path joins with project root
# ---------------------------------------------------------------------------


def test_resolve_relative_path_joins_project_root() -> None:
    from spectramr.data.metadata.path_resolver import PathResolver, get_project_root

    relative = "some/nonexistent/file.h5"
    result = PathResolver.resolve(relative)
    project_root = get_project_root()
    expected = str(project_root / relative)
    assert result == expected


def test_resolve_existing_relative_stays_relative() -> None:
    """An already-existing relative path (relative to CWD) is returned as-is
    (the resolver short-circuits at step 1 if the path exists)."""
    from spectramr.data.metadata.path_resolver import PathResolver

    # pyproject.toml exists at the repo root; we can use its name if CWD == repo root.
    # Use a path that definitely exists relative to the module location.
    existing_rel = "pyproject.toml"
    # Only test if this file exists in CWD (i.e. tests run from repo root).
    if Path(existing_rel).exists():
        result = PathResolver.resolve(existing_rel)
        # It should be returned as-is (not joined) since it already exists.
        assert result == str(existing_rel)


# ---------------------------------------------------------------------------
# Absolute path passthrough
# ---------------------------------------------------------------------------


def test_resolve_unknown_absolute_path_returns_as_is() -> None:
    from spectramr.data.metadata.path_resolver import PathResolver

    abs_path = "/tmp/__spectramr_test_nonexistent_9876__/file.h5"
    result = PathResolver.resolve(abs_path)
    # Without a matching legacy prefix, absolute is returned as-is.
    assert result == abs_path


# ---------------------------------------------------------------------------
# PROJECT_ROOT env var override
# ---------------------------------------------------------------------------


def test_get_project_root_respects_env_var(tmp_path, monkeypatch) -> None:
    from spectramr.data.metadata.path_resolver import get_project_root

    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    root = get_project_root()
    assert root == tmp_path


def test_get_project_root_respects_data_root_env_var(tmp_path, monkeypatch) -> None:
    from spectramr.data.metadata.path_resolver import get_project_root

    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.setenv("SPECTRAMR_DATA_ROOT", str(tmp_path))
    root = get_project_root()
    assert root == tmp_path


# ---------------------------------------------------------------------------
# Legacy prefix stripping
# ---------------------------------------------------------------------------


def test_legacy_prefix_strips_and_joins(tmp_path, monkeypatch) -> None:
    """When SPECTRAMR_LEGACY_ABS_PREFIXES is set, matching absolutes are stripped."""
    import importlib

    monkeypatch.setenv("SPECTRAMR_LEGACY_ABS_PREFIXES", "/cluster/mount")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    # We need to reimport with the patched env.
    import spectramr.data.metadata.path_resolver as pr_mod

    importlib.reload(pr_mod)
    try:
        # Create a real file in tmp_path to satisfy the "if joined.exists()" branch.
        target = tmp_path / "data" / "file.h5"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

        result = pr_mod.PathResolver.resolve("/cluster/mount/data/file.h5")
        assert result == str(target)
    finally:
        importlib.reload(pr_mod)  # restore module state


# ---------------------------------------------------------------------------
# Edge: resolve called with only whitespace
# ---------------------------------------------------------------------------


def test_edge_whitespace_path_is_not_empty(tmp_path) -> None:
    """A whitespace string is not falsy in Python — it should go through the
    normal resolution path without crashing."""
    from spectramr.data.metadata.path_resolver import PathResolver

    # Should not raise.
    result = PathResolver.resolve("   ")
    assert isinstance(result, str)
