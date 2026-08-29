"""Tests for ``PathNormalizer``.

Targets ``mriforge.shared.utils.path_normalizer``. Verifies path resolution
(relative-vs-absolute), environment-variable expansion, the cluster-
prefix auto-remap, and the file/directory must-exist validators.

All tests use ``tmp_path`` for filesystem isolation — none touch real
project paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mriforge.shared.utils.path_normalizer import PathNormalizer


# ---------------------------------------------------------------------------
# get_project_root
# ---------------------------------------------------------------------------


def test_project_root_uses_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``MRIFORGE_ROOT`` env var overrides the cwd default."""
    monkeypatch.setenv("MRIFORGE_ROOT", str(tmp_path))
    assert PathNormalizer.get_project_root() == tmp_path.resolve()


def test_project_root_falls_back_to_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``MRIFORGE_ROOT``, returns ``Path.cwd()``."""
    monkeypatch.delenv("MRIFORGE_ROOT", raising=False)
    assert PathNormalizer.get_project_root() == Path.cwd()


# ---------------------------------------------------------------------------
# normalize_and_validate
# ---------------------------------------------------------------------------


def test_empty_path_raises() -> None:
    """Empty string fails loud (CLAUDE.md #9 — no silent default)."""
    with pytest.raises(ValueError, match="Path cannot be empty"):
        PathNormalizer.normalize_and_validate("")


def test_absolute_path_returned_resolved(tmp_path: Path) -> None:
    """Absolute path is returned as-is (after resolution)."""
    target = tmp_path / "data"
    target.mkdir()
    out = PathNormalizer.normalize_and_validate(str(target))
    assert out == target.resolve()
    assert out.is_absolute()


def test_relative_path_resolved_against_root_dir(tmp_path: Path) -> None:
    """Relative paths are resolved relative to ``root_dir``."""
    out = PathNormalizer.normalize_and_validate("subdir/file.txt", root_dir=tmp_path)
    assert out == (tmp_path / "subdir" / "file.txt").resolve()


def test_environment_variable_expansion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``$VAR`` references are expanded before resolution."""
    monkeypatch.setenv("MY_DATA_ROOT", str(tmp_path))
    out = PathNormalizer.normalize_and_validate("$MY_DATA_ROOT/subdir")
    assert out == (tmp_path / "subdir").resolve()


def test_user_home_expansion() -> None:
    """``~`` is expanded to the user home directory."""
    out = PathNormalizer.normalize_and_validate("~/somefile")
    home = Path.home().resolve()
    assert str(out).startswith(str(home))


def test_cluster_prefix_auto_remap(tmp_path: Path, monkeypatch) -> None:
    """``<configured prefix>/X`` → ``<root>/X`` when running locally.

    The prefix is **configured**, never baked in. This test used to pass one
    site's absolute checkout path and assert the remap fired -- which stopped
    being true when the default became the empty string (disabled), leaving the
    test red on ``dev`` and the "auto"-remap unreachable unless configured.
    Setting the variable the way a real user does makes this test exercise the
    remap rather than one machine's directory layout.
    """
    monkeypatch.setenv(
        "MRIFORGE_LEGACY_CLUSTER_PREFIX", "/project/alpha_lab/researcher/mriforge/"
    )
    out = PathNormalizer.normalize_and_validate(
        "/project/alpha_lab/researcher/mriforge/databases/foo",
        root_dir=tmp_path,
    )
    assert out == (tmp_path / "databases" / "foo").resolve()


def test_cluster_prefix_remap_is_off_unless_configured(tmp_path: Path, monkeypatch) -> None:
    """Unset (or empty) ``MRIFORGE_LEGACY_CLUSTER_PREFIX`` leaves absolute paths alone.

    This is the half the previous test silently asserted the opposite of. An
    absolute path is returned as given -- no site prefix is assumed on anyone's
    behalf, which is what makes the class safe to ship.
    """
    monkeypatch.delenv("MRIFORGE_LEGACY_CLUSTER_PREFIX", raising=False)
    untouched = "/project/alpha_lab/researcher/mriforge/databases/foo"
    assert PathNormalizer.normalize_and_validate(untouched, root_dir=tmp_path) == Path(
        untouched
    )

    monkeypatch.setenv("MRIFORGE_LEGACY_CLUSTER_PREFIX", "")
    assert PathNormalizer.normalize_and_validate(untouched, root_dir=tmp_path) == Path(
        untouched
    )


def test_must_exist_raises_when_missing(tmp_path: Path) -> None:
    """``must_exist=True`` fails loud if the resolved path is absent."""
    with pytest.raises(ValueError, match="Path does not exist"):
        PathNormalizer.normalize_and_validate(
            "definitely_not_here", root_dir=tmp_path, must_exist=True
        )


def test_must_exist_passes_when_present(tmp_path: Path) -> None:
    """``must_exist=True`` succeeds when the path exists."""
    target = tmp_path / "exists.txt"
    target.write_text("ok")
    out = PathNormalizer.normalize_and_validate(
        str(target), must_exist=True
    )
    assert out == target.resolve()


# ---------------------------------------------------------------------------
# normalize_and_validate_directory
# ---------------------------------------------------------------------------


def test_directory_validator_accepts_directory(tmp_path: Path) -> None:
    """Directory validator passes when the path is a directory."""
    target = tmp_path / "mydir"
    target.mkdir()
    out = PathNormalizer.normalize_and_validate_directory(str(target))
    assert out == target.resolve()


def test_directory_validator_rejects_file(tmp_path: Path) -> None:
    """A file path fails the directory validator."""
    f = tmp_path / "file.txt"
    f.write_text("data")
    with pytest.raises(ValueError, match="not a directory"):
        PathNormalizer.normalize_and_validate_directory(str(f))


def test_directory_validator_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Directory does not exist"):
        PathNormalizer.normalize_and_validate_directory(str(tmp_path / "absent"))


# ---------------------------------------------------------------------------
# normalize_and_validate_file
# ---------------------------------------------------------------------------


def test_file_validator_accepts_file(tmp_path: Path) -> None:
    """File validator passes when the path is a file."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00")
    out = PathNormalizer.normalize_and_validate_file(str(f))
    assert out == f.resolve()


def test_file_validator_rejects_directory(tmp_path: Path) -> None:
    d = tmp_path / "dir"
    d.mkdir()
    with pytest.raises(ValueError, match="not a file"):
        PathNormalizer.normalize_and_validate_file(str(d))


def test_file_validator_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="File does not exist"):
        PathNormalizer.normalize_and_validate_file(str(tmp_path / "ghost.txt"))
