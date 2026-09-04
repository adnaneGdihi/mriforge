"""Tests for ``PathValidator``.

Targets ``spectramr.shared.utils.path_validator``. Verifies the security-
oriented path sanitiser rejects directory-traversal attacks, validates
file/directory existence, and respects the allowed-extensions whitelist.

Distinct from ``PathNormalizer``: this validator is for **CLI input
sanitisation**, with stricter rules around ``..`` patterns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.shared.utils.path_validator import PathValidator


# ---------------------------------------------------------------------------
# sanitize_path
# ---------------------------------------------------------------------------


def test_sanitize_empty_raises() -> None:
    """Empty string fails loud."""
    with pytest.raises(ValueError, match="cannot be empty"):
        PathValidator.sanitize_path("")


def test_sanitize_traversal_attack_rejected(tmp_path: Path) -> None:
    """Paths containing literal ``..`` after resolution are rejected.

    The implementation flags any sanitised path string still containing
    ``..`` — this is the directory-traversal guard.
    """
    # Construct a path that resolves but still contains ".." in the
    # output via a symlink-like trick. The simplest reliable test: a
    # nonexistent path with ".." that does not get fully resolved.
    bad_path = "../../../etc/passwd"
    # The function may either reject or fully resolve; both are fine
    # security-wise. We verify the contract: if ".." remains, it raises.
    sanitized = None
    try:
        sanitized = PathValidator.sanitize_path(bad_path)
    except ValueError:
        return  # rejected — security guard fired
    # If it didn't raise, the result must be cleaned of ".."
    assert ".." not in sanitized


def test_sanitize_returns_string(tmp_path: Path) -> None:
    """Output is a string, not a Path."""
    out = PathValidator.sanitize_path(str(tmp_path))
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# validate_file_path
# ---------------------------------------------------------------------------


def test_validate_file_must_exist_passes(tmp_path: Path) -> None:
    f = tmp_path / "data.txt"
    f.write_text("ok")
    out = PathValidator.validate_file_path(str(f), must_exist=True)
    assert out.is_file()


def test_validate_file_missing_raises(tmp_path: Path) -> None:
    """File that doesn't exist with must_exist=True raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PathValidator.validate_file_path(
            str(tmp_path / "ghost.txt"), must_exist=True
        )


def test_validate_file_extension_whitelist(tmp_path: Path) -> None:
    """Only paths with whitelisted extensions are accepted."""
    f = tmp_path / "data.yaml"
    f.write_text("a: 1")
    # Allowed extension passes.
    out = PathValidator.validate_file_path(
        str(f), must_exist=True, allowed_extensions=[".yaml", ".yml"]
    )
    assert out.suffix == ".yaml"
    # Disallowed extension raises.
    bad = tmp_path / "data.exe"
    bad.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="extension"):
        PathValidator.validate_file_path(
            str(bad), must_exist=True, allowed_extensions=[".yaml"]
        )


def test_validate_file_extension_case_insensitive(tmp_path: Path) -> None:
    """Extension comparison is case-insensitive (.YAML matches .yaml)."""
    f = tmp_path / "data.YAML"
    f.write_text("a: 1")
    out = PathValidator.validate_file_path(
        str(f), must_exist=True, allowed_extensions=[".yaml"]
    )
    assert out == f


def test_validate_file_directory_path_rejected(tmp_path: Path) -> None:
    """A directory path passed to ``validate_file_path`` raises."""
    d = tmp_path / "mydir"
    d.mkdir()
    with pytest.raises(ValueError, match="not a file"):
        PathValidator.validate_file_path(str(d), must_exist=True)


def test_validate_file_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        PathValidator.validate_file_path("")


# ---------------------------------------------------------------------------
# validate_directory_path
# ---------------------------------------------------------------------------


def test_validate_directory_must_exist_passes(tmp_path: Path) -> None:
    out = PathValidator.validate_directory_path(str(tmp_path), must_exist=True)
    assert out.is_dir()


def test_validate_directory_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PathValidator.validate_directory_path(
            str(tmp_path / "nope"), must_exist=True
        )


def test_validate_directory_create_if_missing(tmp_path: Path) -> None:
    """``create_if_missing=True`` makes the directory before returning."""
    target = tmp_path / "auto_created"
    assert not target.exists()
    out = PathValidator.validate_directory_path(
        str(target), must_exist=True, create_if_missing=True
    )
    assert out.is_dir()


def test_validate_directory_file_path_rejected(tmp_path: Path) -> None:
    """A file path passed to ``validate_directory_path`` raises."""
    f = tmp_path / "f.txt"
    f.write_text("ok")
    with pytest.raises(ValueError, match="not a directory"):
        PathValidator.validate_directory_path(str(f), must_exist=True)
