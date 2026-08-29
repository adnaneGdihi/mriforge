"""Tests for ``mriforge.shared.utils.security``.

Targets path-traversal guards, file-size limits, filename sanitisation,
image-extension whitelisting, and the secure-temp-file helper. These
are security boundaries — every fail-loud branch is exercised.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mriforge.shared.utils.security import (
    FileSizeLimitError,
    PathTraversalError,
    SecurityError,
    check_file_size,
    sanitize_filename,
    secure_temp_file,
    validate_directory_path,
    validate_file_path,
    validate_image_file,
)


# ---------------------------------------------------------------------------
# validate_file_path
# ---------------------------------------------------------------------------


def test_validate_file_path_rejects_empty() -> None:
    """Empty path → ``ValueError``."""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_file_path("")


def test_validate_file_path_rejects_dot_dot_traversal() -> None:
    """``..`` segment → ``PathTraversalError``."""
    with pytest.raises(PathTraversalError, match="traversal"):
        validate_file_path("../etc/passwd")


def test_validate_file_path_rejects_backslash() -> None:
    """Windows-style backslash → ``PathTraversalError``."""
    with pytest.raises(PathTraversalError, match="traversal"):
        validate_file_path("foo\\bar")


def test_validate_file_path_rejects_absolute_by_default() -> None:
    """Absolute path without ``allow_absolute`` → ``PathTraversalError``."""
    with pytest.raises(PathTraversalError, match="Absolute"):
        validate_file_path("/etc/passwd")


def test_validate_file_path_accepts_absolute_with_flag() -> None:
    """``allow_absolute=True`` permits absolute paths."""
    out = validate_file_path("/tmp/safe.txt", allow_absolute=True)
    assert isinstance(out, Path)
    assert out.is_absolute()


def test_validate_file_path_resolves_relative_to_base_dir(tmp_path: Path) -> None:
    """``base_dir`` resolves relative paths."""
    out = validate_file_path("inner/file.txt", base_dir=tmp_path)
    assert str(out).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# validate_directory_path
# ---------------------------------------------------------------------------


def test_validate_directory_path_existing(tmp_path: Path) -> None:
    """Existing directory → returned unchanged."""
    out = validate_directory_path(tmp_path)
    assert out == tmp_path.resolve()


def test_validate_directory_path_creates_when_missing(tmp_path: Path) -> None:
    """``create_if_missing=True`` creates the directory."""
    target = tmp_path / "new_dir"
    out = validate_directory_path(target, create_if_missing=True)
    assert out.exists()
    assert out.is_dir()


def test_validate_directory_path_missing_no_create_raises(tmp_path: Path) -> None:
    """Missing directory + ``create_if_missing=False`` → ``ValueError``."""
    with pytest.raises(ValueError, match="does not exist"):
        validate_directory_path(tmp_path / "ghost", create_if_missing=False)


def test_validate_directory_path_existing_file_raises(tmp_path: Path) -> None:
    """Path is a file, not a directory → ``ValueError``."""
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        validate_directory_path(f)


# ---------------------------------------------------------------------------
# check_file_size
# ---------------------------------------------------------------------------


def test_check_file_size_returns_size_bytes(tmp_path: Path) -> None:
    """Returns the file size in bytes."""
    f = tmp_path / "x.txt"
    f.write_bytes(b"1234")
    assert check_file_size(f, max_size_mb=1.0) == 4


def test_check_file_size_raises_on_oversize(tmp_path: Path) -> None:
    """File exceeding ``max_size_mb`` → ``FileSizeLimitError``."""
    f = tmp_path / "big.bin"
    f.write_bytes(b"\x00" * 1024)  # 1 KiB
    with pytest.raises(FileSizeLimitError, match="exceeds limit"):
        check_file_size(f, max_size_mb=0.0001)  # 100 bytes limit


def test_check_file_size_missing_file_raises(tmp_path: Path) -> None:
    """Missing file → ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        check_file_size(tmp_path / "ghost.txt")


def test_check_file_size_directory_raises(tmp_path: Path) -> None:
    """Path is directory → ``ValueError``."""
    with pytest.raises(ValueError, match="not a file"):
        check_file_size(tmp_path)


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char", ["/", "\\", ":", "*", "?", '"', "<", ">", "|"])
def test_sanitize_filename_strips_dangerous_chars(char: str) -> None:
    """Each dangerous char is replaced with ``_``."""
    result = sanitize_filename(f"foo{char}bar.txt")
    assert char not in result


def test_sanitize_filename_truncates_long_names() -> None:
    """Names exceeding ``max_length`` are truncated."""
    long_name = "a" * 500
    out = sanitize_filename(long_name, max_length=100)
    assert len(out) <= 100


def test_sanitize_filename_handles_empty_with_default() -> None:
    """Empty filename → default ``'unnamed_file'``."""
    assert sanitize_filename("") == "unnamed_file"


def test_sanitize_filename_prefixes_windows_reserved() -> None:
    """Reserved Windows names (CON, PRN, etc.) get an underscore prefix."""
    out = sanitize_filename("CON.txt")
    assert out.startswith("_")


def test_sanitize_filename_strips_control_chars() -> None:
    """Control characters (ord < 32) are removed."""
    out = sanitize_filename("foo\x01bar")
    assert "\x01" not in out


# ---------------------------------------------------------------------------
# validate_image_file
# ---------------------------------------------------------------------------


def test_validate_image_file_accepts_documented_extensions(tmp_path: Path) -> None:
    """A ``.png`` file under the size limit is accepted."""
    f = tmp_path / "img.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
    out = validate_image_file(f)
    assert out.suffix == ".png"


def test_validate_image_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Unknown extension → ``ValueError``."""
    f = tmp_path / "doc.exe"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported image format"):
        validate_image_file(f)


def test_validate_image_file_handles_compound_nii_gz(tmp_path: Path) -> None:
    """``.nii.gz`` is recognised as a single compound suffix."""
    f = tmp_path / "scan.nii.gz"
    f.write_bytes(b"x")
    out = validate_image_file(f)
    assert out == f.resolve()


def test_validate_image_file_h5_gets_larger_size_budget(tmp_path: Path) -> None:
    """``.h5`` files have a 500 MB budget rather than the 50 MB default."""
    f = tmp_path / "data.h5"
    f.write_bytes(b"\x00" * 1024)  # 1 KiB — well under both budgets
    out = validate_image_file(f, max_size_mb=0.0005)  # 500 bytes for non-h5
    # H5 path overrides max_size_mb internally with 500 MB budget.
    assert out == f.resolve()


# ---------------------------------------------------------------------------
# secure_temp_file
# ---------------------------------------------------------------------------


def test_secure_temp_file_creates_and_cleans_up() -> None:
    """File exists during the context; deleted afterwards."""
    with secure_temp_file(suffix=".tmp", prefix="test_") as path:
        assert path.exists()
        assert path.suffix == ".tmp"
        assert path.name.startswith("test_")
        recorded = path
    assert not recorded.exists()  # cleaned up


def test_secure_temp_file_posix_permissions() -> None:
    """On POSIX, temp file is mode 0600."""
    if os.name != "posix":
        pytest.skip("POSIX permissions test")
    with secure_temp_file() as path:
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_path_traversal_inherits_from_security_error() -> None:
    """``PathTraversalError`` is a ``SecurityError`` subclass."""
    assert issubclass(PathTraversalError, SecurityError)


def test_file_size_limit_inherits_from_security_error() -> None:
    """``FileSizeLimitError`` is a ``SecurityError`` subclass."""
    assert issubclass(FileSizeLimitError, SecurityError)
