"""Unit tests for :mod:`spectramr.shared.utils.path_security`.

Covers the directory-traversal-attack guards in
:class:`PathSecurityValidator` and the global singleton wrapper
:class:`DataPathSecurityManager`.

The security model docstring promises that:

1. Null-byte injection is rejected.
2. ``..`` parent-directory references are rejected (before path
   resolution — the resolver can't be trusted on its own).
3. ``//`` double-slashes are rejected as a normalisation-bypass
   vector.
4. URL-encoded traversal variants (``%2e%2e``, ``..\\``, ``%252e``)
   are rejected even in mixed case.
5. Absolute paths get re-anchored at ``root_dir``.
6. Paths that resolve outside ``root_dir`` raise ``ValueError`` even
   when each individual segment looks benign.
7. The directory / file-only validators add an
   ``IsADirectoryError`` / ``NotADirectoryError`` layer.
8. The singleton manager fails loud on use-before-initialize.

Pure-Python, no torch — runs in the default CPU lane in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.shared.utils.path_security import (
    DataPathSecurityManager,
    PathSecurityValidator,
)


# ── Construction ──────────────────────────────────────────────────


class TestValidatorConstruction:
    def test_default_root_is_cwd(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # When ``root_dir=None`` the validator uses ``os.getcwd()``.
        monkeypatch.chdir(tmp_path)
        v = PathSecurityValidator(root_dir=None)
        assert v.root_dir == tmp_path.resolve()

    def test_root_dir_resolved_to_absolute(self, tmp_path: Path) -> None:
        v = PathSecurityValidator(str(tmp_path))
        assert v.root_dir.is_absolute()
        assert v.root_dir == tmp_path.resolve()

    def test_missing_root_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            PathSecurityValidator(str(tmp_path / "ghost"))

    def test_root_dir_is_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="not a directory"):
            PathSecurityValidator(str(f))


# ── validate_safe_path — happy paths ──────────────────────────────


class TestValidateSafePathHappy:
    @pytest.fixture
    def populated_root(self, tmp_path: Path) -> Path:
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "a.txt").write_text("ok")
        (tmp_path / "data" / "sub").mkdir()
        return tmp_path

    def test_relative_subpath_validates(self, populated_root: Path) -> None:
        v = PathSecurityValidator(str(populated_root))
        out = v.validate_safe_path("data/a.txt")
        assert out.is_absolute()
        assert out.exists()
        # Must resolve under the root.
        out.relative_to(populated_root.resolve())  # no-raise

    def test_must_exist_false_allows_missing(self, populated_root: Path) -> None:
        v = PathSecurityValidator(str(populated_root))
        out = v.validate_safe_path("data/not_yet.txt", must_exist=False)
        # Returned path exists in the namespace but not on disk.
        assert not out.exists()

    def test_must_exist_true_default_raises_for_missing(
        self, populated_root: Path
    ) -> None:
        v = PathSecurityValidator(str(populated_root))
        with pytest.raises(FileNotFoundError):
            v.validate_safe_path("data/ghost.txt")


# ── validate_safe_path — security violations ──────────────────────


class TestValidateSafePathRejects:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "data").mkdir()
        return tmp_path

    def test_null_byte_in_path_raises(self, root: Path) -> None:
        v = PathSecurityValidator(str(root))
        with pytest.raises(ValueError, match="null bytes"):
            v.validate_safe_path("data/x\x00.txt")

    def test_parent_traversal_dotdot_raises(self, root: Path) -> None:
        v = PathSecurityValidator(str(root))
        with pytest.raises(ValueError, match="parent directory reference"):
            v.validate_safe_path("data/../etc/passwd")

    def test_dotdot_anywhere_in_path_raises(self, root: Path) -> None:
        v = PathSecurityValidator(str(root))
        with pytest.raises(ValueError, match="parent directory reference"):
            # Even at the leaf, the segment-by-segment scan catches it.
            v.validate_safe_path("data/sub/..")

    def test_double_slash_raises(self, root: Path) -> None:
        v = PathSecurityValidator(str(root))
        with pytest.raises(ValueError, match="double slashes"):
            v.validate_safe_path("data//x.txt")

    @pytest.mark.parametrize(
        "pattern",
        ["%2e%2e/etc", "data/..\\windows", "%252e/file"],
    )
    def test_suspicious_encoded_pattern_raises(
        self, root: Path, pattern: str
    ) -> None:
        v = PathSecurityValidator(str(root))
        with pytest.raises(ValueError, match="suspicious pattern"):
            v.validate_safe_path(pattern)

    def test_suspicious_pattern_uppercase_also_caught(self, root: Path) -> None:
        """Suspicious-pattern check normalises to lowercase before matching."""
        v = PathSecurityValidator(str(root))
        with pytest.raises(ValueError, match="suspicious pattern"):
            v.validate_safe_path("%2E%2E/etc")

    def test_absolute_path_outside_root_re_anchored(self, root: Path) -> None:
        """Absolute paths get re-anchored to root; ``/data/a`` becomes
        ``<root>/data/a``."""
        v = PathSecurityValidator(str(root))
        # ``/data/`` will be re-anchored to root/data/, which exists.
        out = v.validate_safe_path("/data", must_exist=True)
        out.relative_to(root.resolve())


# ── validate_directory_path / validate_file_path ──────────────────


class TestDirectoryAndFileValidators:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "f.txt").write_text("ok")
        return tmp_path

    def test_validate_directory_path_accepts_dir(self, root: Path) -> None:
        v = PathSecurityValidator(str(root))
        out = v.validate_directory_path("data")
        assert out.is_dir()

    def test_validate_directory_path_rejects_file(self, root: Path) -> None:
        v = PathSecurityValidator(str(root))
        with pytest.raises(NotADirectoryError):
            v.validate_directory_path("data/f.txt")

    def test_validate_file_path_accepts_file(self, root: Path) -> None:
        v = PathSecurityValidator(str(root))
        out = v.validate_file_path("data/f.txt")
        assert out.is_file()

    def test_validate_file_path_rejects_directory(self, root: Path) -> None:
        v = PathSecurityValidator(str(root))
        with pytest.raises(IsADirectoryError):
            v.validate_file_path("data")


# ── validate_file_pattern ────────────────────────────────────────


class TestValidateFilePattern:
    @pytest.fixture
    def v(self, tmp_path: Path) -> PathSecurityValidator:
        return PathSecurityValidator(str(tmp_path))

    def test_clean_pattern_returned_unchanged(self, v: PathSecurityValidator) -> None:
        assert v.validate_file_pattern("*.h5") == "*.h5"

    def test_null_byte_pattern_raises(self, v: PathSecurityValidator) -> None:
        with pytest.raises(ValueError, match="null bytes"):
            v.validate_file_pattern("*\x00.h5")

    def test_traversal_pattern_raises(self, v: PathSecurityValidator) -> None:
        with pytest.raises(ValueError, match="parent directory reference"):
            v.validate_file_pattern("../etc/*")


# ── DataPathSecurityManager singleton ────────────────────────────


class TestDataPathSecurityManager:
    @pytest.fixture(autouse=True)
    def _reset_singleton(self) -> None:
        # Each test starts uninitialised; restore at the end.
        prior = DataPathSecurityManager._validator
        DataPathSecurityManager._validator = None
        yield
        DataPathSecurityManager._validator = prior

    def test_get_validator_before_init_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            DataPathSecurityManager.get_validator()

    def test_validate_path_before_init_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            DataPathSecurityManager.validate_path("x")

    def test_initialize_and_use(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("ok")
        DataPathSecurityManager.initialize(str(tmp_path))
        v = DataPathSecurityManager.get_validator()
        assert isinstance(v, PathSecurityValidator)
        # Round-trip through the class-method facade.
        out = DataPathSecurityManager.validate_path("f.txt")
        assert out.is_file()

    def test_validate_directory_through_facade(self, tmp_path: Path) -> None:
        (tmp_path / "d").mkdir()
        DataPathSecurityManager.initialize(str(tmp_path))
        out = DataPathSecurityManager.validate_directory("d")
        assert out.is_dir()

    def test_validate_file_through_facade(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("ok")
        DataPathSecurityManager.initialize(str(tmp_path))
        out = DataPathSecurityManager.validate_file("f.txt")
        assert out.is_file()

    def test_initialize_with_allow_symlinks(self, tmp_path: Path) -> None:
        DataPathSecurityManager.initialize(
            str(tmp_path), allow_symlinks=True
        )
        v = DataPathSecurityManager.get_validator()
        assert v.allow_symlinks is True
