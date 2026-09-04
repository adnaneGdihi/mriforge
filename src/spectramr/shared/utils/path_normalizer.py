"""Path normalization and validation utilities for configuration and runtime.

This module provides a centralized place for all path normalization and validation
operations, ensuring that:
1. Relative paths are resolved consistently (relative to project root or root_dir)
2. All paths are validated for existence (if required)
3. Symlinks and environment variables are handled properly
4. Single Source of Truth (SSOT) for path resolution logic
"""

import os
from pathlib import Path


class PathNormalizer:
    """Utility class for normalizing and validating filesystem paths."""

    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory.

        Returns the value of SPECTRAMR_ROOT environment variable if set,
        otherwise returns the current working directory.

        Returns:
            Path to project root directory
        """
        env_root = os.environ.get("SPECTRAMR_ROOT")
        if env_root:
            return Path(env_root).expanduser().resolve()
        # Default to current working directory if no env var set
        return Path.cwd()

    @staticmethod
    def normalize_and_validate(
        path: str,
        root_dir: Path | None = None,
        must_exist: bool = False,
    ) -> Path:
        """Normalize path to absolute; optionally validate existence.

        Converts relative paths to absolute paths relative to root_dir
        (or project root if not specified). Handles environment variable
        expansion and symlink resolution.

        Args:
            path: Input path (relative or absolute string)
            root_dir: Root directory for relative path resolution.
                     Defaults to project root (from SPECTRAMR_ROOT env var or cwd).
            must_exist: If True, raise ValueError if path does not exist
                       after normalization.

        Returns:
            Resolved absolute Path object

        Raises:
            ValueError: If must_exist=True and path does not exist after normalization

        Example:
            >>> # Relative path resolved relative to project root
            >>> p = PathNormalizer.normalize_and_validate(
            ...     "./databases/fastmri/datasets/knee_singlecoil_train/singlecoil_train/"
            ... )
            >>> p.is_absolute()
            True

            >>> # Absolute path returned as-is (resolved)
            >>> p = PathNormalizer.normalize_and_validate(
            ...     "/data/fastmri/datasets/knee_singlecoil_train/singlecoil_train/"
            ... )
            >>> p.is_absolute()
            True

            >>> # Environment variable expansion
            >>> os.environ["FASTMRI_DATASETS_ROOT"] = "/data/fastmri/datasets"
            >>> p = PathNormalizer.normalize_and_validate(
            ...     "$FASTMRI_DATASETS_ROOT/knee_singlecoil_train/singlecoil_train/"
            ... )
            >>> str(p).startswith("/data/fastmri/datasets")
            True
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Expand environment variables and user home (~)
        expanded_path = os.path.expandvars(os.path.expanduser(path))

        # Auto-remap a legacy cluster absolute prefix to local paths if running
        # locally. Override via SPECTRAMR_LEGACY_CLUSTER_PREFIX (e.g. set to
        # "/scratch/me/spectramr/" for your own cluster). Empty string disables.
        cluster_prefix = os.environ.get("SPECTRAMR_LEGACY_CLUSTER_PREFIX", "")
        if cluster_prefix and expanded_path.startswith(cluster_prefix):
            rel_part = expanded_path[len(cluster_prefix) :]
            if root_dir is None:
                root_dir = PathNormalizer.get_project_root()
            path_obj = (root_dir / rel_part).resolve()
        else:
            path_obj = Path(expanded_path)
            # If not absolute, resolve relative to root_dir
            if not path_obj.is_absolute():
                if root_dir is None:
                    root_dir = PathNormalizer.get_project_root()
                path_obj = (root_dir / path_obj).resolve()
            else:
                # Still resolve to clean up any ./ or ../ components
                path_obj = path_obj.resolve()

        # Validate existence if required
        if must_exist and not path_obj.exists():
            raise ValueError(
                f"Path does not exist after normalization: {path_obj} (original: {path})"
            )

        return path_obj

    @staticmethod
    def normalize_and_validate_directory(
        path: str,
        root_dir: Path | None = None,
        must_exist: bool = True,
    ) -> Path:
        """Normalize and validate that path is an existing directory.

        Args:
            path: Input path to directory
            root_dir: Root directory for relative path resolution
            must_exist: If True, raise ValueError if directory does not exist

        Returns:
            Resolved absolute Path object pointing to directory

        Raises:
            ValueError: If must_exist=True and path does not exist,
                       or if path exists but is not a directory
        """
        normalized = PathNormalizer.normalize_and_validate(
            path, root_dir=root_dir, must_exist=False
        )

        if must_exist:
            if not normalized.exists():
                raise ValueError(f"Directory does not exist: {normalized}")
            if not normalized.is_dir():
                raise ValueError(f"Path is not a directory: {normalized}")

        return normalized

    @staticmethod
    def normalize_and_validate_file(
        path: str,
        root_dir: Path | None = None,
        must_exist: bool = True,
    ) -> Path:
        """Normalize and validate that path is an existing file.

        Args:
            path: Input path to file
            root_dir: Root directory for relative path resolution
            must_exist: If True, raise ValueError if file does not exist

        Returns:
            Resolved absolute Path object pointing to file

        Raises:
            ValueError: If must_exist=True and path does not exist,
                       or if path exists but is not a file
        """
        normalized = PathNormalizer.normalize_and_validate(
            path, root_dir=root_dir, must_exist=False
        )

        if must_exist:
            if not normalized.exists():
                raise ValueError(f"File does not exist: {normalized}")
            if not normalized.is_file():
                raise ValueError(f"Path is not a file: {normalized}")

        return normalized
