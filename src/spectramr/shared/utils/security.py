#!/usr/bin/env python3
"""Security utilities for safe file operations and input validation.

This module provides security-hardened functions for file operations,
path validation, and input sanitization to prevent common security
vulnerabilities like path traversal attacks and resource exhaustion.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if nibabel is available
try:
    import nibabel  # noqa: F401

    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class SecurityError(Exception):
    """Base exception for security-related errors."""


class PathTraversalError(SecurityError):
    """Exception raised when path traversal is detected."""


class FileSizeLimitError(SecurityError):
    """Exception raised when file size exceeds limits."""


def validate_file_path(
    file_path: str | Path,
    base_dir: str | Path | None = None,
    allow_absolute: bool = False,
) -> Path:
    """Validate and sanitize file path to prevent path traversal attacks.

    Args:
        file_path: The file path to validate
        base_dir: Base directory to resolve relative paths against
        allow_absolute: Whether to allow absolute paths

    Returns:
        Resolved and validated Path object

    Raises:
        PathTraversalError: If path traversal is detected
        ValueError: If path is invalid

    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    path = Path(file_path)

    # Check for path traversal attempts (both Unix and Windows style)
    path_str = str(path)
    if ".." in path_str or "\\" in path_str or (path.is_absolute() and not allow_absolute):
        if not allow_absolute and path.is_absolute():
            raise PathTraversalError(f"Absolute paths not allowed: {path}")
        if ".." in path_str or "\\" in path_str:
            raise PathTraversalError(f"Path traversal detected: {path}")

    # Resolve the path
    if base_dir and not path.is_absolute():
        resolved_path = (Path(base_dir) / path).resolve()
    else:
        resolved_path = path.resolve()

    return resolved_path


def validate_directory_path(
    dir_path: str | Path,
    base_dir: str | Path | None = None,
    create_if_missing: bool = False,
) -> Path:
    """Validate directory path and optionally create it.

    Args:
        dir_path: Directory path to validate
        base_dir: Base directory for relative paths
        create_if_missing: Whether to create directory if it doesn't exist

    Returns:
        Validated directory Path

    Raises:
        PathTraversalError: If path traversal is detected
        NotADirectoryError: If path exists but is not a directory

    """
    validated_path = validate_file_path(dir_path, base_dir, allow_absolute=True)

    if validated_path.exists():
        if not validated_path.is_dir():
            raise ValueError(f"Path is not a directory: {validated_path}")
    elif not create_if_missing:
        raise ValueError(f"Directory does not exist: {validated_path}")
    else:
        validated_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {validated_path}")

    return validated_path


def check_file_size(file_path: str | Path, max_size_mb: float = 100.0) -> int:
    """Check file size and ensure it doesn't exceed limits.

    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed file size in MB

    Returns:
        File size in bytes

    Raises:
        FileSizeLimitError: If file exceeds size limit
        FileNotFoundError: If file doesn't exist

    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    size_bytes = path.stat().st_size
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    if size_bytes > max_size_bytes:
        raise FileSizeLimitError(
            f"File size {size_bytes} bytes exceeds limit of {max_size_bytes} bytes: {path}",
        )

    return size_bytes


def safe_open_file(
    file_path: str | Path,
    mode: str = "r",
    encoding: str = "utf-8",
    base_dir: str | Path | None = None,
    max_size_mb: float = 100.0,
    **kwargs,
):
    """Safely open a file with validation and size limits.

    Args:
        file_path: Path to the file
        mode: File mode ('r', 'w', 'rb', 'wb', etc.)
        encoding: Text encoding for text modes
        base_dir: Base directory for path validation
        max_size_mb: Maximum file size in MB for reading
        **kwargs: Additional arguments for open()

    Returns:
        File object

    Raises:
        PathTraversalError: If path traversal is detected
        FileSizeLimitError: If file exceeds size limit

    """
    validated_path = validate_file_path(file_path, base_dir, allow_absolute=True)

    # For reading modes, check file size
    if mode in ("r", "rb") and "w" not in mode and "a" not in mode:
        check_file_size(validated_path, max_size_mb)

    # Set encoding for text modes
    if "b" not in mode:
        kwargs["encoding"] = encoding

    try:
        return open(validated_path, mode, **kwargs)
    except OSError as e:
        logger.error(f"Failed to open file {validated_path}: {e}")
        raise


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename to prevent security issues.

    Args:
        filename: Original filename
        max_length: Maximum allowed filename length

    Returns:
        Sanitized filename

    Raises:
        ValueError: If filename is invalid or too long

    """
    if not filename:
        return "unnamed_file"

    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]

    # Replace dangerous characters but keep dots
    dangerous_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    sanitized = filename

    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "_")

    # Remove control characters
    sanitized = "".join(c for c in sanitized if ord(c) >= 32)

    # Handle Windows reserved names
    windows_reserved = [
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    ]
    name_part = sanitized.split(".")[0] if "." in sanitized else sanitized
    if name_part.upper() in windows_reserved:
        sanitized = "_" + sanitized

    return sanitized.strip()


def validate_image_file(
    file_path: str | Path,
    allowed_extensions: list[str] | None = None,
    max_size_mb: float = 50.0,
) -> Path:
    """Validate image file with security checks.

    Args:
        file_path: Path to image file
        allowed_extensions: list of allowed extensions (e.g., ['.png', '.jpg'])
        max_size_mb: Maximum file size in MB

    Returns:
        Validated Path object

    Raises:
        ValueError: If file is not a valid image
        FileSizeLimitError: If file exceeds size limit

    """
    if allowed_extensions is None:
        allowed_extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".bmp",
            ".dcm",
            ".nii",
            ".nii.gz",
            ".h5",
            ".hdf5",
        ]

    # Allow absolute paths for inference-time validation (e.g., temp files in
    # tests)
    validated_path = validate_file_path(file_path, allow_absolute=True)

    # Check extension properly (handle compound extensions like .nii.gz)
    file_suffix = validated_path.suffix
    if validated_path.name.endswith(".nii.gz"):
        file_suffix = ".nii.gz"

    if file_suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise ValueError(f"Unsupported image format: {file_suffix}")

    # Allow larger files for H5/HDF5 data files (MRI datasets can be large)
    if file_suffix.lower() in [".h5", ".hdf5"]:
        file_max_size_mb = 500.0  # 500MB for H5 data files
    else:
        file_max_size_mb = max_size_mb  # Use provided limit for image files

    check_file_size(validated_path, file_max_size_mb)

    return validated_path


def secure_temp_file(
    suffix: str = "",
    prefix: str = "tmp_",
    dir_path: str | None = None,
):
    """Create a secure temporary file with proper permissions.

    Args:
        suffix: File suffix
        prefix: File prefix
        dir_path: Directory to create file in

    Returns:
        Context manager yielding Path to temporary file

    """
    import tempfile

    @contextmanager
    def _temp_file_manager():
        # Use system temp directory by default
        """_temp_file_manager.

        Returns:
            Any: Description.
        """
        if dir_path is None:
            dir_path_actual = tempfile.gettempdir()
        else:
            dir_path_actual = dir_path

        # Create temp file with restrictive permissions
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir_path_actual)

        # Close the file descriptor (we'll reopen it as needed)
        os.close(fd)

        temp_path = Path(path)

        # Set restrictive permissions (readable/writable by owner only)
        if os.name == "posix":
            temp_path.chmod(0o600)

        try:
            yield temp_path
        finally:
            # Clean up the file
            if temp_path.exists():
                temp_path.unlink()

    return _temp_file_manager()
