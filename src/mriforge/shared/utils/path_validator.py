"""Path validation and sanitization utilities for CLI inputs."""

from pathlib import Path


class PathValidator:
    """Validates and sanitizes file system paths for security."""

    @staticmethod
    def sanitize_path(path_str: str) -> str:
        """Sanitize a path string to prevent directory traversal attacks.

        Args:
            path_str: The path string to sanitize

        Returns:
            Sanitized path string

        Raises:
            ValueError: If path contains dangerous patterns
        """
        if not path_str:
            raise ValueError("Path cannot be empty")

        # Convert to Path object for proper handling
        path = Path(path_str)

        # For relative paths, try to resolve them relative to project root first
        if not path.is_absolute():
            # Try to find the project root by looking for common markers
            # (pyproject.toml, setup.py, .git, etc.)
            current = Path.cwd()

            # First, check if path exists relative to current directory
            if (current / path).exists():
                resolved_path = (current / path).resolve()
            else:
                # Try to find project root
                project_root = None
                check_path = current
                for _ in range(10):  # Limit search depth
                    if (check_path / "pyproject.toml").exists() or (check_path / ".git").exists():
                        project_root = check_path
                        break
                    parent = check_path.parent
                    if parent == check_path:  # Reached filesystem root
                        break
                    check_path = parent

                # If project root found and path exists relative to it, use that
                if project_root and (project_root / path).exists():
                    resolved_path = (project_root / path).resolve()
                else:
                    # Fall back to standard resolution from current directory
                    resolved_path = path.resolve()
        else:
            # For absolute paths, just resolve normally
            resolved_path = path.resolve()

        # Check for directory traversal attempts
        try:
            # Ensure the resolved path doesn't go outside allowed dirs
            resolved_path.relative_to(resolved_path.parent)
        except ValueError:
            # If we can't make it relative, it might be suspicious
            pass

        # Convert back to string
        sanitized = str(resolved_path)

        # Additional security checks
        if ".." in sanitized:
            raise ValueError("Path contains directory traversal (..)")

        if path_str.startswith("/") and not sanitized.startswith("/"):
            raise ValueError("Path manipulation detected")

        return sanitized

    @staticmethod
    def validate_file_path(
        file_path: str,
        must_exist: bool = True,
        allowed_extensions: list[str] | None = None,
    ) -> Path:
        """Validate a file path with security checks.

        Args:
            file_path: Path to validate
            must_exist: Whether the file must exist
            allowed_extensions: List of allowed file extensions

        Returns:
            Validated Path object

        Raises:
            ValueError: If validation fails
            FileNotFoundError: If file must exist but doesn't
        """
        if not file_path:
            raise ValueError("File path cannot be empty")

        # Sanitize the path first
        sanitized_path = PathValidator.sanitize_path(file_path)
        path = Path(sanitized_path)

        # Check file extension if specified
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise ValueError(
                    f"File extension '{path.suffix}' not allowed. Allowed: {allowed_extensions}"
                )

        # Check if file exists (if required)
        if must_exist and not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Additional security: ensure it's actually a file if it exists
        if path.exists() and not path.is_file():
            raise ValueError(f"Path exists but is not a file: {path}")

        return path

    @staticmethod
    def validate_directory_path(
        dir_path: str, must_exist: bool = True, create_if_missing: bool = False
    ) -> Path:
        """Validate a directory path with security checks.

        Args:
            dir_path: Directory path to validate
            must_exist: Whether the directory must exist
            create_if_missing: Whether to create directory if missing

        Returns:
            Validated Path object

        Raises:
            ValueError: If validation fails
            FileNotFoundError: If directory must exist but doesn't
        """
        if not dir_path:
            raise ValueError("Directory path cannot be empty")

        # Sanitize the path first
        sanitized_path = PathValidator.sanitize_path(dir_path)
        path = Path(sanitized_path)

        # Check if directory exists
        if must_exist and not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except (OSError, PermissionError) as e:
                    raise ValueError(f"Cannot create directory {path}: {e}") from e
            else:
                raise FileNotFoundError(f"Directory not found: {path}")

        # Additional security: ensure it's actually a directory if exists
        if path.exists() and not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")

        return path
