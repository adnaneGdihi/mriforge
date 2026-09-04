"""Centralized path resolution for cross-environment portability.

Resolves data paths between local development and cluster environments
using a single rule: relative paths are joined with the auto-detected
project root; absolute paths are used as-is.

The project root is determined by (in priority order):
    1. ``PROJECT_ROOT`` environment variable
    2. ``SPECTRAMR_DATA_ROOT`` environment variable
    3. Walking up from this file to find ``pyproject.toml``

On a cluster, set ``PROJECT_ROOT`` (or ``SPECTRAMR_DATA_ROOT``) to your data
directory. Locally, auto-detection finds the repo root automatically.
"""

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume CWD is the project root
    return Path.cwd()


def get_project_root() -> Path:
    """Return the project root, checking PROJECT_ROOT / SPECTRAMR_DATA_ROOT first."""
    env_root = os.environ.get("PROJECT_ROOT") or os.environ.get("SPECTRAMR_DATA_ROOT")
    if env_root:
        return Path(env_root)
    return _find_project_root()


# Known absolute prefixes from legacy configs that should be stripped.
# Extend via the SPECTRAMR_LEGACY_ABS_PREFIXES env var (colon-separated) if you
# have your own cluster mount roots to handle.
_LEGACY_ABS_PREFIXES = [
    p for p in os.environ.get("SPECTRAMR_LEGACY_ABS_PREFIXES", "").split(":") if p
]


class PathResolver:
    """Centralized service for resolving paths between local and cluster environments.

    Single rule: relative paths are joined with project root.
    Legacy absolute paths from old configs are normalized to relative first.
    """

    @staticmethod
    def resolve(path: str) -> str:
        """Resolve a data path to an absolute path that exists on this machine.

        Args:
            path: A relative or absolute path string from a manifest or YAML config.

        Returns:
            The resolved absolute path as a string.

        Resolution strategy:
            1. If the path exists as-is (relative to CWD or absolute) → return it
            2. If relative, join with project root and check → return if exists
            3. If absolute with a known legacy prefix, strip to relative and retry
            4. Return the best candidate (may not exist if data is not on this machine)
        """
        if not path:
            return path

        p = Path(path)

        # 1. Already exists as-is
        if p.exists():
            return str(p.resolve()) if p.is_absolute() else str(p)

        project_root = get_project_root()

        # 2. Relative path → join with project root
        if not p.is_absolute():
            joined = project_root / p
            if joined.exists():
                return str(joined)
            # Return joined even if it doesn't exist (best candidate)
            return str(joined)

        # 3. Absolute path with legacy prefix → strip and retry as relative
        path_str = str(p)
        for prefix in _LEGACY_ABS_PREFIXES:
            if path_str.startswith(prefix):
                relative = path_str[len(prefix) :]
                joined = project_root / relative
                if joined.exists():
                    return str(joined)
                return str(joined)

        # 4. Unknown absolute path — return as-is
        return path
