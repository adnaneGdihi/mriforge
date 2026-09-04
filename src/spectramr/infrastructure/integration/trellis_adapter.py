"""TRELLIS Integration Utilities
================================

Helpers to lazily import the upstream TRELLIS reference implementation.
These utilities make sure that the vendored ``TRELLIS-main`` repository
is on ``sys.path`` before attempting imports. They provide consistent
error handling so higher level services can surface actionable messages
when optional TRELLIS dependencies are missing.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType


class TrellisImportError(ImportError):
    """Specialised import error for TRELLIS integration."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        """__init__.

        Args:
            message (str): Description.
            hint (str | None): Description.
        """
        if hint:
            message = f"{message}\nHint: {hint}"
        super().__init__(message)


def _default_trellis_root() -> Path:
    """_default_trellis_root.

    Returns:
        Path: Description.
    """
    return Path(__file__).resolve().parents[3] / "TRELLIS-main"


def ensure_trellis_path(base_path: Path | None = None) -> Path:
    """Ensure the TRELLIS repository is importable.

    Args:
        base_path: Optional explicit path to the TRELLIS repository.

    Returns:
        Path to the TRELLIS repository that has been added to ``sys.path``.

    Raises:
        TrellisImportError: If the repository cannot be located.
    """

    root = Path(base_path) if base_path else _default_trellis_root()
    trellis_pkg = root / "trellis"
    if not trellis_pkg.exists():
        hint = (
            "Expected to find a 'trellis' package inside the 'TRELLIS-main' "
            "directory. Make sure the submodule or extraction is present via "
            "git submodule update --init or by copying the upstream release."
        )
        raise TrellisImportError("Unable to locate TRELLIS repository.", hint=hint)

    import sys

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def import_trellis_module(module: str, base_path: Path | None = None) -> ModuleType:
    """Import a TRELLIS module with consistent error handling."""

    ensure_trellis_path(base_path)
    try:
        return __import__(module, fromlist=["__name__"])
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise TrellisImportError(
            f"Failed to import TRELLIS module '{module}'.", hint=str(exc)
        ) from exc
