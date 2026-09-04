"""Utilities to enforce the exclusion of text-conditioned TRELLIS features.

Migrated from src/implementations/trellis/text_guard.py during consolidation.
"""

from __future__ import annotations

from typing import Final

TEXT_FEATURES_SUPPORTED: Final[bool] = False


def ensure_text_features_disabled(feature_path: str | None = None) -> None:
    """Raise if a text-conditioned TRELLIS feature is accessed.

    Args:
        feature_path: Optional identifier for the text feature. Used for
            contextual logging.

    Raises:
        NotImplementedError: Always, because text-conditioned TRELLIS features
            are excluded from the migration scope.
    """
    # NOTE: Text-conditioned TRELLIS features are intentionally out of scope for this project.
    if feature_path is None:
        feature_path = "<unspecified>"

    raise NotImplementedError(
        "TRELLIS text-conditioned features are out of scope for the "
        f"migration. Attempted access: {feature_path}."
    )


__all__ = ["TEXT_FEATURES_SUPPORTED", "ensure_text_features_disabled"]
