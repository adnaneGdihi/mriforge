"""Tests for ``ensure_text_features_disabled``.

Targets ``spectramr.shared.utils.text_guard``. TRELLIS text-conditioned
features are intentionally out of scope for this codebase — accessing
them must always raise ``NotImplementedError``.
"""

from __future__ import annotations

import pytest

from spectramr.shared.utils.text_guard import (
    TEXT_FEATURES_SUPPORTED,
    ensure_text_features_disabled,
)


def test_text_features_constant_is_false() -> None:
    """``TEXT_FEATURES_SUPPORTED`` is hard-coded to False."""
    assert TEXT_FEATURES_SUPPORTED is False


def test_ensure_text_features_disabled_raises() -> None:
    """The guard always raises ``NotImplementedError``."""
    with pytest.raises(NotImplementedError, match="out of scope"):
        ensure_text_features_disabled()


def test_ensure_text_features_includes_feature_path_in_message() -> None:
    """When ``feature_path`` is given, it appears in the error message."""
    with pytest.raises(NotImplementedError, match="text_clip_encoder"):
        ensure_text_features_disabled("text_clip_encoder")


def test_ensure_text_features_default_path_unspecified() -> None:
    """Without ``feature_path`` the message says ``<unspecified>``."""
    with pytest.raises(NotImplementedError, match="<unspecified>"):
        ensure_text_features_disabled(None)
