"""Regression tests for ``infrastructure/security/file_validation``.

The original ``validate_file_path`` used a *name-membership* check
(``root in p.parts``) which is not a containment check: any absolute path that
merely happens to contain a directory literally named ``experiments`` /
``checkpoints`` / ``artifacts`` was accepted (e.g. ``/etc/experiments/passwd``),
and ``..`` traversal into such a directory was never rejected. A security
validator that silently accepts an untrusted path is worse than none — it
manufactures false confidence (CLAUDE.md pitfall #9: no silent fallbacks).

These tests pin the corrected containment + traversal behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mriforge.infrastructure.security.file_validation import validate_file_path


def test_rejects_absolute_path_with_trusted_named_component() -> None:
    """``/etc/experiments/passwd`` has ``experiments`` in its parts but is NOT
    under the project tree — the name-membership bug accepted it; containment
    must reject it."""
    with pytest.raises(ValueError):
        validate_file_path("/etc/experiments/passwd")


def test_rejects_traversal_into_a_trusted_dir() -> None:
    """A ``..`` sequence that lands inside some other ``experiments`` directory
    must be rejected before resolution (the membership check accepted it because
    the resolved path still contained ``experiments``)."""
    with pytest.raises(ValueError):
        validate_file_path("experiments/../../../tmp/experiments/evil.txt")


def test_accepts_legitimate_relative_trusted_path() -> None:
    """A genuine project-relative path under a trusted root still resolves."""
    resolved = validate_file_path("experiments/inprogress/x.yaml")
    assert isinstance(resolved, Path)
    assert resolved.is_relative_to(Path.cwd().resolve())
    assert "experiments" in resolved.parts


def test_rejects_path_outside_every_trusted_root() -> None:
    """A project-relative path that is not under any trusted root is rejected."""
    with pytest.raises(ValueError):
        validate_file_path("src/mriforge/main.py")
