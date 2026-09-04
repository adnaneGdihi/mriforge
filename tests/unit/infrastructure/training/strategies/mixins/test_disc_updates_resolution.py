"""Regression tests for the disc_updates resolution helper.

Pin the post-audit-05-F10 fix: discriminator-updates-per-G-step lives at
``config.losses.gan.disc_updates`` (v6.0 SSOT). Reading
``config.disc_updates`` is the legacy flat pattern (CLAUDE.md pitfall
#1) and silently falls back to 1 because ``extra="forbid"`` rejects the
unknown top-level field.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    _resolve_disc_updates,
)


def _config_with_disc_updates(n: int) -> SimpleNamespace:
    return SimpleNamespace(losses=SimpleNamespace(gan=SimpleNamespace(disc_updates=n)))


def test_resolve_disc_updates_reads_from_losses_gan() -> None:
    assert _resolve_disc_updates(_config_with_disc_updates(3)) == 3


def test_resolve_disc_updates_returns_int() -> None:
    cfg = SimpleNamespace(losses=SimpleNamespace(gan=SimpleNamespace(disc_updates=5)))
    out = _resolve_disc_updates(cfg)
    assert isinstance(out, int)
    assert out == 5


def test_resolve_disc_updates_raises_when_gan_block_missing() -> None:
    cfg = SimpleNamespace(losses=SimpleNamespace(gan=None))
    with pytest.raises(ValueError, match="config.losses.gan"):
        _resolve_disc_updates(cfg)


def test_resolve_disc_updates_raises_when_losses_missing() -> None:
    cfg = SimpleNamespace()
    with pytest.raises(ValueError, match="config.losses.gan"):
        _resolve_disc_updates(cfg)


def test_resolve_disc_updates_does_not_silently_fall_back_to_1() -> None:
    """CLAUDE.md pitfall #1 + #9: no flat-config fallback, no silent default."""
    cfg = SimpleNamespace(disc_updates=7, losses=SimpleNamespace(gan=None))
    with pytest.raises(ValueError):
        _resolve_disc_updates(cfg)
