"""Tests for ``EMAConfigSchema``.

Targets ``mriforge.config.schemas.ema``. Pydantic v2 schema with custom
validators: ``decay >= 0.5``, ``final_decay >= initial_decay``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.ema import EMAConfigSchema


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_ema_disabled() -> None:
    """Default state: EMA off."""
    cfg = EMAConfigSchema()
    assert cfg.enabled is False
    assert cfg.decay == 0.999
    assert cfg.update_frequency == 1


def test_explicit_construction() -> None:
    """Explicit values are persisted."""
    cfg = EMAConfigSchema(enabled=True, decay=0.95, update_frequency=10)
    assert cfg.enabled is True
    assert cfg.decay == 0.95
    assert cfg.update_frequency == 10


# ---------------------------------------------------------------------------
# decay validator
# ---------------------------------------------------------------------------


def test_decay_below_half_raises() -> None:
    """``decay < 0.5`` is rejected as unstable."""
    with pytest.raises(ValidationError, match="decay < 0.5 is likely unstable"):
        EMAConfigSchema(decay=0.4)


def test_decay_above_one_raises() -> None:
    """``decay > 1`` violates ``le=1`` constraint."""
    with pytest.raises(ValidationError):
        EMAConfigSchema(decay=1.1)


def test_decay_below_zero_raises() -> None:
    """``decay < 0`` violates ``ge=0`` constraint."""
    with pytest.raises(ValidationError):
        EMAConfigSchema(decay=-0.1)


def test_decay_at_boundary_accepted() -> None:
    """``decay == 0.5`` (lower bound) and ``decay == 1.0`` (upper) both accepted."""
    EMAConfigSchema(decay=0.5)
    EMAConfigSchema(decay=1.0)


# ---------------------------------------------------------------------------
# Adaptive EMA — final_decay >= initial_decay
# ---------------------------------------------------------------------------


def test_final_decay_below_initial_raises() -> None:
    """``final_decay < initial_decay`` → ``ValueError``."""
    with pytest.raises(ValidationError, match="final_decay must be >= initial_decay"):
        EMAConfigSchema(initial_decay=0.95, final_decay=0.5)


def test_final_decay_equal_to_initial_accepted() -> None:
    """``final_decay == initial_decay`` is allowed (constant decay)."""
    cfg = EMAConfigSchema(initial_decay=0.9, final_decay=0.9)
    assert cfg.initial_decay == cfg.final_decay


# ---------------------------------------------------------------------------
# Constraints on auxiliary fields
# ---------------------------------------------------------------------------


def test_warmup_steps_non_negative() -> None:
    """``warmup_steps >= 0``."""
    with pytest.raises(ValidationError):
        EMAConfigSchema(warmup_steps=-1)


def test_update_frequency_at_least_one() -> None:
    """``update_frequency >= 1``."""
    with pytest.raises(ValidationError):
        EMAConfigSchema(update_frequency=0)


def test_stability_threshold_non_negative() -> None:
    """``stability_threshold >= 0``."""
    with pytest.raises(ValidationError):
        EMAConfigSchema(stability_threshold=-0.001)


# ---------------------------------------------------------------------------
# Frozen + extra='forbid'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """Schema is immutable after construction."""
    cfg = EMAConfigSchema()
    with pytest.raises(ValidationError):
        cfg.decay = 0.5


def test_extra_fields_rejected() -> None:
    """Legacy aliases like ``ema_decay`` are explicitly forbidden."""
    with pytest.raises(ValidationError, match="extra"):
        EMAConfigSchema(ema_decay=0.999)  # legacy alias
