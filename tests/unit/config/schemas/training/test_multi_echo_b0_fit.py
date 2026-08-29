"""Tests for ``MultiEchoB0FitConfig`` (audit 2026-07 I1)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.multi_echo_b0_fit import MultiEchoB0FitConfig


def test_t_shift_is_required_and_positive() -> None:
    with pytest.raises(ValidationError):
        MultiEchoB0FitConfig()  # t_shift has no default
    with pytest.raises(ValidationError):
        MultiEchoB0FitConfig(t_shift=0.0)  # gt=0
    cfg = MultiEchoB0FitConfig(t_shift=1e-3)
    assert cfg.t_shift == 1e-3
    assert cfg.gamma > 0.0
    assert cfg.lambda_field == 1.0


def test_rejects_unknown_field() -> None:
    # StrictSchema forbids extras — an unread knob must be rejected (#15).
    with pytest.raises(ValidationError):
        MultiEchoB0FitConfig(t_shift=1e-3, not_a_real_knob=1)


def test_bounds_on_cg_and_gamma() -> None:
    with pytest.raises(ValidationError):
        MultiEchoB0FitConfig(t_shift=1e-3, gamma=0.0)
    with pytest.raises(ValidationError):
        MultiEchoB0FitConfig(t_shift=1e-3, max_cg_iterations=0)
