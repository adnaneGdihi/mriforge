"""Tests for DoobBridgeConfig (B-1.10 7T-marginal-score SDEdit bridge)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.doob_bridge import DoobBridgeConfig


def test_defaults() -> None:
    c = DoobBridgeConfig()
    assert c.timesteps == 1000
    assert 0.0 < c.strength <= 1.0
    assert c.h_scale >= 0.0
    assert (
        c.anchor_scale == 0
    )  # ILVR source anchor OFF by default -> bare bridge (back-compat)
    assert (
        c.residual is False
    )  # unconditional 7T-marginal bridge by default (back-compat)


def test_residual_knob() -> None:
    # residual=true selects the source-conditioned residual reparameterisation (needs the
    # doob_residual_score_unet). Default False keeps every existing arm on the unconditional bridge.
    assert DoobBridgeConfig(residual=True).residual is True
    assert DoobBridgeConfig().residual is False


def test_anchor_scale_knob_validates() -> None:
    # The ILVR structural anchor knob: >=0, bounded; an out-of-range value RAISES (#9/#15).
    assert DoobBridgeConfig(anchor_scale=8).anchor_scale == 8
    with pytest.raises(ValidationError):
        DoobBridgeConfig(anchor_scale=-1)
    with pytest.raises(ValidationError):
        DoobBridgeConfig(anchor_scale=128)  # le=64


def test_mounted_on_training_schema() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "doob_bridge" in TrainingStrategyConfigSchema.model_fields
