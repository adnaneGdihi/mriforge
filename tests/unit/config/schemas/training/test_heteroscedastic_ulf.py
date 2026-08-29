"""Tests for HeteroscedasticULFConfig (B-2.9)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.heteroscedastic_ulf import HeteroscedasticULFConfig


def test_defaults_and_frozen() -> None:
    c = HeteroscedasticULFConfig()
    assert c.sigma_floor > 0.0
    assert c.lambda_var_prior >= 0.0
    assert 0.0 < c.max_sigma <= 1.0
    assert c.lambda_mean >= 0.0
    assert c.logvar_max > c.logvar_min
    with pytest.raises(ValidationError):
        c.sigma_floor = 1.0


def test_logvar_band_validator_raises_on_inverted_bounds() -> None:
    # The anti-degeneracy logvar clamp must be a non-empty band (max > min).
    with pytest.raises(ValueError, match="logvar_max"):
        HeteroscedasticULFConfig(logvar_min=2.0, logvar_max=-1.0)


def test_guard_knobs_accepted() -> None:
    c = HeteroscedasticULFConfig(lambda_mean=0.5, logvar_min=-10.0, logvar_max=2.0)
    assert c.lambda_mean == 0.5
    assert (c.logvar_min, c.logvar_max) == (-10.0, 2.0)


def test_mounted_on_training_schema() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "heteroscedastic_ulf" in TrainingStrategyConfigSchema.model_fields
