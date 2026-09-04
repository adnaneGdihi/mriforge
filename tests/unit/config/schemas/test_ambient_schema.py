"""Tests for the Ambient Diffusion sub-schema, its mount, and registration."""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.ambient import AmbientTrainingConfigSchema


def test_defaults():
    cfg = AmbientTrainingConfigSchema()
    assert cfg.theta_fraction == 0.4
    assert cfg.ambient_weight == 1.0
    assert cfg.denoise_weight == 1.0
    assert cfg.split_seed == 0


def test_accepts_valid_fields():
    cfg = AmbientTrainingConfigSchema(
        theta_fraction=0.3, ambient_weight=0.5, denoise_weight=2.0, split_seed=7
    )
    assert cfg.theta_fraction == 0.3
    assert cfg.ambient_weight == 0.5


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        AmbientTrainingConfigSchema(theta_fractions=0.4)  # typo -> extra="forbid"


def test_rejects_out_of_range_theta():
    with pytest.raises(ValidationError):
        AmbientTrainingConfigSchema(theta_fraction=1.5)
    with pytest.raises(ValidationError):
        AmbientTrainingConfigSchema(theta_fraction=0.0)


def test_mounts_on_training_schema():
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "ambient" in TrainingStrategyConfigSchema.model_fields


def test_ambient_key_registered_and_resolvable():
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "ambient_diffusion" in paths
    assert paths["ambient_diffusion"].endswith(
        "ambient_diffusion_strategy.AmbientDiffusionStrategy"
    )
