"""Tests for the SSDU training sub-schema, its mount, and factory registration."""

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.ssdu import SSDUTrainingConfigSchema


def test_accepts_valid_fields():
    cfg = SSDUTrainingConfigSchema(theta_fraction=0.35, num_masks=4, split_seed=7)
    assert cfg.theta_fraction == 0.35
    assert cfg.num_masks == 4
    assert cfg.split_seed == 7


def test_defaults():
    cfg = SSDUTrainingConfigSchema()
    assert cfg.theta_fraction == 0.4
    assert cfg.num_masks == 1
    assert cfg.split_seed == 0


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(theta_fractions=0.4)  # typo -> extra="forbid"


def test_rejects_out_of_range_fraction():
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(theta_fraction=1.5)
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(theta_fraction=0.0)


def test_ssdu_mounts_on_training_schema():
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "ssdu" in TrainingStrategyConfigSchema.model_fields, (
        "training.ssdu sub-block not mounted on TrainingStrategyConfigSchema"
    )


def test_ssdu_keys_registered_and_resolvable():
    from mriforge.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    for key in ("self_supervised_reconstruction", "ssdu", "robust_ssdu"):
        assert key in paths
        assert paths[key].endswith("ssdu_strategy.SSDUReconstructionStrategy")


# --- Robust SSDU (Noisier2Noise) fields ------------------------------------
def test_robust_ssdu_fields_default_off():
    cfg = SSDUTrainingConfigSchema()
    assert cfg.noisier2noise_correction is False
    assert cfg.noise_std_estimate is None
    assert cfg.noise_model == "gaussian_kspace"
    assert cfg.n_coils == 1


def test_robust_ssdu_accepts_valid_fields():
    cfg = SSDUTrainingConfigSchema(
        noisier2noise_correction=True,
        noise_std_estimate=0.02,
        noise_model="ncchi_magnitude",
        n_coils=4,
    )
    assert cfg.noisier2noise_correction is True
    assert cfg.noise_std_estimate == 0.02
    assert cfg.noise_model == "ncchi_magnitude"
    assert cfg.n_coils == 4


def test_robust_ssdu_requires_noise_std():
    with pytest.raises(ValidationError, match="noise_std_estimate"):
        SSDUTrainingConfigSchema(noisier2noise_correction=True)


def test_robust_ssdu_rejects_nonpositive_noise_std():
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(noisier2noise_correction=True, noise_std_estimate=0.0)


def test_robust_ssdu_rejects_unknown_noise_model():
    with pytest.raises(ValidationError):
        SSDUTrainingConfigSchema(noise_model="poisson")
