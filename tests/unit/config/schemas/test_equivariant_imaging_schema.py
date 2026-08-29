"""Tests for the Equivariant Imaging sub-schema, its mount, and registration."""

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.equivariant_imaging import (
    TrainingConfigEquivariantImaging,
)


def test_accepts_valid_fields():
    cfg = TrainingConfigEquivariantImaging(
        group="small_rotation",
        alpha_equivariance=0.5,
        n_group_samples=2,
        max_angle_deg=6.0,
        num_angles=3,
    )
    assert cfg.group == "small_rotation"
    assert cfg.alpha_equivariance == 0.5
    assert cfg.n_group_samples == 2
    assert cfg.max_angle_deg == 6.0


def test_defaults():
    cfg = TrainingConfigEquivariantImaging()
    assert cfg.group == "dihedral"
    assert cfg.alpha_equivariance == 1.0
    assert cfg.n_group_samples == 1
    assert cfg.robust_correction is False
    assert cfg.noise_model == "gaussian_kspace"
    assert cfg.n_coils == 1


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TrainingConfigEquivariantImaging(grup="dihedral")  # typo -> extra="forbid"


def test_rejects_unknown_group():
    with pytest.raises(ValidationError):
        TrainingConfigEquivariantImaging(group="so2_full")


def test_robust_correction_requires_noise_std():
    """pitfall #15: an advertised robust correction with no σ must fail at load."""
    with pytest.raises(ValidationError, match="noise_std_estimate"):
        TrainingConfigEquivariantImaging(robust_correction=True)


def test_robust_correction_with_noise_std_ok():
    cfg = TrainingConfigEquivariantImaging(
        robust_correction=True, noise_std_estimate=0.01, noise_model="ncchi_magnitude", n_coils=4
    )
    assert cfg.robust_correction is True
    assert cfg.noise_std_estimate == 0.01
    assert cfg.noise_model == "ncchi_magnitude"


def test_rejects_nonpositive_noise_std():
    with pytest.raises(ValidationError):
        TrainingConfigEquivariantImaging(robust_correction=True, noise_std_estimate=0.0)


def test_mounts_on_training_schema():
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "equivariant_imaging" in TrainingStrategyConfigSchema.model_fields, (
        "training.equivariant_imaging sub-block not mounted on TrainingStrategyConfigSchema"
    )


def test_ei_keys_registered_and_resolvable():
    from mriforge.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    for key in ("equivariant_imaging", "robust_ei"):
        assert key in paths
        assert paths[key].endswith(
            "equivariant_imaging_strategy.EquivariantImagingStrategy"
        )
