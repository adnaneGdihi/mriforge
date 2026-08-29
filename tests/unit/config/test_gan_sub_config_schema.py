"""Tests for the ``training.gan`` sub-config (progressive-growing controls).

Added with the Phase-3 ``progressive_gan`` model so its phase schedule is
configurable from YAML (``config.training.gan.phase_schedule``), read by
``ProgressiveGANStrategy``.
"""

from __future__ import annotations

import pytest

from mriforge.config.schemas.training.base import (
    GANSubConfigSchema,
    TrainingStrategyConfigSchema,
)


def test_defaults() -> None:
    g = GANSubConfigSchema()
    assert g.phase_schedule is None
    assert g.progan_gradient_penalty_lambda == 10.0


def test_phase_schedule_list() -> None:
    g = GANSubConfigSchema(phase_schedule=[200, 200, 400])
    assert g.phase_schedule == [200, 200, 400]


def test_negative_gp_lambda_rejected() -> None:
    with pytest.raises(Exception):
        GANSubConfigSchema(progan_gradient_penalty_lambda=-1.0)


def test_training_config_exposes_gan_subblock() -> None:
    """config.training (TrainingStrategyConfigSchema) carries .gan so
    ProgressiveGANStrategy's ``config.training.gan.phase_schedule`` read
    resolves."""
    assert "gan" in TrainingStrategyConfigSchema.model_fields
    cfg = TrainingStrategyConfigSchema(gan={"phase_schedule": [10, 20]})
    assert cfg.gan is not None
    assert cfg.gan.phase_schedule == [10, 20]
