"""Tests for RecoverabilityVIBConfig (B-2.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.recoverability_vib import RecoverabilityVIBConfig


def test_defaults() -> None:
    c = RecoverabilityVIBConfig()
    assert c.beta >= 0.0
    assert c.beta_warmup_iters == 0  # off by default (immediate full beta)


def test_beta_warmup_iters_accepted_and_frozen() -> None:
    c = RecoverabilityVIBConfig(beta_warmup_iters=5000)
    assert c.beta_warmup_iters == 5000
    with pytest.raises(ValidationError):
        c.beta_warmup_iters = 1  # frozen schema


def test_beta_warmup_iters_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        RecoverabilityVIBConfig(beta_warmup_iters=-1)


def test_free_bits_defaults_off_and_accepts_positive() -> None:
    assert RecoverabilityVIBConfig().free_bits == 0.0  # disabled by default
    assert RecoverabilityVIBConfig(free_bits=0.3).free_bits == 0.3


def test_free_bits_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        RecoverabilityVIBConfig(free_bits=-0.1)


def test_mounted_on_training_schema() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "recoverability_vib" in TrainingStrategyConfigSchema.model_fields


def test_rate_reduction_defaults_to_legacy_sum() -> None:
    # Default must NOT change silently: existing arms keep the summed rate (and thus
    # their existing effective beta) until they opt in per-arm.
    assert RecoverabilityVIBConfig().rate_reduction == "sum"


def test_rate_reduction_accepts_mean_per_dim() -> None:
    assert RecoverabilityVIBConfig(rate_reduction="mean_per_dim").rate_reduction == "mean_per_dim"


def test_rate_reduction_rejects_unimplemented_value() -> None:
    # Anti-#9/#15: an unadvertised reduction must RAISE at config load, never degrade
    # to the legacy sum (which would silently restore the 158,704x beta inflation).
    with pytest.raises(ValidationError):
        RecoverabilityVIBConfig(rate_reduction="per_channel")


def test_lambda_recon_is_retired_in_favour_of_the_image_loss_entry() -> None:
    """The VIB read its reconstruction weight here while losses.image_losses documented
    another value; the loss-weight table owns it now (mrixfields review 2026-09-03)."""
    from spectramr.config.schemas.renames import RENAMES

    assert "lambda_recon" not in RecoverabilityVIBConfig.model_fields
    assert RENAMES["training.recoverability_vib.lambda_recon"].posture == "raise"
