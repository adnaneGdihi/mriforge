"""Tests for FieldBridgeConfig (B-3.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.field_bridge import FieldBridgeConfig


def test_defaults() -> None:
    c = FieldBridgeConfig()
    assert c.sigma_bridge >= 0.0
    assert c.velocity_norm in ("l1", "l2")
    assert c.lambda_endpoint_l1 == 0.0  # off by default (pure velocity matching)


def test_lambda_endpoint_l1_accepted_and_frozen() -> None:
    c = FieldBridgeConfig(lambda_endpoint_l1=0.5)
    assert c.lambda_endpoint_l1 == 0.5
    with pytest.raises(ValidationError):
        c.lambda_endpoint_l1 = 1.0  # frozen schema


def test_lambda_endpoint_l1_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        FieldBridgeConfig(lambda_endpoint_l1=-0.1)


def test_mounted_on_training_schema() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "field_bridge" in TrainingStrategyConfigSchema.model_fields
