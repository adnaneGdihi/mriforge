"""Regression: the SE(3)-equivariance weight must be a validated config field.

2026-05-31 audit: SE3EquivariantNavigatorStrategy read lambda_se3_equivariance
from losses.reconstruction (an extra="ignore" block with no such field), so the
YAML value was silently dropped and the equivariance objective ran at weight 0.
The knob now lives on the extra="forbid" SE3NavigatorConfig where it is read +
validated.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.se3_navigator import SE3NavigatorConfig


def test_lambda_se3_equivariance_is_a_validated_field():
    c = SE3NavigatorConfig(lambda_se3_equivariance=0.5)
    assert c.lambda_se3_equivariance == 0.5


def test_lambda_se3_equivariance_rejects_negative():
    with pytest.raises(ValidationError):
        SE3NavigatorConfig(lambda_se3_equivariance=-1.0)  # ge=0
