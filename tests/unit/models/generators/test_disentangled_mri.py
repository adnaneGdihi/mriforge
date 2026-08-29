"""Regression tests for ``BiasFieldGenerator`` order dispatch.

Pitfall #9 / CLAUDE.md non-negotiable #3: an unknown ``order`` value must
RAISE at construction time, not silently build no head and crash later with
``UnboundLocalError`` in ``forward`` (where ``bias_field`` is never assigned).

Pre-fix: ``BiasFieldGenerator(order="polynomial3")`` constructs without error,
then ``forward`` raises ``UnboundLocalError``. Post-fix: construction raises a
clear ``ValueError`` naming the offending value and the advertised set.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.constants import BIAS_FIELD_MAX_VALUE, BIAS_FIELD_MIN_VALUE
from mriforge.models.generators.disentangled_mri import BiasFieldGenerator


def test_unknown_order_raises_at_construction() -> None:
    # Deliberate typo: 'polynomial3' (missing underscore) must fail loud.
    with pytest.raises(ValueError, match="Unknown bias_field order"):
        BiasFieldGenerator(in_dim=8, order="polynomial3")


@pytest.mark.parametrize("order", ["polynomial_3", "lowrank_unet"])
def test_valid_orders_construct_and_forward(order: str) -> None:
    torch.manual_seed(0)
    gen = BiasFieldGenerator(in_dim=8, order=order, output_resolution=(16, 16))
    features = torch.randn(2, 8, 16, 16, device="cpu")
    field = gen(features)
    assert field.shape == (2, 1, 16, 16)
    # lowrank_unet path is bounded to [MIN, MAX] exactly; polynomial path is a
    # smoothed tanh-scaled field which stays within the same target range.
    assert torch.isfinite(field).all()
    assert field.min().item() >= BIAS_FIELD_MIN_VALUE - 1e-3
    assert field.max().item() <= BIAS_FIELD_MAX_VALUE + 1e-3
