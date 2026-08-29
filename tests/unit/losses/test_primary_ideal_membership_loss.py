"""Tests for the primary-ideal membership loss (closes primary_ideal façade)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.primary_ideal_membership_loss import (
    PrimaryIdealMembershipLoss,
)
from mriforge.models.losses.registry import LossRegistry


def test_registered_and_domain():
    assert "primary_ideal_membership" in {
        n.lower() for n in LossRegistry._custom_losses.keys()
    }


def test_contract_properties():
    loss = PrimaryIdealMembershipLoss()
    out = loss(torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8))
    assert isinstance(out, torch.Tensor) and out.ndim == 0


def test_lower_when_prediction_lies_on_component_structure():
    """A prediction that reuses the target's tissue-component structure scores
    lower than a scattered random prediction."""
    torch.manual_seed(0)
    # target has clear cluster structure (two tissue classes).
    a = torch.full((1, 3, 8, 8), 1.0)
    b = torch.full((1, 3, 8, 8), 5.0)
    target = torch.cat([a, b], dim=2)  # [1,3,16,8]
    loss = PrimaryIdealMembershipLoss(n_components=2)
    on_struct = loss(target.clone(), target)
    scattered = loss(torch.rand_like(target) * 6.0, target)
    assert float(on_struct) < float(scattered)


def test_primitive_is_exercised(monkeypatch):
    import mriforge.models.losses.primary_ideal_membership_loss as mod

    called = {"n": 0}
    real = mod.vanishing_ideal_residual

    def spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "vanishing_ideal_residual", spy)
    mod.PrimaryIdealMembershipLoss()(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
    assert called["n"] >= 1


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        PrimaryIdealMembershipLoss()(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 9, 8))
