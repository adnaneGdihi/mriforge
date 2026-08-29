"""Tests for cross-field translation losses (MICCAI MRIxFields2026)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.cross_field_losses import (
    CocycleConsistencyLoss,
    FieldFlowVelocityLoss,
    FieldIdentityLoss,
    LatentCycleLoss,
)
from mriforge.models.losses.registry import LossRegistry


def test_zero_when_consistent() -> None:
    loss = LatentCycleLoss()
    q = torch.rand(2, 8, 4, 4)
    assert float(loss(prediction=q, target=q)) == 0.0


def test_positive_when_inconsistent() -> None:
    loss = LatentCycleLoss()
    q = torch.rand(2, 8, 4, 4)
    assert float(loss(prediction=q + 1.0, target=q)) > 0.0


def test_weight_scales() -> None:
    q = torch.rand(2, 8, 4, 4)
    base = float(LatentCycleLoss(weight=1.0)(prediction=q + 1.0, target=q))
    scaled = float(LatentCycleLoss(weight=2.0)(prediction=q + 1.0, target=q))
    assert abs(scaled - 2.0 * base) < 1e-5


def test_registered() -> None:
    assert "latent_cycle" in LossRegistry._custom_losses


def test_field_flow_velocity_zero_when_matched() -> None:
    loss = FieldFlowVelocityLoss()
    u = torch.rand(2, 1, 8, 8)
    assert float(loss(prediction=u, target=u)) == 0.0


def test_field_flow_velocity_positive_when_mismatched() -> None:
    loss = FieldFlowVelocityLoss()
    u = torch.rand(2, 1, 8, 8)
    assert float(loss(prediction=u + 0.5, target=u)) > 0.0


def test_field_flow_velocity_norm_validates() -> None:
    with pytest.raises(ValueError):
        FieldFlowVelocityLoss(norm="l3")


def test_field_flow_velocity_registered() -> None:
    assert "field_flow_velocity" in LossRegistry._custom_losses


def test_losses_reachable_via_framework_create_loss() -> None:
    # Regression for the adversarial-review finding: a @register_loss is silently
    # dead unless losses/__init__.py imports its module. Assert the FRAMEWORK path
    # (create_loss, used by LossBuilder at build_container) resolves all three —
    # not just direct-import membership. Importing the package triggers the
    # curated-list import that must include cross_field_losses.
    from mriforge.models.losses import create_loss

    assert isinstance(create_loss("latent_cycle"), LatentCycleLoss)
    assert isinstance(create_loss("field_flow_velocity"), FieldFlowVelocityLoss)
    assert isinstance(create_loss("field_flow"), FieldFlowVelocityLoss)  # alias


# --- idea 4.2 cocycle-consistent unified operator losses ---------------------


def test_cocycle_zero_on_exact_factorization() -> None:
    """Theorem 6 / Corollary (5): a family that factorises through one canonicaliser
    has exactly zero cocycle residual. Here composite == direct by construction."""
    loss = CocycleConsistencyLoss()
    x_direct = torch.rand(2, 1, 16, 16)
    assert float(loss(prediction=x_direct, target=x_direct)) == 0.0


def test_cocycle_positive_on_perturbed_family() -> None:
    loss = CocycleConsistencyLoss()
    x_direct = torch.rand(2, 1, 16, 16)
    composite = x_direct + 0.1 * torch.randn_like(x_direct)
    assert float(loss(prediction=composite, target=x_direct)) > 0.0


def test_field_identity_zero_when_reproduced() -> None:
    loss = FieldIdentityLoss()
    x = torch.rand(2, 1, 8, 8)
    assert float(loss(prediction=x, target=x)) == 0.0


def test_field_identity_positive_when_not() -> None:
    loss = FieldIdentityLoss()
    x = torch.rand(2, 1, 8, 8)
    assert float(loss(prediction=x + 0.3, target=x)) > 0.0


def test_cocycle_identity_reachable_via_create_loss() -> None:
    from mriforge.models.losses import create_loss

    assert isinstance(create_loss("cocycle_consistency"), CocycleConsistencyLoss)
    assert isinstance(create_loss("field_identity"), FieldIdentityLoss)
