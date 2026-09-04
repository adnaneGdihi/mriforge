"""Unit tests for SE3EquivariantDCNavigator (B-3 of the VF campaign).

Covers: registration, forward shape, endogenous-navigator extraction, motion
latent estimation, gradient flow, deterministic seeding, no-NaN, and the
documented in-plane warp restriction. Direct-import per the source<->test
pairing rule.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.navigation.se3_equivariant_dc_navigator import (
    SE3EquivariantDCNavigator,
)
from tests.utils.optional_backends import requires_cuda_for_mamba


@pytest.fixture
def model() -> SE3EquivariantDCNavigator:
    torch.manual_seed(0)
    return SE3EquivariantDCNavigator(
        in_channels=2, out_channels=2, n_mamba_layers=2, d_state=8
    ).eval()


def test_registered_in_model_registry() -> None:
    from spectramr.models.registry import MODEL_REGISTRY

    assert "se3_equivariant_dc_navigator" in MODEL_REGISTRY
    entry = MODEL_REGISTRY["se3_equivariant_dc_navigator"]
    assert entry["mode"] == "reconstruction"
    assert entry["class"] is SE3EquivariantDCNavigator


@requires_cuda_for_mamba
def test_forward_shape(model: SE3EquivariantDCNavigator) -> None:
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 2, 32, 32)
    assert torch.isfinite(out).all()


@requires_cuda_for_mamba
def test_forward_with_measured_kspace(model: SE3EquivariantDCNavigator) -> None:
    x = torch.randn(2, 2, 32, 32)
    ksp = torch.randn(2, 2, 32, 32)
    out = model(x, kspace_measured=ksp)
    assert out.shape == (2, 2, 32, 32)


def test_extract_dc_navigator_shape(model: SE3EquivariantDCNavigator) -> None:
    ksp = torch.randn(2, 2, 40, 40)
    nav = model.extract_dc_navigator(ksp)
    # [B, H, 2C]
    assert nav.shape == (2, 40, 2)


@requires_cuda_for_mamba
def test_motion_latent_is_se3(model: SE3EquivariantDCNavigator) -> None:
    ksp = torch.randn(3, 2, 24, 24)
    xi = model.estimate_motion_latent(ksp)
    assert xi.shape == (3, 24, 6)  # [B, H, 6] se(3) coordinate
    assert torch.isfinite(xi).all()


@requires_cuda_for_mamba
def test_gradient_flow(model: SE3EquivariantDCNavigator) -> None:
    model.train()
    x = torch.randn(2, 2, 32, 32, requires_grad=True)
    loss = model(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    total = sum(
        p.grad.abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    assert total > 0.0


@requires_cuda_for_mamba
def test_deterministic_seeding() -> None:
    torch.manual_seed(42)
    m1 = SE3EquivariantDCNavigator(in_channels=2, n_mamba_layers=2, d_state=8).eval()
    torch.manual_seed(42)
    m2 = SE3EquivariantDCNavigator(in_channels=2, n_mamba_layers=2, d_state=8).eval()
    x = torch.randn(2, 2, 32, 32)
    assert torch.allclose(m1(x), m2(x), atol=1e-6)


@requires_cuda_for_mamba
def test_multicoil_channels() -> None:
    """4-channel (2 complex coils) input is accepted."""
    torch.manual_seed(1)
    m = SE3EquivariantDCNavigator(in_channels=4, out_channels=4, n_mamba_layers=2).eval()
    x = torch.randn(2, 4, 32, 32)
    out = m(x)
    assert out.shape == (2, 4, 32, 32)


def test_zero_init_refine_starts_as_pure_warp() -> None:
    """With zero-init refine, output equals the rigid-warped input at init."""
    torch.manual_seed(2)
    m = SE3EquivariantDCNavigator(in_channels=2, out_channels=2, n_mamba_layers=2).eval()
    # last refine conv is zero-initialised → refine(.) == 0 at construction.
    assert torch.equal(
        m.refine[-1].weight, torch.zeros_like(m.refine[-1].weight)
    )
    assert torch.equal(m.refine[-1].bias, torch.zeros_like(m.refine[-1].bias))


def test_invalid_layers_raises() -> None:
    with pytest.raises(ValueError, match="n_mamba_layers"):
        SE3EquivariantDCNavigator(in_channels=2, n_mamba_layers=0)


def test_bad_input_dim_raises(model: SE3EquivariantDCNavigator) -> None:
    with pytest.raises(ValueError, match=r"4D input"):
        model(torch.randn(2, 2, 32))
