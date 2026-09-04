"""Tests for the hyperspherical (vMF) VAE (``hyperspherical_vae``).

Write-only: executed on the cluster, not here.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.vae.von_mises_fisher import VonMisesFisher
from spectramr.models.vae.hyperspherical_vae import HypersphericalVAE

from ._vae_base import (
    assert_gradient_flows,
    assert_kl_nonneg,
    assert_recon_shape,
    assert_registered,
)


def _model() -> HypersphericalVAE:
    return HypersphericalVAE(in_channels=1, out_channels=1, latent_dim=8, hidden=32)


def test_registry_resolution():
    assert_registered("hyperspherical_vae", "vae")


def test_constructs_with_no_kwargs():
    HypersphericalVAE()


def test_rejects_latent_dim_below_two():
    with pytest.raises(ValueError):
        HypersphericalVAE(latent_dim=1)


def test_forward_shape_and_kl():
    model = _model()
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    x_hat, kl = out["reconstruction"], out["kl_override"]
    assert_recon_shape(x_hat, x)
    assert_kl_nonneg(kl)


def test_sample_on_sphere():
    """rsample must land on the unit sphere within tolerance."""
    mu = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    kappa = torch.rand(16) * 10 + 1.0
    z = VonMisesFisher(mu, kappa).rsample()
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_vmf_kl_nonnegative():
    mu = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    kappa = torch.rand(16) * 20 + 1.0
    kl = VonMisesFisher(mu, kappa).kl_uniform()
    assert torch.isfinite(kl).all()
    assert float(kl.min()) >= -1e-3


def test_vmf_rejects_non_unit_mu():
    with pytest.raises(ValueError):
        VonMisesFisher(torch.randn(4, 8) * 3.0, torch.ones(4), validate=True)


def test_gradient_flow_into_mu_and_kappa():
    model = _model()
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    x_hat, kl = out["reconstruction"], out["kl_override"]
    loss = torch.nn.functional.mse_loss(x_hat, x) + kl
    assert_gradient_flows(model, loss)
    assert model.mu_head.weight.grad is not None
    assert model.kappa_head.weight.grad is not None
