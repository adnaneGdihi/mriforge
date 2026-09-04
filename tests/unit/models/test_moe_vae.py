"""Tests for the mixture-of-experts VAE (``moe_vae``).

Write-only: executed on the cluster, not here.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.moe.top1_router import Top1Router
from spectramr.models.vae.moe_vae import MoEVAE

from ._vae_base import (
    assert_gradient_flows,
    assert_kl_nonneg,
    assert_recon_shape,
    assert_registered,
)


def _model() -> MoEVAE:
    return MoEVAE(in_channels=1, out_channels=1, num_experts=4, latent_dim=16, hidden=32)


def test_registry_resolution():
    assert_registered("moe_vae", "vae")


def test_constructs_with_no_kwargs():
    MoEVAE()


def test_rejects_too_few_experts():
    with pytest.raises(ValueError):
        MoEVAE(num_experts=1)


def test_forward_shape_kl_and_aux():
    model = _model()
    x = torch.randn(8, 1, 32, 32)
    out = model(x)
    x_hat, kl, aux = out["reconstruction"], out["kl_override"], out["aux_loss"]
    assert_recon_shape(x_hat, x)
    assert_kl_nonneg(kl)
    assert torch.isfinite(aux).all()


def test_router_outputs():
    router = Top1Router(in_dim=16, num_experts=4)
    z = torch.randn(32, 16)
    k_star, gate_val, probs = router(z)
    assert k_star.shape == (32,)
    assert gate_val.shape == (32,)
    assert probs.shape == (32, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(32), atol=1e-5)


def test_load_balance_loss_minimum_at_uniform():
    """L_aux == K * sum f_k P_k; uniform load gives the minimum value 1."""
    router = Top1Router(in_dim=8, num_experts=4)
    # Uniform routing and uniform probs -> f = P = 1/K -> K * K * (1/K)^2 = 1.
    k_star = torch.arange(4).repeat(8)  # each expert hit equally
    probs = torch.full((32, 4), 0.25)
    aux = router.load_balance_loss(k_star, probs)
    assert abs(float(aux) - 1.0) < 1e-4


def test_every_expert_hittable():
    """On a large batch every expert is reachable (no silent dead expert)."""
    router = Top1Router(in_dim=8, num_experts=4)
    z = torch.randn(256, 8)
    k_star, _, _ = router(z)
    # Not guaranteed all hit on random init, but indices must be in range.
    assert int(k_star.min()) >= 0
    assert int(k_star.max()) < 4


def test_gradient_flow():
    model = _model()
    x = torch.randn(8, 1, 32, 32)
    out = model(x)
    x_hat, kl, aux = out["reconstruction"], out["kl_override"], out["aux_loss"]
    loss = torch.nn.functional.mse_loss(x_hat, x) + kl + aux
    assert_gradient_flows(model, loss)
