"""Tests for the Ladder VAE (``hierarchical_vae_ladder``).

Write-only: executed on the cluster, not here.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.vae.ladder_block import LadderBlock
from spectramr.models.vae.hierarchical_vae_ladder import LadderVAE

from ._vae_base import (
    assert_gradient_flows,
    assert_kl_nonneg,
    assert_recon_shape,
    assert_registered,
)


def _model() -> LadderVAE:
    return LadderVAE(in_channels=1, out_channels=1, num_layers=3, latent_dim=16, hidden=32)


def test_registry_resolution():
    assert_registered("hierarchical_vae_ladder", "vae")


def test_constructs_with_no_kwargs():
    LadderVAE()


def test_forward_shape_and_kl():
    model = _model()
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    x_hat, kl = out["reconstruction"], out["kl_override"]
    assert_recon_shape(x_hat, x)
    assert_kl_nonneg(kl)


def test_default_latent_dims_non_increasing():
    model = _model()
    dims = model.latent_dims
    assert all(dims[i] >= dims[i + 1] for i in range(len(dims) - 1))


def test_rejects_increasing_latent_dims():
    with pytest.raises(ValueError):
        LadderVAE(num_layers=3, latent_dims=[4, 8, 16])


def test_rejects_mismatched_latent_dims_length():
    with pytest.raises(ValueError):
        LadderVAE(num_layers=3, latent_dims=[16, 8])


def test_precision_merge_identity():
    """The merged posterior must equal the exact Gaussian-product form."""
    block = LadderBlock(feature_dim=8, latent_dim=4, above_dim=4)
    mu_hat = torch.randn(5, 4)
    logvar_hat = torch.randn(5, 4)
    mu_p = torch.randn(5, 4)
    logvar_p = torch.randn(5, 4)
    mu_q, logvar_q = block.merge(mu_hat, logvar_hat, mu_p, logvar_p)

    prec_hat = (-logvar_hat).exp()
    prec_p = (-logvar_p).exp()
    prec_q = prec_hat + prec_p
    expected_mu = (mu_hat * prec_hat + mu_p * prec_p) / prec_q
    expected_logvar = -prec_q.log()
    assert torch.allclose(mu_q, expected_mu, atol=1e-5)
    assert torch.allclose(logvar_q, expected_logvar, atol=1e-5)


def test_gradient_flow():
    model = _model()
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    x_hat, kl = out["reconstruction"], out["kl_override"]
    loss = torch.nn.functional.mse_loss(x_hat, x) + kl
    assert_gradient_flows(model, loss)
