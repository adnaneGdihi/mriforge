"""Tests for the β-TC-VAE-GAN (``beta_vae_gan``).

Write-only: executed on the cluster, not here.
"""

from __future__ import annotations

import torch

from mriforge.models.generative.beta_vae_gan import BetaVAEGAN

from ._vae_base import assert_gradient_flows, assert_recon_shape, assert_registered


def _model() -> BetaVAEGAN:
    return BetaVAEGAN(in_channels=1, out_channels=1, latent_dim=16, hidden=32)


def test_registry_resolution():
    assert_registered("beta_vae_gan", "gan")


def test_constructs_with_no_kwargs():
    BetaVAEGAN()


def test_forward_returns_tensor_reconstruction():
    model = _model()
    x = torch.randn(4, 1, 32, 32)
    import torch as _t
    assert isinstance(model(x), _t.Tensor)  # generator contract for GANTrainingStrategy
    out = model.forward_full(x)
    for key in (
        "x_hat",
        "mi",
        "tc",
        "dw_kl",
        "beta_tc_loss",
        "d_real",
        "d_fake",
        "adv_loss",
        "total_loss",
    ):
        assert key in out, f"missing output {key}"
    assert_recon_shape(out["x_hat"], x)


def test_discriminator_outputs_finite():
    model = _model()
    x = torch.randn(4, 1, 32, 32)
    out = model.forward_full(x)
    assert torch.isfinite(out["d_real"]).all()
    assert torch.isfinite(out["d_fake"]).all()


def test_has_vae_and_discriminator_submodules():
    model = _model()
    assert hasattr(model, "vae")
    assert hasattr(model, "discriminator")


def test_hinge_losses():
    d_real = torch.randn(4, 1, 8, 8)
    d_fake = torch.randn(4, 1, 8, 8)
    g = BetaVAEGAN.hinge_g_loss(d_fake)
    d = BetaVAEGAN.hinge_d_loss(d_real, d_fake)
    assert g.ndim == 0 and torch.isfinite(g)
    assert d.ndim == 0 and torch.isfinite(d)


def test_total_loss_finite_and_gradient_flow():
    model = _model()
    x = torch.randn(4, 1, 32, 32)
    out = model.forward_full(x)
    assert torch.isfinite(out["total_loss"]).all()
    assert_gradient_flows(model, out["total_loss"])
