"""Correctness tests for the β-TC-VAE-GAN cross-paradigm composition.

The plan (phase3 §12) states beta_vae_gan exists "primarily as a
cross-paradigm composition test: the composer must combine beta_tc_vae, an
adversarial term, and a reconstruction term cleanly". These tests assert the
composite is exactly that sum, finite, and differentiable, and that the
generator exposes both a tensor ``forward`` (the GAN-strategy contract) and a
scalar ``beta_tc_term`` (the aux objective BetaVAEGANStrategy adds to G).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.generative.beta_vae_gan import BetaVAEGAN  # noqa: E402


def _model() -> BetaVAEGAN:
    return BetaVAEGAN(latent_dim=8, hidden=32, pooled_size=4)


def test_composition_is_exact_sum() -> None:
    torch.manual_seed(0)
    m = _model().train()
    x = torch.randn(4, 1, 32, 32)
    out = m.forward_full(x)
    recomposed = out["beta_tc_loss"] + m.lambda_adv * out["adv_loss"] + out["recon_loss"]
    assert torch.isfinite(out["total_loss"])
    assert torch.isclose(out["total_loss"], recomposed, atol=1e-5), (
        "total_loss must equal beta_tc + lambda_adv*adv + recon"
    )


def test_tc_term_is_finite_and_positive_pressure() -> None:
    torch.manual_seed(0)
    out = _model().forward_full(torch.randn(4, 1, 32, 32))
    for key in ("mi", "tc", "dw_kl", "beta_tc_loss", "adv_loss", "recon_loss"):
        assert torch.isfinite(out[key]).all(), f"{key} non-finite"


def test_forward_returns_tensor_for_gan_strategy() -> None:
    m = _model()
    y = m(torch.randn(2, 1, 32, 32))
    assert isinstance(y, torch.Tensor)
    assert y.shape[-2:] == (32, 32)


def test_beta_tc_term_is_differentiable_scalar() -> None:
    torch.manual_seed(0)
    m = _model()
    bt = m.beta_tc_term(torch.randn(4, 1, 32, 32))
    assert bt.ndim == 0 and torch.isfinite(bt)
    bt.backward()
    assert m.vae.fc_mu.weight.grad is not None, "β-TC gradient must reach the encoder"


def test_composition_is_differentiable_end_to_end() -> None:
    torch.manual_seed(0)
    m = _model().train()
    out = m.forward_full(torch.randn(4, 1, 32, 32))
    out["total_loss"].backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert any(grads), "composite total_loss produced no gradients"
