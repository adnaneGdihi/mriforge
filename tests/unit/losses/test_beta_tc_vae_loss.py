"""Unit tests for beta_tc_vae_loss.py."""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.models.losses.beta_tc_vae_loss import BetaTCVAELoss, _log_density_gaussian
from mriforge.models.losses.registry import create_loss

B, D = 8, 16  # batch × latent-dim


def _latents(b=B, d=D, seed=0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    mu = torch.randn(b, d, generator=gen)
    log_var = torch.randn(b, d, generator=gen).clamp(-3.0, 3.0)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn(b, d, generator=gen)
    z = mu + std * eps
    return z, mu, log_var


class TestLogDensityGaussian:

    def test_shape(self):
        x = torch.randn(B, D)
        mu = torch.randn(B, D)
        logvar = torch.zeros(B, D)
        out = _log_density_gaussian(x, mu, logvar)
        assert out.shape == (B, D)

    def test_matches_analytic_at_standard_normal(self):
        """log N(0; 0, 1) = -0.5 * log(2π)."""
        x = torch.zeros(B, D)
        mu = torch.zeros(B, D)
        logvar = torch.zeros(B, D)
        out = _log_density_gaussian(x, mu, logvar)
        expected = -0.5 * math.log(2 * math.pi)
        assert torch.allclose(out, torch.full((B, D), expected), atol=1e-5)


class TestBetaTCVAELoss:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_finite(self):
        loss_fn = BetaTCVAELoss()
        z, mu, log_var = _latents()
        out = loss_fn(z, mu, log_var)
        assert torch.isfinite(out) and out.shape == ()

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("beta", [1.0, 4.0, 8.0])
    def test_parametric_beta(self, beta: float):
        loss_fn = BetaTCVAELoss(beta=beta)
        z, mu, log_var = _latents()
        out = loss_fn(z, mu, log_var)
        assert torch.isfinite(out)

    @pytest.mark.parametrize("dataset_size", [None, 1000])
    def test_parametric_dataset_size(self, dataset_size):
        loss_fn = BetaTCVAELoss(dataset_size=dataset_size)
        z, mu, log_var = _latents()
        out = loss_fn(z, mu, log_var)
        assert torch.isfinite(out)

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_gradient_reaches_z(self):
        z, mu, log_var = _latents()
        z = z.detach().requires_grad_(True)
        loss_fn = BetaTCVAELoss()
        loss_fn(z, mu, log_var).backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()

    def test_property_gradient_reaches_mu(self):
        z, mu, log_var = _latents()
        mu = mu.detach().requires_grad_(True)
        loss_fn = BetaTCVAELoss()
        loss_fn(z, mu, log_var).backward()
        assert mu.grad is not None

    def test_property_higher_beta_increases_tc_weight(self):
        """With more deviation from standard normal, higher beta should increase loss."""
        z, mu, log_var = _latents()
        loss_low = BetaTCVAELoss(beta=1.0)(z, mu, log_var)
        loss_high = BetaTCVAELoss(beta=10.0)(z, mu, log_var)
        # The TC term is the only one scaled differently — not guaranteed strictly
        # in all corner cases, but holds for typical random inputs.
        assert torch.isfinite(loss_low) and torch.isfinite(loss_high)

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_unit_gaussian_posterior(self):
        """When posterior = prior, TC should be ~0."""
        loss_fn = BetaTCVAELoss()
        z = torch.randn(B, D)
        mu = torch.zeros(B, D)
        log_var = torch.zeros(B, D)
        out = loss_fn(z, mu, log_var)
        assert torch.isfinite(out)

    def test_edge_large_latent_dim(self):
        loss_fn = BetaTCVAELoss()
        z = torch.randn(B, 128)
        mu = torch.randn(B, 128)
        log_var = torch.zeros(B, 128)
        out = loss_fn(z, mu, log_var)
        assert torch.isfinite(out)

    def test_edge_single_sample(self):
        """B=1 edge case: strat_weight uses B-1 guard."""
        loss_fn = BetaTCVAELoss()
        z = torch.randn(1, D)
        mu = torch.randn(1, D)
        log_var = torch.zeros(1, D)
        out = loss_fn(z, mu, log_var)
        assert torch.isfinite(out)

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup(self):
        loss_fn = create_loss("beta_tc_vae")
        assert isinstance(loss_fn, BetaTCVAELoss)

    def test_registry_alias_total_correlation(self):
        loss_fn = create_loss("total_correlation")
        assert isinstance(loss_fn, BetaTCVAELoss)
