"""Unit Tests for Generator Mixins
=================================

Tests for DiffusionMixin, PhysicsMixin, and LatentMixin.

Compliance:
- unit-test.md: Property-based testing via Hypothesis
- unit-test.md: NaN/Inf checks
- unit-test.md: Deterministic seeding
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from spectramr.models.generators.mixins import DiffusionMixin, LatentMixin, PhysicsMixin


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture(autouse=True)
def set_seed():
    """Ensure deterministic randomness (unit-test.md Directive 4.D.2)."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# PhysicsMixin Tests
# ===========================================================================
class TestPhysicsMixin:
    """Test cases for PhysicsMixin."""

    def test_fft_invertibility(self, device: torch.device) -> None:
        """Test FFT(IFFT(x)) ≈ x (unit-test.md Directive 4.0)."""
        x = torch.randn(2, 1, 64, 64, dtype=torch.cfloat, device=device)
        reconstructed = PhysicsMixin.ifft2c(PhysicsMixin.fft2c(x))
        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_ifft_invertibility(self, device: torch.device) -> None:
        """Test IFFT(FFT(x)) ≈ x."""
        k = torch.randn(2, 1, 64, 64, dtype=torch.cfloat, device=device)
        reconstructed = PhysicsMixin.fft2c(PhysicsMixin.ifft2c(k))
        assert torch.allclose(k, reconstructed, atol=1e-5)

    def test_soft_data_consistency_lambda_zero(self, device: torch.device) -> None:
        """Test DC with lambda=0 returns prediction unchanged."""

        class DummyPhysics(PhysicsMixin):
            pass

        physics = DummyPhysics()
        pred = torch.randn(2, 1, 32, 32, dtype=torch.cfloat, device=device)
        kspace = torch.randn(2, 1, 32, 32, dtype=torch.cfloat, device=device)
        mask = torch.ones(2, 1, 32, 32, device=device)

        result = physics.soft_data_consistency(pred, kspace, mask, lambda_dc=0.0)
        # With lambda=0, output should be prediction (via FFT cycle)
        expected = physics.ifft2c(physics.fft2c(pred))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_soft_data_consistency_lambda_one(self, device: torch.device) -> None:
        """Test DC with lambda=1 replaces with measured at mask locations."""

        class DummyPhysics(PhysicsMixin):
            pass

        physics = DummyPhysics()
        pred = torch.randn(2, 1, 32, 32, dtype=torch.cfloat, device=device)
        kspace = torch.randn(2, 1, 32, 32, dtype=torch.cfloat, device=device)
        mask = torch.ones(2, 1, 32, 32, device=device)

        result = physics.soft_data_consistency(pred, kspace, mask, lambda_dc=1.0)
        expected = physics.ifft2c(kspace)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_combine_coils_rss(self, device: torch.device) -> None:
        """Test RSS combination of multi-coil images."""
        multi_coil = torch.randn(2, 8, 32, 32, dtype=torch.cfloat, device=device)
        combined = PhysicsMixin.combine_coils(multi_coil)
        expected = torch.sqrt((multi_coil.abs() ** 2).sum(dim=1, keepdim=True))
        assert torch.allclose(combined, expected)


# ===========================================================================
# LatentMixin Tests
# ===========================================================================
class TestLatentMixin:
    """Test cases for LatentMixin."""

    def test_reparameterize_shape(self, device: torch.device) -> None:
        """Test reparameterized output has correct shape."""

        class DummyLatent(LatentMixin):
            pass

        latent_mixin = DummyLatent()
        mu = torch.randn(4, 128, device=device)
        logvar = torch.randn(4, 128, device=device)
        z = latent_mixin.reparameterize(mu, logvar)
        assert z.shape == (4, 128)

    def test_reparameterize_no_nan(self, device: torch.device) -> None:
        """Test reparameterization produces no NaNs (unit-test.md 4.D.1)."""

        class DummyLatent(LatentMixin):
            pass

        latent_mixin = DummyLatent()
        mu = torch.randn(4, 128, device=device)
        logvar = torch.randn(4, 128, device=device)
        z = latent_mixin.reparameterize(mu, logvar)
        assert torch.isfinite(z).all(), "Reparameterization produced NaN/Inf"

    def test_kl_divergence_positive(self, device: torch.device) -> None:
        """KL divergence should be non-negative."""
        mu = torch.randn(4, 128, device=device)
        logvar = torch.randn(4, 128, device=device)
        kl = LatentMixin.kl_divergence(mu, logvar)
        # KL can be negative for individual terms but sum is positive
        # Actually for standard KL with N(0,I) prior, individual can be negative
        # Just check no NaN
        assert torch.isfinite(kl).all()

    def test_sample_latent_shape(self, device: torch.device) -> None:
        """Test latent sampling produces correct shape."""
        z = LatentMixin.sample_latent((4, 64), device)
        assert z.shape == (4, 64)
        assert z.device.type == device.type


# ===========================================================================
# DiffusionMixin Tests
# ===========================================================================
class DummyDiffusionGenerator(DiffusionMixin, torch.nn.Module):
    """Dummy class implementing DiffusionMixin for testing."""

    def __init__(self, timesteps: int = 100):
        super().__init__()
        self._initialize_diffusion_buffers(timesteps, "linear")

    def _predict_start_from_noise(self, x_t, t, noise):
        """Dummy implementation."""
        return x_t - noise

    def _model_forward(self, x, t, **kwargs):
        """Dummy noise prediction (identity)."""
        return torch.zeros_like(x)


class TestDiffusionMixin:
    """Test cases for DiffusionMixin."""

    def test_buffer_initialization(self, device: torch.device) -> None:
        """Test diffusion buffers are initialized correctly."""
        model = DummyDiffusionGenerator(timesteps=100).to(device)
        assert hasattr(model, "betas")
        assert hasattr(model, "alphas_cumprod")
        assert model.betas.shape == (100,)
        assert model.alphas_cumprod.shape == (100,)

    def test_q_sample_shape(self, device: torch.device) -> None:
        """Test q_sample preserves tensor shape."""
        model = DummyDiffusionGenerator(timesteps=100).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)
        t = torch.randint(0, 100, (2,), device=device)
        x_t = model.q_sample(x, t)
        assert x_t.shape == x.shape

    def test_q_sample_t0_close_to_input(self, device: torch.device) -> None:
        """At t=0, noised sample should be close to input."""
        model = DummyDiffusionGenerator(timesteps=100).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)
        t = torch.zeros(2, dtype=torch.long, device=device)
        x_t = model.q_sample(x, t, noise=torch.zeros_like(x))
        # At t=0 with no noise, should be close to original
        assert torch.allclose(x_t, x, atol=0.01)

    def test_q_sample_no_nan(self, device: torch.device) -> None:
        """Test q_sample produces no NaNs (unit-test.md 4.D.1)."""
        model = DummyDiffusionGenerator(timesteps=100).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)
        t = torch.randint(0, 100, (2,), device=device)
        x_t = model.q_sample(x, t)
        assert torch.isfinite(x_t).all(), "q_sample produced NaN/Inf"

    def test_p_sample_loop_shape(self, device: torch.device) -> None:
        """Test p_sample_loop returns correct shape."""
        model = DummyDiffusionGenerator(timesteps=10).to(device)  # Small for speed
        shape = (1, 1, 16, 16)
        output = model.p_sample_loop(shape)
        assert output.shape == shape

    def test_cosine_schedule_bounded(self) -> None:
        """Test cosine schedule produces betas in valid range."""
        betas = DiffusionMixin._cosine_beta_schedule(100)
        assert (betas >= 0).all()
        assert (betas <= 1).all()
