"""Unit tests for RicianConsistencyLoss."""

import math

import torch

from spectramr.models.losses.rician_consistency_loss import (
    RicianConsistencyLoss,
    _estimate_sigma_mad,
    _rician_unbiased,
)


class TestRicianUnbiasedEstimator:
    """Tests for the _rician_unbiased helper."""

    def test_zero_signal_produces_zero(self):
        """When y² <= 2σ², output should be zero (clamped)."""
        sigma = torch.tensor([[[[0.5]]]])
        # y < sigma√2, so y² < 2σ²
        y = torch.zeros(1, 1, 8, 8)
        result = _rician_unbiased(y, sigma)
        assert result.min() >= 0.0  # clamp(min=0) then sqrt-safe

    def test_high_snr_approximates_identity(self):
        """For high SNR, unbiased estimate ≈ original signal."""
        sigma = torch.tensor([[[[0.001]]]])  # nearly zero noise
        y = torch.ones(1, 1, 8, 8) * 10.0
        result = _rician_unbiased(y, sigma)
        # sqrt(100 - 0.000002) ≈ 10.0
        assert torch.allclose(result, y, atol=0.01)

    def test_correct_formula(self):
        """Verify exact formula: sqrt(y² - 2σ²)."""
        sigma = torch.tensor([[[[1.0]]]])
        y = torch.tensor([[[[3.0, 4.0], [5.0, math.sqrt(2)]]]])
        result = _rician_unbiased(y, sigma)
        expected = torch.sqrt(torch.clamp(y**2 - 2.0, min=0.0) + 1e-12)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_output_shape(self):
        """Output shape matches input shape."""
        sigma = torch.rand(4, 1, 1, 1).abs() + 0.01
        y = torch.rand(4, 2, 64, 64).abs()
        result = _rician_unbiased(y, sigma)
        assert result.shape == y.shape


class TestSigmaMADEstimator:
    """Tests for background MAD noise estimator."""

    def test_known_gaussian_noise(self):
        """MAD/0.6745 should approximate std of Gaussian noise."""
        torch.manual_seed(0)
        sigma_true = 0.05
        image = torch.randn(1, 1, 128, 128).abs() * sigma_true
        # Floor at very low values (background-only image)
        sigma_est = _estimate_sigma_mad(image, background_quantile=0.9)
        # Should be within 50% of true sigma for pure noise image
        assert sigma_est.shape == (1, 1, 1, 1)
        assert (sigma_est > sigma_true * 0.5).all()
        assert (sigma_est < sigma_true * 2.0).all()

    def test_per_sample_estimation(self):
        """Returns per-sample sigma [B,1,1,1]."""
        image = torch.rand(4, 1, 32, 32)
        sigma = _estimate_sigma_mad(image)
        assert sigma.shape == (4, 1, 1, 1)
        assert (sigma > 0).all()


class TestRicianConsistencyLoss:
    """Tests for RicianConsistencyLoss."""

    def test_image_domain_forward_returns_scalar(self):
        """Basic forward pass returns scalar loss."""
        loss_fn = RicianConsistencyLoss(use_fourier_bridge=False)
        pred = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)
        val = loss_fn(pred, target)
        assert val.shape == ()
        assert val.item() >= 0.0

    def test_identical_prediction_gives_low_loss(self):
        """Prediction equal to unbiased target → near-zero loss."""
        loss_fn = RicianConsistencyLoss(
            use_fourier_bridge=False, sigma=0.001, bias_correction=True
        )
        # High-amplitude signal where 2σ² is negligible
        target = torch.ones(1, 1, 64, 64) * 5.0
        # Unbiased target ≈ sqrt(25 - 2*0.001²) ≈ 4.9999...
        unbiased = torch.sqrt(torch.clamp(target**2 - 2 * 0.001**2, min=0.0))
        val = loss_fn(unbiased, target)
        assert val.item() < 0.01

    def test_zero_prediction_gives_positive_loss(self):
        """Zero prediction against non-trivial target → positive loss."""
        loss_fn = RicianConsistencyLoss(use_fourier_bridge=False, sigma=0.01)
        pred = torch.zeros(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32) * 2.0
        val = loss_fn(pred, target)
        assert val.item() > 0.0

    def test_fixed_sigma_vs_auto(self):
        """Fixed sigma and auto-MAD both run without error."""
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)

        fixed = RicianConsistencyLoss(use_fourier_bridge=False, sigma=0.05)
        auto = RicianConsistencyLoss(use_fourier_bridge=False, sigma=None)

        val_fixed = fixed(pred, target)
        val_auto = auto(pred, target)

        assert val_fixed.item() >= 0.0
        assert val_auto.item() >= 0.0

    def test_bias_correction_disabled(self):
        """When bias_correction=False, behaves as plain MAE."""
        loss_fn = RicianConsistencyLoss(use_fourier_bridge=False, bias_correction=False)
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)
        val = loss_fn(pred, target)
        expected = (pred.abs() - target.abs()).abs().mean()
        assert torch.allclose(val, expected, atol=1e-5)

    def test_kspace_forwad_via_registry(self):
        """Loss created via registry with k-space bridge runs correctly."""
        from spectramr.models.losses import create_loss

        loss_fn = create_loss("rician_consistency", use_fourier_bridge=True)
        # 2-channel stacked k-space [B, 2, H, W]
        pred = torch.randn(1, 2, 32, 32)
        target = torch.randn(1, 2, 32, 32)
        val = loss_fn(pred, target)
        assert val.shape == ()
        assert val.item() >= 0.0

    def test_gradients_flow_through_bridge(self):
        """Gradients must propagate through the Fourier bridge."""
        loss_fn = RicianConsistencyLoss(use_fourier_bridge=True)
        pred = torch.randn(1, 2, 32, 32, requires_grad=True)
        target = torch.randn(1, 2, 32, 32)
        val = loss_fn(pred, target)
        val.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0
