"""Unit tests for PhysicsGuidedReverseSampler."""

import pytest
import torch


class TestPhysicsGuidedReverseSampler:
    """Tests for the composable DC-projected reverse sampler."""

    @pytest.fixture
    def make_sampler(self):
        from spectramr.models.diffusion.physics_guided_sampler import (
            PhysicsGuidedReverseSampler,
        )
        def _factory(dc_mode="hard", lambda_dc=1.0, **kwargs):
            return PhysicsGuidedReverseSampler(
                dc_mode=dc_mode, lambda_dc=lambda_dc, **kwargs
            )
        return _factory

    @pytest.fixture
    def kspace_data(self):
        """Create paired k-space test data (interleaved real/imag format)."""
        B, C, H, W = 2, 2, 32, 32  # C=2 for real/imag
        # Create a "measured" k-space with a mask
        measured = torch.randn(B, C, H, W)
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, H // 4 : 3 * H // 4, :] = 1.0  # Central lines sampled
        pred = torch.randn(B, C, H, W)
        return pred, measured, mask

    def test_hard_dc_replaces_sampled_lines(self, make_sampler, kspace_data):
        """Hard DC should replace k-space at sampled locations."""
        sampler = make_sampler(dc_mode="hard")
        pred, measured, mask = kspace_data

        result = sampler(pred, measured, mask, is_kspace_domain=True)

        # At sampled locations, result should match measured
        sampled_result = result * mask
        sampled_measured = measured * mask
        assert torch.allclose(sampled_result, sampled_measured, atol=1e-5), (
            "Hard DC failed: sampled locations differ from measurements"
        )

    def test_hard_dc_preserves_unsampled_lines(self, make_sampler, kspace_data):
        """Hard DC should keep predictions at unsampled locations."""
        sampler = make_sampler(dc_mode="hard", noise_level=0.0)
        pred, measured, mask = kspace_data

        result = sampler(pred, measured, mask, is_kspace_domain=True)

        # At unsampled locations, result should match prediction
        unsampled_mask = 1.0 - mask
        unsampled_result = result * unsampled_mask
        unsampled_pred = pred * unsampled_mask
        assert torch.allclose(unsampled_result, unsampled_pred, atol=1e-5), (
            "Hard DC failed: unsampled locations changed"
        )

    def test_gradient_dc_reduces_residual(self, make_sampler, kspace_data):
        """Gradient DC should reduce k-space residual at sampled locations."""
        sampler = make_sampler(dc_mode="gradient", lambda_dc=1.0)
        pred, measured, mask = kspace_data

        # Compute initial residual
        from spectramr.infrastructure.physics.fft_ops import fft2c

        pred_complex = torch.complex(pred[:, 0:1], pred[:, 1:2])
        meas_complex = torch.complex(measured[:, 0:1], measured[:, 1:2])
        k_pred_init = fft2c(pred_complex)
        init_residual = torch.norm(mask * (k_pred_init - meas_complex))

        result = sampler(pred, measured, mask)
        result_complex = torch.complex(result[:, 0:1], result[:, 1:2])
        k_result = fft2c(result_complex)
        final_residual = torch.norm(mask * (k_result - meas_complex))

        assert final_residual < init_residual, (
            f"Gradient DC failed: residual increased {init_residual:.4f} -> {final_residual:.4f}"
        )

    def test_soft_dc_produces_valid_output(self, make_sampler, kspace_data):
        """Soft DC should produce finite output."""
        sampler = make_sampler(dc_mode="soft", lambda_dc=1.0)
        pred, measured, mask = kspace_data

        result = sampler(pred, measured, mask, is_kspace_domain=True)
        assert torch.isfinite(result).all(), "Soft DC produced non-finite output"
        assert result.shape == pred.shape

    def test_gradient_flow(self, make_sampler, kspace_data):
        """Gradients should flow through the sampler."""
        sampler = make_sampler(dc_mode="gradient", lambda_dc=1.0)
        pred, measured, mask = kspace_data
        pred.requires_grad_(True)

        result = sampler(pred, measured, mask)
        loss = result.sum()
        loss.backward()

        assert pred.grad is not None, "No gradient on predictions"

    def test_kspace_domain_flag(self, make_sampler):
        """is_kspace_domain should skip FFT transform."""
        sampler = make_sampler(dc_mode="hard", noise_level=0.0)
        B, C, H, W = 2, 2, 32, 32
        pred = torch.randn(B, C, H, W)
        measured = torch.randn(B, C, H, W)
        mask = torch.ones(B, 1, H, W)

        result = sampler(pred, measured, mask, is_kspace_domain=True)
        # With full mask and hard DC, result should equal measured
        assert torch.allclose(result, measured, atol=1e-5)

    def test_invalid_mode_raises(self, make_sampler):
        """Invalid DC mode should raise ValueError."""
        with pytest.raises(ValueError):
            make_sampler(dc_mode="invalid_mode")

    def test_learnable_lambda(self, make_sampler):
        """Learnable lambda should be a nn.Parameter."""
        sampler = make_sampler(
            dc_mode="gradient", lambda_dc=0.5, learnable_lambda=True
        )
        params = list(sampler.parameters())
        assert len(params) > 0, "No learnable parameters in gradient mode"
