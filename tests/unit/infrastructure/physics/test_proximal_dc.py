"""Unit tests for ProximalDCStep.

Tests the data consistency projection used at every reverse diffusion
timestep to prevent hallucination in sampled k-space bands.
"""

import pytest
import torch

from mriforge.infrastructure.physics.fft_ops import fft2c
from mriforge.infrastructure.physics.proximal_dc import ProximalDCStep


class TestProximalDCStep:
    """Tests for the proximal data consistency projection."""

    @pytest.fixture
    def simple_setup(self):
        """Create a simple test scenario."""
        B, H, W = 1, 16, 16
        # Ground truth image (complex)
        gt_img = torch.complex(torch.randn(B, 1, H, W), torch.randn(B, 1, H, W))
        # Measured k-space (from ground truth)
        measured_kspace = fft2c(gt_img)
        # Sampling mask (sample center 8 lines)
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, 4:12, :] = 1.0
        return gt_img, measured_kspace, mask

    def test_hard_dc_preserves_sampled_frequencies(self, simple_setup):
        """Sampled frequencies should match measured data exactly."""
        gt_img, measured_kspace, mask = simple_setup
        dc = ProximalDCStep(mode="hard")

        # Create a "wrong" prediction
        wrong_pred = gt_img + torch.complex(
            torch.randn_like(gt_img.real) * 0.5,
            torch.randn_like(gt_img.imag) * 0.5,
        )
        result = dc(wrong_pred, measured_kspace, mask)

        # Check: sampled frequencies in result should match measured
        result_k = fft2c(result)
        diff_sampled = (result_k * mask - measured_kspace * mask).abs()
        assert diff_sampled.max() < 1e-4

    def test_hard_dc_unsampled_from_prediction(self, simple_setup):
        """Unsampled frequencies should come from the prediction."""
        gt_img, measured_kspace, mask = simple_setup
        dc = ProximalDCStep(mode="hard")

        pred = gt_img * 2.0  # Scaled prediction
        result = dc(pred, measured_kspace, mask)

        result_k = fft2c(result)
        pred_k = fft2c(pred)
        inv_mask = 1.0 - mask

        # Unsampled should match prediction k-space
        diff_unsampled = (result_k * inv_mask - pred_k * inv_mask).abs()
        assert diff_unsampled.max() < 1e-4

    def test_soft_dc_matches_hard_at_lambda_1(self, simple_setup):
        """Soft mode with λ=1.0 should match hard mode."""
        gt_img, measured_kspace, mask = simple_setup
        dc_hard = ProximalDCStep(mode="hard")
        dc_soft = ProximalDCStep(mode="soft", soft_lambda=1.0)

        pred = gt_img + torch.complex(
            torch.randn_like(gt_img.real) * 0.1,
            torch.randn_like(gt_img.imag) * 0.1,
        )

        result_hard = dc_hard(pred, measured_kspace, mask)
        result_soft = dc_soft(pred, measured_kspace, mask)

        assert torch.allclose(result_hard, result_soft, atol=1e-5)

    def test_2channel_real_input(self):
        """2-channel real/imag format should produce stacked real output."""
        dc = ProximalDCStep(mode="hard")
        B, H, W = 1, 16, 16

        # Create a complex image and convert to 2-channel stacked format
        img_c = torch.complex(torch.randn(B, 1, H, W), torch.randn(B, 1, H, W))
        pred_ri = torch.cat([img_c.real, img_c.imag], dim=1)  # (B, 2, H, W)

        # Measured k-space must go through the same _to_complex path
        meas_k = fft2c(img_c)  # (B, 1, H, W) complex
        meas_ri = torch.cat([meas_k.real, meas_k.imag], dim=1)  # (B, 2, H, W)

        mask = torch.ones(B, 1, H, W)

        result = dc(pred_ri, meas_ri, mask)
        # Output should be 2-channel stacked real
        assert result.shape == (B, 2, H, W)
        assert not torch.is_complex(result)

    def test_complex_input_returns_complex(self):
        """Complex input should produce complex output."""
        dc = ProximalDCStep(mode="hard")
        pred = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        meas = fft2c(pred)
        mask = torch.ones(1, 1, 8, 8)

        result = dc(pred, meas, mask)
        assert torch.is_complex(result)

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="must be 'hard' or 'soft'"):
            ProximalDCStep(mode="invalid")

    def test_soft_lambda_is_learnable(self):
        """Soft mode λ should be a learnable parameter."""
        dc = ProximalDCStep(mode="soft", soft_lambda=0.5)
        params = list(dc.parameters())
        assert len(params) == 1
        assert params[0].item() == pytest.approx(0.5)

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        dc = ProximalDCStep(mode="hard")
        for shape in [(2, 1, 32, 32), (1, 1, 64, 64)]:
            pred = torch.complex(torch.randn(*shape), torch.randn(*shape))
            meas = fft2c(pred)
            mask = torch.ones(shape[0], 1, shape[2], shape[3])
            result = dc(pred, meas, mask)
            assert result.shape == pred.shape

    def test_soft_lambda_clamped_above_one(self, simple_setup):
        """soft_lambda > 1 is clamped to 1.0, i.e. identical to hard DC.

        Pins the documented clamp contract: the soft-mode docstring now
        states ``λ`` is clamped to ``[0, 1]`` before use, so a soft block
        constructed with ``soft_lambda=2.0`` must behave exactly like hard DC.
        """
        gt_img, measured_kspace, mask = simple_setup
        pred = gt_img + torch.complex(
            torch.randn_like(gt_img.real) * 0.1,
            torch.randn_like(gt_img.imag) * 0.1,
        )
        dc_hard = ProximalDCStep(mode="hard")
        dc_over = ProximalDCStep(mode="soft", soft_lambda=2.0)
        assert torch.allclose(
            dc_hard(pred, measured_kspace, mask),
            dc_over(pred, measured_kspace, mask),
            atol=1e-5,
        )
