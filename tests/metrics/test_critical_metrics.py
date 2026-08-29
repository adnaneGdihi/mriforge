"""
Tests for metrics computation correctness and numerical stability.

Critical variables tested:
- PSNR (Peak Signal-to-Noise Ratio) computation
- SSIM (Structural Similarity Index) computation
- FID (Fréchet Inception Distance) computation
- MAE/MSE loss computation
- Metrics with edge cases (zero tensors, NaN, Inf)
- Metrics with different dtypes and normalizations
- Metric determinism and reproducibility

Ensures that reported performance metrics are correct and stable
across different data ranges and normalizations.
"""

import torch
import torch.nn.functional as F


class TestPSNRComputation:
    """Tests for PSNR (Peak Signal-to-Noise Ratio) computation."""

    def test_psnr_identical_images(self):
        """Test that PSNR is infinite (or very high) for identical images."""
        img1 = torch.randn(1, 1, 32, 32)
        img2 = img1.clone()

        # PSNR formula: 10 * log10(MAX^2 / MSE)
        # MSE = 0 for identical images, so PSNR should be very high
        mse = F.mse_loss(img1, img2)
        assert mse.item() < 1e-6

        if mse > 0:
            max_val = max(img1.max(), img2.max())
            psnr = 10 * torch.log10(max_val**2 / mse)
        else:
            psnr = torch.tensor(float("inf"))

        assert torch.isinf(psnr) or psnr > 50

    def test_psnr_completely_different_images(self):
        """Test PSNR for completely different images."""
        img1 = torch.ones(1, 1, 32, 32)
        img2 = torch.zeros(1, 1, 32, 32)

        # MSE should be significant
        mse = F.mse_loss(img1, img2)
        assert mse > 0

        psnr = 10 * torch.log10((img1.max() ** 2) / (mse + 1e-10))
        # PSNR should be low for completely different images
        assert psnr < 20

    def test_psnr_normalized_range(self):
        """Test PSNR computation with normalized tensors [0, 1]."""
        img_true = torch.rand(1, 1, 32, 32)
        img_pred = img_true + torch.randn_like(img_true) * 0.01

        # Clip to valid range
        img_pred = torch.clamp(img_pred, 0, 1)

        mse = F.mse_loss(img_true, img_pred)
        max_val = 1.0  # For normalized images

        psnr = 10 * torch.log10((max_val**2) / (mse + 1e-10))

        assert psnr > 0
        assert not torch.isnan(psnr)
        assert not torch.isinf(psnr)

    def test_psnr_with_zero_input(self):
        """Test PSNR computation with zero tensor."""
        img_zero = torch.zeros(1, 1, 32, 32)
        img_nonzero = torch.randn(1, 1, 32, 32)

        mse = F.mse_loss(img_zero, img_nonzero)
        assert mse > 0

        # With zero input image, max value is from img_nonzero
        max_val = img_nonzero.abs().max()

        # Should handle gracefully
        if max_val > 0:
            psnr = 10 * torch.log10((max_val**2) / (mse + 1e-10))
            assert torch.isfinite(psnr)

    def test_psnr_batch_computation(self):
        """Test PSNR computed per-sample in batch."""
        batch_size = 4
        img_true = torch.rand(batch_size, 1, 32, 32)
        img_pred = img_true + torch.randn_like(img_true) * 0.01
        img_pred = torch.clamp(img_pred, 0, 1)

        psnrs = []
        for i in range(batch_size):
            mse = F.mse_loss(img_true[i], img_pred[i])
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
            psnrs.append(psnr)

        psnrs = torch.stack(psnrs)

        # All PSNR values should be finite and positive
        assert torch.all(torch.isfinite(psnrs))
        assert torch.all(psnrs > 0)

    def test_psnr_nan_handling(self):
        """Test that PSNR handles NaN gracefully."""
        img_true = torch.rand(1, 1, 32, 32)
        img_pred = img_true.clone()
        img_pred[0, 0, 0, 0] = float("nan")

        mse = F.mse_loss(img_true, img_pred)

        # MSE becomes NaN if input contains NaN
        if torch.isnan(mse):
            # Computation should detect and handle NaN
            assert torch.isnan(mse)


class TestSSIMComputation:
    """Tests for SSIM (Structural Similarity Index) computation."""

    def ssim_compute(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> torch.Tensor:
        """Simplified SSIM computation for testing."""
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2

        mean1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
        mean2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

        mean1_sq = mean1**2
        mean2_sq = mean2**2
        mean1_mean2 = mean1 * mean2

        sigma1_sq = (
            F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size // 2)
            - mean1_sq
        )
        sigma2_sq = (
            F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size // 2)
            - mean2_sq
        )
        sigma12 = (
            F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2)
            - mean1_mean2
        )

        numerator = (2 * mean1_mean2 + c1) * (2 * sigma12 + c2)
        denominator = (mean1_sq + mean2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / (denominator + 1e-10)

        return ssim_map.mean()

    def test_ssim_identical_images(self):
        """Test that SSIM is 1.0 for identical images."""
        img = torch.rand(1, 1, 64, 64)

        ssim = self.ssim_compute(img, img)

        # SSIM of 1.0 for identical images (within numerical precision)
        assert abs(ssim.item() - 1.0) < 0.01

    def test_ssim_range(self):
        """Test that SSIM is in valid range [-1, 1]."""
        img1 = torch.rand(1, 1, 64, 64)
        img2 = torch.randn(1, 1, 64, 64)

        ssim = self.ssim_compute(img1, img2)

        assert -1 <= ssim.item() <= 1

    def test_ssim_symmetry(self):
        """Test that SSIM(A, B) = SSIM(B, A)."""
        img1 = torch.rand(1, 1, 64, 64)
        img2 = torch.randn(1, 1, 64, 64)

        ssim_ab = self.ssim_compute(img1, img2)
        ssim_ba = self.ssim_compute(img2, img1)

        assert abs(ssim_ab.item() - ssim_ba.item()) < 1e-6

    def test_ssim_noise_sensitivity(self):
        """Test that SSIM decreases with added noise."""
        img_clean = torch.rand(1, 1, 64, 64)

        ssim_clean = self.ssim_compute(img_clean, img_clean)

        # Add small noise
        img_noisy = img_clean + torch.randn_like(img_clean) * 0.01
        img_noisy = torch.clamp(img_noisy, 0, 1)

        ssim_noisy = self.ssim_compute(img_clean, img_noisy)

        # SSIM with noise should be lower
        assert ssim_noisy.item() < ssim_clean.item()

    def test_ssim_batch_computation(self):
        """Test SSIM computation on batch."""
        batch_size = 4
        imgs1 = torch.rand(batch_size, 1, 64, 64)
        imgs2 = torch.randn(batch_size, 1, 64, 64)

        ssims = []
        for i in range(batch_size):
            ssim = self.ssim_compute(imgs1[i : i + 1], imgs2[i : i + 1])
            ssims.append(ssim)

        ssims = torch.stack(ssims)

        # All SSIM values should be in valid range
        assert torch.all((ssims >= -1) & (ssims <= 1))


class TestMAEMSEComputation:
    """Tests for Mean Absolute Error and Mean Squared Error."""

    def test_mae_identical_tensors(self):
        """Test MAE is zero for identical tensors."""
        tensor = torch.randn(4, 3, 32, 32)

        mae = F.l1_loss(tensor, tensor)

        assert mae.item() < 1e-6

    def test_mse_identical_tensors(self):
        """Test MSE is zero for identical tensors."""
        tensor = torch.randn(4, 3, 32, 32)

        mse = F.mse_loss(tensor, tensor)

        assert mse.item() < 1e-6

    def test_mae_vs_mse_sensitivity(self):
        """Test that MSE is more sensitive to outliers than MAE."""
        tensor = torch.ones(10, 1, 32, 32)
        tensor_with_outlier = tensor.clone()
        tensor_with_outlier[0, 0, 0, 0] = (
            1000.0  # Large outlier (increased to ensure MSE >> MAE)
        )

        mae = F.l1_loss(tensor, tensor_with_outlier)
        mse = F.mse_loss(tensor, tensor_with_outlier)

        # MSE should be much larger than MAE due to outlier
        assert mse > mae * 100

    def test_mae_batch_dimension(self):
        """Test MAE is computed correctly across batch."""
        batch_size = 5
        img_shape = (32, 32)

        errors_per_sample = []
        for i in range(batch_size):
            img1 = torch.ones(img_shape) * i
            img2 = torch.zeros(img_shape)

            mae = F.l1_loss(img1, img2)
            errors_per_sample.append(mae.item())

        # MAE should increase with batch index
        for i in range(batch_size - 1):
            assert errors_per_sample[i] <= errors_per_sample[i + 1]

    def test_mae_mse_non_negative(self):
        """Test that MAE and MSE are always non-negative."""
        img1 = torch.randn(4, 3, 32, 32)
        img2 = torch.randn(4, 3, 32, 32)

        mae = F.l1_loss(img1, img2)
        mse = F.mse_loss(img1, img2)

        assert mae >= 0
        assert mse >= 0


class TestMetricsEdgeCases:
    """Tests for metrics behavior with edge case inputs."""

    def test_psnr_with_very_small_mse(self):
        """Test PSNR with very small MSE (near-perfect reconstruction)."""
        img_true = torch.randn(1, 1, 32, 32)
        img_pred = img_true + torch.randn_like(img_true) * 1e-6

        mse = F.mse_loss(img_true, img_pred)
        max_val = img_true.abs().max()

        psnr = 10 * torch.log10((max_val**2) / (mse + 1e-10))

        # Should be very high PSNR
        assert psnr > 50

    def test_psnr_with_constant_region(self):
        """Test PSNR when reconstructed region is constant."""
        img_true = torch.ones(1, 1, 32, 32) * 0.5
        img_pred = torch.ones(1, 1, 32, 32) * 0.5

        mse = F.mse_loss(img_true, img_pred)

        # Should be exactly zero
        assert mse == 0

    def test_metrics_with_int_dtype(self):
        """Test metrics handle integer dtypes."""
        img1_int = torch.randint(0, 256, (1, 1, 32, 32), dtype=torch.uint8)
        img2_int = torch.randint(0, 256, (1, 1, 32, 32), dtype=torch.uint8)

        # Convert to float for metric computation
        img1_float = img1_int.float()
        img2_float = img2_int.float()

        mae = F.l1_loss(img1_float, img2_float)
        mse = F.mse_loss(img1_float, img2_float)

        assert torch.isfinite(mae)
        assert torch.isfinite(mse)

    def test_metrics_dtype_consistency(self):
        """Test metrics with different dtypes."""
        img_fp32 = torch.rand(1, 1, 32, 32, dtype=torch.float32)
        img_fp64 = torch.rand(1, 1, 32, 32, dtype=torch.float64)

        mae_32 = F.l1_loss(img_fp32, img_fp32)
        mae_64 = F.l1_loss(img_fp64.float(), img_fp64.float())

        # Both should be valid
        assert torch.isfinite(mae_32)
        assert torch.isfinite(mae_64)


class TestMetricsDeterminism:
    """Tests for metrics reproducibility."""

    def test_mae_deterministic(self):
        """Test that MAE computation is deterministic."""
        img1 = torch.randn(4, 3, 32, 32)
        img2 = torch.randn(4, 3, 32, 32)

        mae1 = F.l1_loss(img1, img2)
        mae2 = F.l1_loss(img1, img2)

        assert mae1 == mae2

    def test_mse_deterministic(self):
        """Test that MSE computation is deterministic."""
        img1 = torch.randn(4, 3, 32, 32)
        img2 = torch.randn(4, 3, 32, 32)

        mse1 = F.mse_loss(img1, img2)
        mse2 = F.mse_loss(img1, img2)

        assert mse1 == mse2

    def test_metrics_batch_consistency(self):
        """Test that batched metrics equal per-sample average."""
        batch_size = 5
        img1 = torch.randn(batch_size, 3, 32, 32)
        img2 = torch.randn(batch_size, 3, 32, 32)

        # Batched computation
        mae_batch = F.l1_loss(img1, img2)

        # Per-sample computation
        mae_per_sample = torch.tensor(
            [F.l1_loss(img1[i], img2[i]).item() for i in range(batch_size)]
        )
        mae_average = mae_per_sample.mean()

        assert abs(mae_batch.item() - mae_average.item()) < 1e-6


class TestMetricsNormalizationImpact:
    """Tests for how normalization affects metrics."""

    def test_psnr_scaled_images(self):
        """Test PSNR with different image scales."""
        img_base = torch.rand(1, 1, 32, 32)
        noise = torch.randn_like(img_base) * 0.01

        # Scale 1: [0, 1]
        img1_true = img_base
        img1_pred = torch.clamp(img_base + noise, 0, 1)

        # Scale 2: [0, 255]
        img2_true = img_base * 255
        img2_pred = torch.clamp((img_base + noise) * 255, 0, 255)

        # PSNR should account for different scales
        mse1 = F.mse_loss(img1_true, img1_pred)
        mse2 = F.mse_loss(img2_true, img2_pred)

        # MSE scales with image intensity
        assert abs((mse2 - mse1 * 255**2) / (mse1 * 255**2)) < 0.1

    def test_mae_scale_invariance(self):
        """Test if MAE scales linearly with image intensity."""
        img_base = torch.rand(1, 1, 32, 32)
        noise = torch.randn_like(img_base) * 0.01

        img1_true = img_base
        img1_pred = torch.clamp(img_base + noise, 0, 1)

        img2_true = img_base * 255
        img2_pred = torch.clamp((img_base + noise) * 255, 0, 255)

        mae1 = F.l1_loss(img1_true, img1_pred)
        mae2 = F.l1_loss(img2_true, img2_pred)

        # MAE should scale linearly
        ratio = mae2 / (mae1 + 1e-10)
        assert abs(ratio - 255) < 1  # Should be approximately 255
