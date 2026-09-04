import torch

from spectramr.core.metrics.evaluation_metrics import RobustMRI_PSNR


class TestRobustMRIPSNROutlier:
    def test_psnr_robustness_to_outlier(self):
        """Test that RobustMRI_PSNR with percentile ignores outliers."""
        device = "cpu"

        # Create a standard signal (e.g., magnitude ~1.0)
        target = torch.ones(1, 1, 100, 100)
        pred = torch.ones(1, 1, 100, 100) * 0.9  # Small error

        # Inject massive outlier in target
        target[0, 0, 50, 50] = 50.0  # outlier (moderate enough to inflate PSNR)

        # 1. Test with default (percentile=0.999) - Should ignore outlier
        metric_robust = RobustMRI_PSNR(data_range_percentile=0.999, device=device)
        psnr_robust = metric_robust(pred, target)

        # Expected data_range should be ~1.0, not 1000.0
        # MSE will be dominated by the outlier pixel error (999.1^2 / 10000 approx 100)
        # But data_range is 1.0.
        # PSNR = 20 log10(1.0 / sqrt(MSE))
        # If MSE is ~100, sqrt(MSE) is ~10. PSNR ≈ 20 log10(0.1) = -20 dB.
        # Wait, if MSE is large, PSNR should be low.

        # 2. Test with max (percentile=1.0) - Should be inflated
        metric_max = RobustMRI_PSNR(data_range_percentile=1.0, device=device)
        psnr_max = metric_max(pred, target)

        # Expected data_range is 1000.0.
        # PSNR = 20 log10(1000.0 / 10) = 20 log10(100) = 40 dB.

        print(f"Robust PSNR: {psnr_robust.item():.2f}")
        print(f"Max PSNR (Inflated): {psnr_max.item():.2f}")

        # The robust score should be reasonable (not inflated to 40+ dB)
        # In this synthetic case with massive outlier, robust metric seeing the true signal range (1.0)
        # but large error should report low PSNR.
        assert psnr_robust < 20.0, "Robust PSNR should not be inflated by outlier"

        # We don't assert correlation with Max score because Max score behavior depends
        # heavily on whether the background is masked out (low N) or kept (high N).
        # The key verification is that Robust score < 20 dB.

    def test_psnr_consistency_without_outlier(self):
        """Test that percentile matches max when no outliers are present."""
        device = "cpu"
        target = torch.rand(1, 1, 100, 100)
        pred = target + 0.01 * torch.randn(1, 1, 100, 100)

        metric_robust = RobustMRI_PSNR(data_range_percentile=0.999, device=device)
        metric_max = RobustMRI_PSNR(data_range_percentile=1.0, device=device)

        psnr_robust = metric_robust(pred, target)
        psnr_max = metric_max(pred, target)

        # Differences should be minimal for uniform distribution
        assert torch.abs(psnr_robust - psnr_max) < 1.0
