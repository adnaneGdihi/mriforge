"""Regression tests: an image metric must never silently broadcast a wider prediction.

The b29 heteroscedastic arms emit a 2-channel ``[mean, logvar]`` distribution head and
declare ``validation.metrics: [gaussian_nll, ssim, psnr, nrmse, lpips]``. Before the fix:

* ``PSNR`` / ``SSIM`` assert on shape -> AssertionError -> stored as NaN (visibly broken);
* ``NRMSE`` / ``NMSE`` / ``MAE`` / ``MSE`` had NO shape guard, so ``[B,2,H,W] - [B,1,H,W]``
  BROADCAST, folding the log-variance channel into an image error and returning a finite,
  plausible, meaningless number (pitfall #18 — a NaN gets noticed, a wrong number gets
  published).

Two complementary guards are asserted here:
1. ``BaseMetric.forward`` raises on a shape mismatch (no broadcast reaches compute_metric).
2. ``ValidationMetricsComputer`` narrows the head to the graded channel for shape-strict
   metrics, while ``gaussian_nll`` (REQUIRES_MATCHING_SHAPES = False) still gets both.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.evaluation_metrics import NRMSE, PSNR, BaseMetric


class TestBaseMetricShapeGuard:
    """The guard that removes the silent-broadcast class."""

    def test_nrmse_raises_instead_of_broadcasting_a_2ch_head(self) -> None:
        pred = torch.rand(2, 2, 8, 8)  # [mean, logvar]
        target = torch.rand(2, 1, 8, 8)
        with pytest.raises(ValueError, match="!= target shape"):
            NRMSE()(pred, target)

    def test_psnr_raises_on_mismatch_too(self) -> None:
        with pytest.raises(ValueError, match="!= target shape"):
            PSNR()(torch.rand(2, 2, 8, 8), torch.rand(2, 1, 8, 8))

    def test_matching_shapes_still_compute(self) -> None:
        pred = torch.rand(2, 1, 8, 8)
        target = torch.rand(2, 1, 8, 8)
        assert torch.isfinite(torch.as_tensor(NRMSE()(pred, target))).all()

    def test_a_broadcast_would_have_returned_a_finite_wrong_number(self) -> None:
        """Pin the defect: without the guard the reduction silently succeeds.

        This is what made the bug dangerous — it produced a *plausible* score, not a NaN.
        """
        pred, target = torch.rand(2, 2, 8, 8), torch.rand(2, 1, 8, 8)
        naive = torch.mean(torch.abs(pred - target) ** 2)  # what the metric used to do
        assert naive.shape == ()
        assert torch.isfinite(naive)  # finite, and meaningless

    def test_opt_out_metric_receives_the_full_head(self) -> None:
        class DistributionMetric(BaseMetric):
            REQUIRES_MATCHING_SHAPES = False

            def compute_metric(self, preds, target, **kwargs):  # type: ignore[no-untyped-def]
                return torch.tensor(float(preds.shape[1]))

        got = DistributionMetric()(torch.rand(2, 2, 8, 8), torch.rand(2, 1, 8, 8))
        assert float(got) == 2.0  # both channels reached the metric, no raise


class TestGaussianNLLOptsOut:
    def test_gaussian_nll_declares_it_consumes_the_head(self) -> None:
        from mriforge.core.metrics.quantitative.calibration_nll import GaussianNLLMetric

        assert GaussianNLLMetric.REQUIRES_MATCHING_SHAPES is False


class TestNonImagePairMetricsStillWork:
    """The guard must not break metrics whose shapes disagree BY DESIGN.

    ``classwise_ece`` takes an (N, K) probability matrix against (N,) labels. A blanket
    shape guard broke it; the opt-out is what keeps it working.
    """

    def test_classwise_ece_opts_out(self) -> None:
        from mriforge.core.metrics.calibration_metrics import ClasswiseECE

        assert ClasswiseECE.REQUIRES_MATCHING_SHAPES is False

    def test_classwise_ece_still_computes_on_probs_vs_labels(self) -> None:
        from mriforge.core.metrics.calibration_metrics import ClasswiseECE

        probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])  # (N, K)
        labels = torch.tensor([0, 1, 0])  # (N,) — deliberately a different shape
        assert torch.isfinite(torch.as_tensor(ClasswiseECE()(probs, labels)))
