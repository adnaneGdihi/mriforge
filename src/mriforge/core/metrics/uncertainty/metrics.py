"""Uncertainty Metrics.

Provides metrics for evaluating the quality and calibration of uncertainty
estimates.

Historically this lived in a sibling module ``core/metrics/uncertainty.py``.
When the ``core/metrics/uncertainty/`` PACKAGE was added (PR-CC physics-residual
conformal track), the package shadowed that module, so
``from mriforge.core.metrics.uncertainty import UncertaintyMetrics`` raised
ImportError and silently broke both ``CampaignEvaluator`` (unimportable) and
``tests/unit/core/metrics/test_uncertainty.py`` (collection error). The class now
lives inside the package and is re-exported from ``__init__`` so that import path
resolves again.
"""

import torch


class UncertaintyMetrics:
    """Metrics for evaluating uncertainty quality."""

    @staticmethod
    def expected_calibration_error(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            uncertainties: Uncertainty estimates
            n_bins: Number of bins for calibration

        Returns:
            ECE value

        """
        # Convert uncertainties to confidence scores
        confidence = 1.0 - uncertainties

        # Bin predictions by confidence
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this confidence bin. Close the final bin on the
            # right so confidence == 1.0 (uncertainty == 0.0) is counted rather
            # than silently dropped through the half-open ``< bin_upper`` test.
            if i == n_bins - 1:
                in_bin = (confidence >= bin_lower) & (confidence <= bin_upper)
            else:
                in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
            if in_bin.sum() == 0:
                continue

            # Accuracy in this bin
            bin_acc = (predictions[in_bin] == targets[in_bin]).float().mean()

            # Average confidence in this bin
            bin_confidence = confidence[in_bin].mean()

            # Contribution to ECE
            bin_weight = in_bin.float().mean()
            ece += bin_weight * abs(bin_acc - bin_confidence)

        return float(ece)

    @staticmethod
    def sharpness(uncertainties: torch.Tensor) -> float:
        """Compute prediction sharpness (lower is sharper).

        Args:
            uncertainties: Uncertainty estimates

        Returns:
            Sharpness value

        """
        return uncertainties.mean().item()

    @staticmethod
    def uncertainty_correlation(
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
    ) -> float:
        """Compute correlation between uncertainty and prediction error.

        Args:
            uncertainties: Uncertainty estimates
            errors: Prediction errors

        Returns:
            Correlation coefficient

        """
        # Flatten tensors
        uncertainties_flat = uncertainties.flatten()
        errors_flat = errors.flatten()

        # Compute correlation
        corr_matrix = torch.corrcoef(torch.stack([uncertainties_flat, errors_flat]))
        return corr_matrix[0, 1].item()
