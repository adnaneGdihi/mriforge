"""Intraclass Correlation Coefficient (ICC) Metric.

Based on Cluster.md Section 5.1.
Used for evaluating "Radiomic Stability".
Measures reliability of features across test-retest (or scanner-scanner) domains.
"""

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

# Note: Typically ICC is computed on scalar radiomic features extracted from images.
# Here we implement ICC(3, 1) or ICC(2, 1) - One-way random effects or two-way mixed.
# Reference: Shrout & Fleiss (1979).


class ICCMetric:
    """Computes Intraclass Correlation Coefficient."""

    def __init__(self, mode: str = "oneway"):
        """__init__.

        Args:
            mode (str): Description.
        """
        self.mode = mode

    def calculate(self, measurements: np.ndarray | Float[Tensor, "N K"]) -> float:
        """
        Args:
            measurements: [N, K] matrix where N is subjects/samples, K is raters/scanners.

        Returns:
            icc: scalar value between 0 and 1.
        """
        if isinstance(measurements, torch.Tensor):
            measurements = measurements.detach().cpu().numpy()

        N, K = measurements.shape
        # ICC is undefined for a single subject or a single rater: the
        # mean-square denominators below divide by (N - 1) and (K - 1). Raise
        # rather than return a silent NaN/inf.
        if N < 2 or K < 2:
            raise ValueError(f"ICC requires N>=2 subjects and K>=2 raters, got N={N}, K={K}.")

        # Mean Square Between samples (rows)
        # Mean Square Within samples

        # Total Mean
        grand_mean = np.mean(measurements)

        # Row Means (Subject Means)
        row_means = np.mean(measurements, axis=1)

        # Col Means (Scanner Means) - Only for Two-Way
        col_means = np.mean(measurements, axis=0)

        # Sum of Squares Between Subjects (Rows)
        SS_rows = K * np.sum((row_means - grand_mean) ** 2)
        MS_rows = SS_rows / (N - 1)

        # Sum of Squares Total
        SS_total = np.sum((measurements - grand_mean) ** 2)

        if self.mode == "oneway":
            # Model 1: One-way random effects ANOVA
            # Each subject is rated by a different set of K raters (randomly chosen).
            # Usually applicable if scanners are effectively random instances.

            SS_within = np.sum((measurements - row_means[:, None]) ** 2)
            MS_within = SS_within / (N * (K - 1))

            icc = (MS_rows - MS_within) / (MS_rows + (K - 1) * MS_within)

        elif self.mode == "twoway":
            # Model 3: Two-way mixed effects ANOVA
            # Each subject is rated by the SAME set of K raters (scanners).
            # This is correct for paired data across K scanners.
            # ICC(3, 1) - Consistency

            # Sum of Squares Columns (Raters/Scanners)
            SS_cols = N * np.sum((col_means - grand_mean) ** 2)
            MS_cols = SS_cols / (K - 1)

            # Sum of Squares Residual (Interaction)
            SS_residual = SS_total - SS_rows - SS_cols
            MS_residual = SS_residual / ((N - 1) * (K - 1))

            # ICC(3, 1) formulation
            icc = (MS_rows - MS_residual) / (MS_rows + (K - 1) * MS_residual)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return float(icc)

    def __call__(self, measurements):
        """__call__.

        Args:
            measurements (Any): Description.
        Returns:
            Any: Description.
        """
        return self.calculate(measurements)
