"""Frechet Radiomic Distance (MedFID) Metric.

Calculates the Frechet Distance between two distributions of features.
"""

import logging

import numpy as np
import torch.nn as nn
from jaxtyping import Float
from scipy import linalg
from torch import Tensor

logger = logging.getLogger(__name__)


class FrdMetric(nn.Module):
    """Calculates Frechet Radiomic Distance (connected to FID)."""

    def __init__(self):
        """__init__."""
        super().__init__()

    def forward(
        self, real_features: Float[Tensor, "N D"], fake_features: Float[Tensor, "M D"]
    ) -> float:
        """Compute FRD/FID given feature matrices.

        Args:
            real_features: Features from real images.
            fake_features: Features from generated images.

        forward method for FrdMetric.

        Executes PyTorch tensor operations.

        Args:
            real_features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            fake_features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            float: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        mu1, sigma1 = self._compute_statistics(real_features)
        mu2, sigma2 = self._compute_statistics(fake_features)

        return self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def _compute_statistics(self, features: Tensor):
        """Compute mean and covariance."""
        # Transfer to CPU numpy
        x = features.detach().cpu().numpy()

        mu = np.mean(x, axis=0)
        sigma = np.cov(x, rowvar=False)
        return mu, sigma

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of Frechet Distance."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape
        assert sigma1.shape == sigma2.shape

        diff = mu1 - mu2

        # Product might be almost singular. ``disp=False`` was deprecated
        # in SciPy 1.18 — the function now always returns just the
        # matrix square root (no error estimate). We discarded the
        # error estimate anyway, so this is purely a signature change.
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give complex numbers
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                logger.debug(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
