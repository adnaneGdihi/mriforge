"""3D Gaussian Splatting for MRI Reconstruction (3DGSMR).

This package implements explicit volumetric reconstruction using 3D Gaussians
with analytical Fourier transforms for direct k-space consistency.

Key Components:
- GaussianCloud: Collection of 3D Gaussians with learnable params.
- AnalyticalFourierTransform: Computes G(k) directly from params.
"""

from .gaussian_primitives import GaussianCloud

__all__ = ["GaussianCloud"]
