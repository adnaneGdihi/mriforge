"""Physics Mixin - Data Consistency & K-Space Operations
======================================================

This mixin provides shared physics-informed methods for generators
that perform MRI reconstruction with data consistency.

Compliance:
- soft_design.md: Composition over Inheritance (Mixins)
- perf_design.md: Zero-copy data movement
- perf.md: No CPU-GPU sync inside loops, contiguous tensors
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mriforge.infrastructure.physics.fft_ops import fft2c as canonical_fft2c
from mriforge.infrastructure.physics.fft_ops import ifft2c as canonical_ifft2c

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PhysicsMixin:
    """Mixin providing physics-informed methods for MRI generators.

    Provides:
    - Data consistency (soft and hard DC)
    - K-space to image and image to k-space transforms
    - Sensitivity map application/combination

    Subclasses must:
    - Have access to `self.device` or tensors on GPU
    - Optionally provide sensitivity maps
    """

    # ---------------------------------------------------------------------------
    # FFT Operations (Zero-Copy where possible)
    # ---------------------------------------------------------------------------
    @staticmethod
    def fft2c(x: Tensor) -> Tensor:
        """Centered 2D FFT - delegates to canonical implementation.

        Args:
            x: Image tensor of shape (..., H, W) or (..., H, W, 2).

        Returns:
            K-space tensor (complex or 2-channel).
        """
        # Ensure contiguous for FFT performance
        if not x.is_contiguous():
            x = x.contiguous()
        return canonical_fft2c(x)

    @staticmethod
    def ifft2c(k: Tensor) -> Tensor:
        """Centered 2D Inverse FFT - delegates to canonical implementation.

        Args:
            k: K-space tensor of shape (..., H, W) or complex.

        Returns:
            Image tensor.
        """
        # Ensure contiguous for FFT performance
        if not k.is_contiguous():
            k = k.contiguous()
        return canonical_ifft2c(k)

    # ---------------------------------------------------------------------------
    # Data Consistency
    # ---------------------------------------------------------------------------
    def soft_data_consistency(
        self,
        prediction: Tensor,
        kspace_measured: Tensor,
        mask: Tensor,
        lambda_dc: float = 0.5,
    ) -> Tensor:
        """Apply soft data consistency.

        Blends predicted k-space with measured k-space at acquired locations.

        Args:
            prediction: Predicted image tensor (B, C, H, W) or complex (B, H, W).
            kspace_measured: Measured k-space (B, C, H, W) or complex.
            mask: Undersampling mask (B, 1, H, W).
            lambda_dc: Blending weight [0, 1]. Higher = more trust in measured.

        Returns:
            Data-consistent image tensor.
        """
        # Convert prediction to k-space
        kspace_pred = self.fft2c(prediction)

        # Blend: k_dc = (1 - λ*mask) * k_pred + λ*mask * k_measured
        kspace_dc = (1 - lambda_dc * mask) * kspace_pred + lambda_dc * mask * kspace_measured

        # Convert back to image
        return self.ifft2c(kspace_dc)

    def hard_data_consistency(
        self,
        prediction: Tensor,
        kspace_measured: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply hard data consistency.

        Replaces predicted k-space with measured k-space at acquired locations.

        Args:
            prediction: Predicted image tensor.
            kspace_measured: Measured k-space.
            mask: Undersampling mask.

        Returns:
            Data-consistent image tensor.
        """
        return self.soft_data_consistency(
            prediction,
            kspace_measured,
            mask,
            lambda_dc=1.0,
        )

    # ---------------------------------------------------------------------------
    # Sensitivity Maps
    # ---------------------------------------------------------------------------
    @staticmethod
    def apply_sensitivity_maps(
        image: Tensor,
        sensitivity_maps: Tensor,
    ) -> Tensor:
        """Apply sensitivity maps to expand single-coil to multi-coil.

        Args:
            image: Image tensor (B, 1, H, W) or (B, H, W).
            sensitivity_maps: Sensitivity maps (B, Coils, H, W).

        Returns:
            Multi-coil image (B, Coils, H, W).
        """
        if image.ndim == 3:
            image = image.unsqueeze(1)
        return sensitivity_maps * image

    @staticmethod
    def combine_coils(
        multi_coil_image: Tensor,
        sensitivity_maps: Tensor | None = None,
    ) -> Tensor:
        """Combine multi-coil images.

        Args:
            multi_coil_image: Multi-coil image (B, Coils, H, W).
            sensitivity_maps: Optional sensitivity maps for SENSE combination.

        Returns:
            Combined image (B, 1, H, W).
        """
        if sensitivity_maps is not None:
            # SENSE combination
            return (sensitivity_maps.conj() * multi_coil_image).sum(dim=1, keepdim=True)
        # Root-sum-of-squares combination
        return torch.sqrt((multi_coil_image.abs() ** 2).sum(dim=1, keepdim=True))


class LatentMixin:
    """Mixin providing latent space methods for VAE-style generators.

    Provides:
    - Reparameterization trick
    - KL divergence computation
    - Latent space sampling
    """

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + std * epsilon.

        Args:
            mu: Mean tensor (B, D).
            logvar: Log variance tensor (B, D).

        Returns:
            Sampled latent tensor (B, D).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute KL divergence: KL(q(z|x) || p(z)).

        Assumes p(z) = N(0, I).

        Args:
            mu: Mean tensor (B, D).
            logvar: Log variance tensor (B, D).

        Returns:
            KL divergence per sample (B,).
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    @staticmethod
    def sample_latent(
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Sample from standard normal latent space.

        Args:
            shape: Shape of latent tensor.
            device: Target device.

        Returns:
            Sampled latent tensor.
        """
        return torch.randn(shape, device=device)


__all__ = ["LatentMixin", "PhysicsMixin"]
