"""Gaussian cloud update and restoration."""

import logging

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.implementations.gaussian_fourier_op import GaussianCloud

from .parameter_extractor import OptimizableParameters

logger = logging.getLogger(__name__)


class GaussianUpdater:
    """Updates Gaussian clouds with optimized parameters.

    Handles two update paths:
    1. nn.Module-based Gaussians: Updates internal buffers (_coeffs, _xyz, etc.)
    2. Standard GaussianCloud: Creates new optimized cloud instance
    """

    @staticmethod
    def update(
        gaussians: GaussianCloud,
        params: OptimizableParameters,
    ) -> GaussianCloud:
        """Update Gaussian cloud with optimized parameters.

        Args:
            gaussians: Original Gaussian cloud (may be nn.Module or standard class).
            params: Optimized parameters with gradients detached.

        Returns:
            Updated Gaussian cloud (either modified input or new instance).
        """
        # Case 1: nn.Module-based Gaussians (MedGS, etc.)
        if isinstance(gaussians, nn.Module):
            return GaussianUpdater._update_module(gaussians, params)

        # Case 2: Standard GaussianCloud
        return GaussianUpdater._update_standard(params)

    @staticmethod
    def _update_module(
        gaussians: nn.Module,
        params: OptimizableParameters,
    ) -> nn.Module:
        """Update an nn.Module-based Gaussian cloud in-place.

        Args:
            gaussians: Module instance with parameter buffers.
            params: Optimized parameters.

        Returns:
            The updated gaussians object.
        """
        with torch.no_grad():
            # Update _coeffs (MedGS format)
            if hasattr(gaussians, "_coeffs") and params.means is not None:
                gaussians._coeffs.copy_(params.means)

            # Update position (_xyz)
            if hasattr(gaussians, "_xyz") and params.means is not None:
                gaussians._xyz.copy_(params.means)

            # Update scales
            if hasattr(gaussians, "_scale") and params.scales is not None:
                gaussians._scale.copy_(params.scales)

            # Update rotations
            if hasattr(gaussians, "_rotation") and params.rotations is not None:
                gaussians._rotation.copy_(params.rotations)

            # Update features
            if hasattr(gaussians, "_features") and params.features is not None:
                gaussians._features.copy_(params.features)

            # Update opacity
            if hasattr(gaussians, "_opacity") and params.opacity is not None:
                gaussians._opacity.copy_(params.opacity)

        return gaussians

    @staticmethod
    def _update_standard(params: OptimizableParameters) -> GaussianCloud:
        """Create a new GaussianCloud from optimized parameters.

        Args:
            params: Optimized parameters.

        Returns:
            New GaussianCloud instance.
        """
        final_cloud = GaussianCloud(
            means=params.means.detach() if params.means is not None else None,
            scales=params.scales.detach() if params.scales is not None else None,
            quaternions=(params.rotations.detach() if params.rotations is not None else None),
            features=(
                params.features.detach()
                if params.features is not None
                else (torch.ones_like(params.opacity) if params.opacity is not None else None)
            ),
            opacity=(
                params.opacity.detach()
                if params.opacity is not None
                else (torch.ones_like(params.features) if params.features is not None else None)
            ),
        )
        return final_cloud
