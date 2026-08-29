"""K-space rendering from Gaussian clouds."""

import logging

import torch

from mriforge.infrastructure.physics.implementations.gaussian_fourier_op import (
    differentiable_gaussian_fourier,
)

from .parameter_extractor import OptimizableParameters

logger = logging.getLogger(__name__)


class KSpaceRenderer:
    """Renders k-space from Gaussian primitives using differentiable Fourier projection.

    Handles both Cartesian (grid) and non-Cartesian (arbitrary trajectory) sampling.
    """

    def __init__(self, device: torch.device):
        """
        Args:
            device: Target device for computations.
        """
        self.device = device

    def render(
        self,
        params: OptimizableParameters,
        kspace_shape: tuple,
        trajectory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render k-space from Gaussian parameters.

        Args:
            params: Optimizable Gaussian parameters (means, scales, rotations, features, opacity).
            kspace_shape: Target k-space shape (B, C, H, W) or (B, C, M) for non-Cartesian.
            trajectory: Optional k-space trajectory (M, 3). If None, uses Cartesian grid.

        Returns:
            K-space tensor in real/imag channel format:
                - Cartesian: (B, 2*C, H, W) where first C channels are real, next C are imaginary
                - Non-Cartesian: (B, 2*C, M) in same format

        Note:
            If params.features or params.opacity is None, creates default uniform opacity.
        """
        # Prepare means (flatten if 3D)
        means_eval = params.means[:, 0, :] if params.means.dim() == 3 else params.means

        # Prepare k-space coordinates
        k_coords = self._prepare_kspace_coords(kspace_shape, trajectory)

        # Select features for projection
        proj_features = self._prepare_projection_features(params, means_eval.shape[0])

        # Render k-space via differentiable Gaussian Fourier transform
        kspace_complex = differentiable_gaussian_fourier(
            means=means_eval,
            scales=params.scales,
            quaternions=params.rotations,
            features=proj_features,
            k_trajectory=k_coords,
        )

        # Convert to real/imag channel format
        kspace_pred = self._format_kspace(
            kspace_complex, kspace_shape, trajectory, proj_features.shape[-1]
        )

        return kspace_pred

    def _prepare_kspace_coords(
        self, kspace_shape: tuple, trajectory: torch.Tensor | None
    ) -> torch.Tensor:
        """Prepare k-space sampling coordinates.

        Args:
            kspace_shape: Target k-space shape.
            trajectory: Optional custom trajectory.

        Returns:
            K-space coordinates (M, 3) where M is number of samples.
        """
        if trajectory is not None:
            return trajectory.to(self.device)

        # Cartesian grid
        H, W = kspace_shape[-2:]
        kz = torch.linspace(-1, 1, H, device=self.device)
        ky = torch.linspace(-1, 1, W, device=self.device)
        gy, gx = torch.meshgrid(kz, ky, indexing="ij")
        gz = torch.zeros_like(gy)
        k_coords = torch.stack([gz, gy, gx], dim=-1).reshape(-1, 3)
        return k_coords

    def _prepare_projection_features(
        self, params: OptimizableParameters, num_gaussians: int
    ) -> torch.Tensor:
        """Prepare features for Gaussian projection.

        Args:
            params: Optimizable parameters.
            num_gaussians: Number of Gaussian primitives.

        Returns:
            Feature tensor (N, C) suitable for projection.
        """
        proj_features = params.features if params.features is not None else params.opacity

        if proj_features is None:
            # Create default uniform opacity
            logger.warning("No features/opacity found, using default uniform opacity")
            proj_features = torch.ones(num_gaussians, 1, device=self.device, requires_grad=True)

        return proj_features

    def _format_kspace(
        self,
        kspace_complex: torch.Tensor,
        kspace_shape: tuple,
        trajectory: torch.Tensor | None,
        num_feature_channels: int,
    ) -> torch.Tensor:
        """Convert complex k-space to real/imag channel format.

        Args:
            kspace_complex: Complex k-space (M, C) or other shape.
            kspace_shape: Target shape.
            trajectory: Optional trajectory (determines Cartesian vs non-Cartesian).
            num_feature_channels: Number of feature channels.

        Returns:
            Real/imag format k-space (B, 2*C, H, W) or (B, 2*C, M).
        """
        if trajectory is None:
            # Cartesian: reshape to grid and permute
            H, W = kspace_shape[-2:]
            kspace_complex = kspace_complex.view(-1, H, W, num_feature_channels)
            real_part = kspace_complex.real.permute(0, 3, 1, 2)  # (B, C, H, W)
            imag_part = kspace_complex.imag.permute(0, 3, 1, 2)
            kspace_pred = torch.cat([real_part, imag_part], dim=1)  # (B, 2*C, H, W)
        else:
            # Non-Cartesian: transpose and concatenate
            real_part = kspace_complex.real.transpose(-1, -2)  # (B, C, M)
            imag_part = kspace_complex.imag.transpose(-1, -2)
            kspace_pred = torch.cat([real_part, imag_part], dim=1)  # (B, 2*C, M)

        return kspace_pred

    def match_target_shape(
        self, kspace_pred: torch.Tensor, kspace_target: torch.Tensor
    ) -> torch.Tensor:
        """Match predicted k-space to target shape (batch size and channels).

        Args:
            kspace_pred: Predicted k-space.
            kspace_target: Target k-space.

        Returns:
            Predicted k-space with matched shape.
        """
        # Match batch size
        if kspace_pred.shape[0] < kspace_target.shape[0]:
            kspace_pred = kspace_pred.repeat(
                kspace_target.shape[0], 1, *([1] * (kspace_pred.dim() - 2))
            )

        # Match channels
        if kspace_pred.shape[1] != kspace_target.shape[1]:
            if kspace_pred.shape[1] < kspace_target.shape[1]:
                # Pad with zeros
                padding = torch.zeros(
                    kspace_pred.shape[0],
                    kspace_target.shape[1] - kspace_pred.shape[1],
                    *kspace_pred.shape[2:],
                    device=kspace_pred.device,
                )
                kspace_pred = torch.cat([kspace_pred, padding], dim=1)
            else:
                # Truncate
                kspace_pred = kspace_pred[:, : kspace_target.shape[1]]

        return kspace_pred
