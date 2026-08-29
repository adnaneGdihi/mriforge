"""Neural Advection PDE Generator — Marker-Conditioned Deformation Solver.

Predicts a dense velocity field from a 1-D physiological marker signal
and uses a differentiable Spatial Transformer Network (STN) to advect
corrupted images backward to their stable rest state.

Physics:
    The transport (advection) equation governs image deformation:

    .. math::

        \\frac{\\partial I}{\\partial t} + \\mathbf{v} \\cdot \\nabla I = 0

    The network predicts :math:`\\mathbf{v}(x, y, t)` from the marker
    signal, then integrates backward to undo the physiological motion.

Architecture:
    .. code-block:: text

        marker_signal → MarkerEncoder → conditioning vector
        corrupted_image + conditioning → UNet → velocity_field [B, 2, H, W]
        velocity_field → STN (grid_sample) → unwarped_image

Reference:
    Krebs et al., "Learning a Generative Motion Model from Image Sequences
    based on a Latent Neural ODE," *MICCAI*, 2021.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


class _MarkerEncoder(nn.Module):
    """Encode 1-D marker signal into a spatial conditioning vector."""

    def __init__(self, marker_dim: int = 1, cond_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(marker_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

    def forward(self, marker: Tensor) -> Tensor:
        """Encode marker signal.

        Args:
            marker: ``[B]`` or ``[B, D]`` marker signal.

        Returns:
            Conditioning vector ``[B, cond_dim]``.

        forward method for _MarkerEncoder.

        Executes PyTorch tensor operations.

        Args:
            marker (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker.dim() == 1:
            marker = marker.unsqueeze(-1)
        return self.net(marker)


class _VelocityFieldUNet(nn.Module):
    """Lightweight U-Net predicting a 2-D velocity field."""

    def __init__(
        self,
        in_channels: int = 2,
        cond_dim: int = 64,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
        )

        # Bottleneck with FiLM conditioning
        self.film_scale = nn.Linear(cond_dim, base_channels * 4)
        self.film_bias = nn.Linear(cond_dim, base_channels * 4)

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )

        # Output: 2-channel velocity field (vx, vy)
        self.out = nn.Conv2d(base_channels, 2, 1)
        # Initialize output to near-zero (identity deformation)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Predict velocity field.

        Args:
            x: Input image ``[B, C, H, W]``.
            cond: Conditioning vector ``[B, cond_dim]``.

        Returns:
            Velocity field ``[B, 2, H, W]``.

        forward method for _VelocityFieldUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encoder
        e1 = self.enc1(x)  # [B, 32, H, W]
        e2 = self.enc2(e1)  # [B, 64, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 128, H/4, W/4]

        # FiLM conditioning at bottleneck
        scale = self.film_scale(cond).unsqueeze(-1).unsqueeze(-1)
        bias = self.film_bias(cond).unsqueeze(-1).unsqueeze(-1)
        e3 = e3 * (1.0 + scale) + bias

        # Decoder with skip connections
        d3 = self.dec3(e3)  # [B, 64, H/2, W/2]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [B, 32, H, W]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # [B, 32, H, W]

        return self.out(d1)  # [B, 2, H, W]


@register_model(name="neural_advection", training_mode="virtual_fiducial")
class NeuralAdvectionGenerator(nn.Module, IGenerator):
    """Marker-conditioned neural advection for motion correction.

    Takes a corrupted image + 1-D marker signal, predicts a dense
    velocity field, and advects the image backward to rest state.

    Args:
        in_channels: Input image channels (2 for complex stacked).
        out_channels: Output image channels.
        marker_dim: Dimension of marker signal input.
        cond_dim: FiLM conditioning dimension.
        base_channels: Base channels for velocity U-Net.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        marker_dim: int = 1,
        cond_dim: int = 64,
        base_channels: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.marker_encoder = _MarkerEncoder(marker_dim, cond_dim)
        self.velocity_net = _VelocityFieldUNet(in_channels, cond_dim, base_channels)
        # Cache of the last STN velocity field (set in forward); the strategy
        # routes it into the hyperelastic-Jacobian loss as intermediate_outputs.
        self.last_deformation_field: Tensor | None = None

    @property
    def name(self) -> str:
        return "neural_advection"

    def predict_velocity(
        self,
        corrupted: Tensor,
        marker_signal: Tensor,
    ) -> Tensor:
        """Predict dense velocity field from corrupted image and marker.

        Args:
            corrupted: ``[B, C, H, W]``.
            marker_signal: ``[B]`` or ``[B, D]``.

        Returns:
            Velocity field ``[B, 2, H, W]``.
        """
        cond = self.marker_encoder(marker_signal)
        return self.velocity_net(corrupted, cond)

    @staticmethod
    def warp_image(image: Tensor, velocity: Tensor) -> Tensor:
        """Warp image using velocity field via differentiable STN.

        Applies backward warping: each output pixel samples from
        ``input[x + vx, y + vy]``.

        Args:
            image: ``[B, C, H, W]``.
            velocity: ``[B, 2, H, W]`` displacement in pixels.

        Returns:
            Warped image ``[B, C, H, W]``.
        """
        B, C, H, W = image.shape

        # Build identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing="ij",
        )
        identity = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        identity = identity.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # Convert pixel displacement to normalised coordinates
        velocity_norm = torch.zeros_like(identity)
        velocity_norm[..., 0] = velocity[:, 0] * (2.0 / W)  # x
        velocity_norm[..., 1] = velocity[:, 1] * (2.0 / H)  # y

        grid = identity + velocity_norm

        return F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def forward(
        self,
        x: Tensor,
        marker_signal: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Forward pass: predict velocity and unwarp.

        Args:
            x: Corrupted image ``[B, C, H, W]``.
            marker_signal: 1-D marker signal ``[B]``.  If ``None``,
                defaults to zeros (identity transform).

        Returns:
            Motion-corrected image ``[B, C, H, W]``.

        forward method for NeuralAdvectionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker_signal is None:
            marker_signal = torch.zeros(x.shape[0], device=x.device)

        velocity = self.predict_velocity(x, marker_signal)
        # Expose the dense STN velocity (deformation) field so a strategy can
        # route it into the hyperelastic-Jacobian incompressibility loss as
        # ``intermediate_outputs[0]`` — that regulariser must act on the
        # deformation field, NOT on the reconstructed image (where det(J) is
        # meaningless). Without this the field was computed and discarded.
        self.last_deformation_field = velocity
        return self.warp_image(x, velocity)

    def generate(self, z: Tensor, **kwargs: Any) -> Tensor:
        """Generate from latent (wrapper for IGenerator compatibility)."""
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["NeuralAdvectionGenerator"]
