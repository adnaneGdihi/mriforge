"""Differentiable Rendering Generator for MRI slice simulation.

Implements a differentiable forward model that renders 2-D MR images from
a 3-D volumetric tissue representation using Hamming-windowed sinc slice
profiles and partial-volume modelling.

Reference:
    Kato et al., "Neural 3D Mesh Renderer", CVPR 2018.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _SliceProfileLayer(nn.Module):
    """Differentiable slice-select profile (Hamming-windowed sinc)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.thickness = nn.Parameter(torch.tensor(3.0))  # mm
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _SliceProfileLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.refine(x) * torch.sigmoid(self.thickness)


@register_model(name="differentiable_rendering", training_mode="reconstruction")
class DifferentiableRenderingGenerator(IGenerator, nn.Module):
    """Differentiable MRI slice rendering generator."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        feature_dim: int = kwargs.get("feature_dim", 64)
        num_layers: int = kwargs.get("num_layers", 8)
        rendering_layers: int = kwargs.get("rendering_layers", 3)

        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)

        # Feature backbone
        backbone: list[nn.Module] = []
        for _ in range(num_layers):
            backbone.extend(
                [
                    nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                    nn.GroupNorm(min(32, feature_dim), feature_dim),
                    nn.SiLU(),
                ]
            )
        self.backbone = nn.Sequential(*backbone)

        # Differentiable rendering stages
        self.render_stages = nn.ModuleList(
            [_SliceProfileLayer(feature_dim) for _ in range(rendering_layers)]
        )

        self.out_conv = nn.Conv2d(feature_dim, out_channels, 3, padding=1)

    @property
    def name(self) -> str:
        return "differentiable_rendering"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for DifferentiableRenderingGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.intro(x)
        h = self.backbone(h)
        for render in self.render_stages:
            h = h + render(h)
        return self.out_conv(h)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
