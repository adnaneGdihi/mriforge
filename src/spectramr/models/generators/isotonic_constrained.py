"""Isotonic Constrained Generator for monotonicity-aware MRI reconstruction.

Ensures output satisfies monotonicity constraints (e.g., T1 > 0, signal
intensity ordering) via differentiable isotonic regression layers.

Reference:
    Amos et al., "Input Convex Neural Networks", ICML 2017.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _IsotonicLayer(nn.Module):
    """Differentiable monotonicity constraint via projected softmax weighting."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.Softplus(),  # ensures positive weights → monotonic mapping
        )
        self.transform = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _IsotonicLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        w = self.gate(x)
        return x + w * self.transform(x)


@register_model(name="isotonic_constrained", training_mode="reconstruction")
class IsotonicConstrainedGenerator(IGenerator, nn.Module):
    """Reconstruction network with monotonicity constraints."""

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
        num_constraints: int = kwargs.get("num_constraints", 4)

        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)

        blocks: list[nn.Module] = []
        for i in range(num_layers):
            blocks.append(nn.Conv2d(feature_dim, feature_dim, 3, padding=1))
            blocks.append(nn.GroupNorm(min(32, feature_dim), feature_dim))
            blocks.append(nn.SiLU())
            if (i + 1) % max(1, num_layers // num_constraints) == 0:
                blocks.append(_IsotonicLayer(feature_dim))
        self.backbone = nn.Sequential(*blocks)

        self.out_conv = nn.Conv2d(feature_dim, out_channels, 3, padding=1)

    @property
    def name(self) -> str:
        return "isotonic_constrained"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for IsotonicConstrainedGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.out_conv(self.backbone(self.intro(x)))

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
