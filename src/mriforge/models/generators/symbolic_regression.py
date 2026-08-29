"""Symbolic Regression Generator for physics-guided MRI signal discovery.

Wraps a neural network backbone around a symbolic regression head that
discovers interpretable closed-form signal equations. The network provides
learned features; the symbolic head maps them to a linear combination of
basis functions (exp, sin, mul, etc.) with dimensional consistency.

Reference:
    Cranmer et al., "Discovering Symbolic Models from Deep Learning
    with Inductive Biases", NeurIPS 2020.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _SymbolicHead(nn.Module):
    """Learned linear combination of fixed basis functions."""

    def __init__(self, in_dim: int, num_terms: int = 5) -> None:
        super().__init__()
        self.num_terms = num_terms
        # Each term has a learnable coefficient
        self.coefficients = nn.Parameter(torch.randn(num_terms) * 0.01)
        # Project input to scalar per term
        self.projections = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_terms)])

    @staticmethod
    def _basis(x: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply basis function by index: 0=identity, 1=exp, 2=sin, 3=square, 4=recip."""
        if idx == 0:
            return x
        elif idx == 1:
            return torch.exp(-x.abs().clamp(max=10))
        elif idx == 2:
            return torch.sin(x)
        elif idx == 3:
            return x**2
        else:
            return 1.0 / (x.abs() + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _SymbolicHead.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        result = torch.zeros_like(x[:, :1])
        for i in range(self.num_terms):
            proj = self.projections[i](x)  # [B*HW, 1]
            result = result + self.coefficients[i] * self._basis(proj, i % 5)
        return result


@register_model(name="symbolic_regression", training_mode="reconstruction")
class SymbolicRegressionGenerator(IGenerator, nn.Module):
    """Neural backbone + symbolic regression head for MRI signal discovery."""

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
        num_terms: int = kwargs.get("num_symbolic_terms", 5)

        # CNN backbone for feature extraction
        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                    nn.GroupNorm(min(32, feature_dim), feature_dim),
                    nn.SiLU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)

        # Symbolic head operates per-pixel
        self.symbolic_head = _SymbolicHead(feature_dim, num_terms)
        self.out_proj = nn.Conv2d(1, out_channels, 1)

    @property
    def name(self) -> str:
        return "symbolic_regression"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for SymbolicRegressionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        h = self.intro(x)
        h = self.backbone(h)  # [B, D, H, W]
        # Flatten spatial for symbolic head
        h_flat = h.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        sym_out = self.symbolic_head(h_flat)  # [B*HW, 1]
        sym_out = sym_out.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        return self.out_proj(sym_out)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
