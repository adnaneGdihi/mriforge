"""SIREN Generator for implicit neural field MRI reconstruction.

Sinusoidal Representation Network mapping (x,y) coordinates to complex-valued
field estimates. Used for PINN-style boundary-value field interpolation.

Reference:
    Sitzmann et al., "Implicit Neural Representations with Periodic
    Activation Functions", NeurIPS 2020.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class SirenLayer(nn.Module):
    """Single SIREN layer with sinusoidal activation."""

    def __init__(
        self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1.0 / self.linear.in_features, 1.0 / self.linear.in_features
                )
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for SirenLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.sin(self.omega_0 * self.linear(x))


@register_model(name="siren", training_mode="reconstruction")
class SIRENGenerator(IGenerator, nn.Module):
    """SIREN implicit neural field generator for MRI.

    Maps 2-D normalised coordinates → complex field (real/imag channels).
    Can optionally operate as a standard image-to-image network by creating
    an internal coordinate grid.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        hidden_features: int = kwargs.get("hidden_features", 256)
        hidden_layers: int = kwargs.get("hidden_layers", 3)
        first_omega_0: float = kwargs.get("first_omega_0", 30.0)
        hidden_omega_0: float = kwargs.get("hidden_omega_0", 1.0)

        actual_in_features = 2 if in_channels == 2 else in_channels + 2
        layers: list[nn.Module] = [
            SirenLayer(actual_in_features, hidden_features, omega_0=first_omega_0, is_first=True)
        ]
        for _ in range(hidden_layers):
            layers.append(SirenLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*layers)

        # Final linear (no sine activation)
        self.final_linear = nn.Linear(hidden_features, out_channels)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            self.final_linear.weight.uniform_(-bound, bound)

    @property
    def name(self) -> str:
        return "siren"

    def _make_coords(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create normalised [-1, 1] coordinate grid."""
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass.

        For image-to-image mode: x is [B, C, H, W], we create an internal
        coordinate grid and concatenate the image features per-pixel.
        For coordinate mode: x is [N, 2] coordinates.

        forward method for SIRENGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() == 4:
            B, C, H, W = x.shape
            coords = self._make_coords(H, W, x.device)  # [H, W, 2]
            # If in_channels == 2, operate purely on coordinates
            if self.in_channels == 2:
                coords_flat = coords.reshape(-1, 2)  # [H*W, 2]
                out_flat = self.final_linear(self.net(coords_flat))  # [H*W, out]
                out = out_flat.reshape(H, W, self.out_channels).permute(2, 0, 1)
                return out.unsqueeze(0).expand(B, -1, -1, -1)
            else:
                # Concatenate coords with image features
                coords_expanded = coords.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
                combined = torch.cat([x, coords_expanded], dim=1)
                # Flatten spatial dims
                combined = combined.permute(0, 2, 3, 1).reshape(B * H * W, -1)
                out = self.final_linear(self.net(combined))
                return out.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)
        else:
            # Pure coordinate mode [N, 2]
            return self.final_linear(self.net(x))

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) == 4:
            return (input_shape[0], self.out_channels, *input_shape[2:])
        return (input_shape[0], self.out_channels)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
