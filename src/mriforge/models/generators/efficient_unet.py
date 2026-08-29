"""Efficient UNet Generator for memory-constrained MRI reconstruction.

Lightweight UNet with depthwise-separable convolutions and inverted residuals,
designed for model pruning and quantization experiments.

Reference:
    Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _InvertedResidual(nn.Module):
    """MBConv-style inverted residual block."""

    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 4) -> None:
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_residual = in_ch == out_ch
        self.block = nn.Sequential(
            # Pointwise expand
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            # Squeeze-excite
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, mid // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid // 4, mid, 1),
            nn.Sigmoid(),
        )
        # SE is applied as a gate — we split this into two paths:
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, max(1, mid // 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, mid // 4), mid, 1),
            nn.Sigmoid(),
        )
        self.pw_project = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _InvertedResidual.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.pw_expand(x)
        h = self.dw(h)
        h = h * self.se(h)
        h = self.pw_project(h)
        if self.use_residual:
            h = h + x
        return h


@register_model(name="efficient_unet", training_mode="reconstruction")
class EfficientUNetGenerator(IGenerator, nn.Module):
    """Memory-efficient UNet with inverted residual blocks for MRI reconstruction."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        features: list[int] = kwargs.get("features", [32, 64, 128, 256])

        # Encoder
        self.intro = nn.Conv2d(in_channels, features[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.encoders.append(_InvertedResidual(features[i], features[i]))
            self.downs.append(nn.Conv2d(features[i], features[i + 1], 2, stride=2))

        # Bottleneck
        self.mid = _InvertedResidual(features[-1], features[-1])

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(features[i], features[i - 1], 2, stride=2))
            self.decoders.append(_InvertedResidual(features[i - 1] * 2, features[i - 1]))

        self.out_conv = nn.Conv2d(features[0], out_channels, 1)

    @property
    def name(self) -> str:
        return "efficient_unet"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for EfficientUNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.intro(x)
        skips: list[torch.Tensor] = []
        for enc, down in zip(self.encoders, self.downs, strict=False):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.mid(h)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips), strict=False):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return self.out_conv(h)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
