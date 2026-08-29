"""Neural Cache Generator for rapid retrieval-augmented MRI reconstruction.

Maintains a learnable key-value memory bank. During forward pass, input
features query the cache via attention to retrieve relevant reconstruction
priors, enabling fast adaptation to new anatomies.

Reference:
    Grave et al., "Improving Neural Language Models with a Continuous Cache",
    ICLR 2017.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _CacheAttention(nn.Module):
    """Cross-attention between spatial features and a key-value memory bank."""

    def __init__(self, ch: int, cache_size: int, kv_dim: int) -> None:
        super().__init__()
        self.keys = nn.Parameter(torch.randn(cache_size, kv_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(cache_size, kv_dim) * 0.02)
        self.q_proj = nn.Conv2d(ch, kv_dim, 1)
        self.out_proj = nn.Conv2d(kv_dim, ch, 1)
        self.scale = kv_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _CacheAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        q = self.q_proj(x).reshape(B, -1, H * W).permute(0, 2, 1)  # [B, HW, D]
        attn = torch.matmul(q, self.keys.T) * self.scale  # [B, HW, K]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, self.values)  # [B, HW, D]
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)
        return x + self.out_proj(out)


@register_model(name="neural_cache", training_mode="reconstruction")
class NeuralCacheGenerator(IGenerator, nn.Module):
    """Retrieval-augmented reconstruction network with neural memory cache."""

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
        cache_size: int = kwargs.get("cache_size", 1024)
        kv_dim: int = kwargs.get("key_value_dim", 64)

        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(feature_dim, feature_dim, 3, padding=1))
            layers.append(nn.GroupNorm(min(32, feature_dim), feature_dim))
            layers.append(nn.SiLU())
            if i == num_layers // 2:
                layers.append(_CacheAttention(feature_dim, cache_size, kv_dim))
        self.backbone = nn.Sequential(*layers)
        self.cache_attn = _CacheAttention(feature_dim, cache_size, kv_dim)
        self.out_conv = nn.Conv2d(feature_dim, out_channels, 3, padding=1)

    @property
    def name(self) -> str:
        return "neural_cache"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for NeuralCacheGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.intro(x)
        h = self.backbone(h)
        h = self.cache_attn(h)
        return self.out_conv(h)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
