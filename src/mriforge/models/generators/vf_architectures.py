"""Virtual Fiducial Experiment Architectures.

Purpose-built generator architectures for the VF experiment pipeline.
Each model is registered via ``@register_model`` and follows the standard
``IGenerator`` interface: ``__init__(in_channels, out_channels, **kwargs)``
plus ``forward(x) -> Tensor``.

Models
------
- ``bloch_aware_unet``: Bloch-equation-aware U-Net for bSSFP/B1 experiments
- ``cascade_residual``: Cascaded residual blocks for iterative correction
- ``cin_unet``: Conditional Instance Normalization U-Net
- ``cross_attention_fusion``: Cross-attention multi-modal fusion network
- ``deep_unfolding``: Learned ADMM / deep unfolding network
- ``learned_dictionary``: LISTA (unrolled sparse coding) network
- ``pinn_mri``: Physics-Informed Neural Network for MRI
- ``variational_network``: Variational Network (VarNet)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared mixin for IGenerator abstract methods
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _VFGeneratorMixin:
    """Provides default implementations for IGenerator abstract methods."""

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate output (alias for forward)."""
        return self.forward(x, **kwargs)  # type: ignore[attr-defined]

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        return input_shape

    def get_parameter_count(self) -> int:
        """Returns total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # type: ignore[attr-defined]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared building blocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ConvBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.block(x)


class _DownBlock(nn.Module):
    """Encoder block: MaxPool → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _DownBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.conv(self.pool(x))


class _UpBlock(nn.Module):
    """Decoder block: Upsample → Concat skip → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """forward method for _UpBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            skip (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.up(x)
        # Pad if spatial dims differ
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


def _build_encoder_decoder(
    in_ch: int,
    out_ch: int,
    features: list[int] | None = None,
) -> tuple[nn.Module, list[nn.Module], list[nn.Module], nn.Module]:
    """Build a standard 4-level U-Net encoder-decoder."""
    if features is None:
        features = [32, 64, 128, 256]
    inc = _ConvBlock(in_ch, features[0])
    downs = nn.ModuleList(
        [_DownBlock(features[i], features[i + 1]) for i in range(len(features) - 1)]
    )
    ups = nn.ModuleList(
        [_UpBlock(features[i + 1], features[i]) for i in reversed(range(len(features) - 1))]
    )
    outc = nn.Conv2d(features[0], out_ch, 1)
    return inc, downs, ups, outc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Bloch-Aware U-Net
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _BlochModulationLayer(nn.Module):
    """Learns a spatially-varying Bloch-parameter modulation map.

    Predicts per-pixel scale/shift conditioned on the input to make the
    network aware of T1/T2/B0 signal evolution.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.shift_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _BlochModulationLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        scale = self.scale_net(x).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift_net(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale + shift


@register_model(name="bloch_aware_unet", training_mode="reconstruction")
class BlochAwareUNet(_VFGeneratorMixin, IGenerator, nn.Module):
    """U-Net with Bloch-equation modulation layers for bSSFP/B1 experiments.

    Injects learned T1/T2/B0 scale-shift modulation after each encoder
    stage, making the network intrinsically aware of MR signal physics.

    Args:
        in_channels: Input channel count (default: 2 for real/imag).
        out_channels: Output channel count.
        features: Channel counts per encoder level.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.inc, self.downs, self.ups, self.outc = _build_encoder_decoder(
            in_channels, out_channels, features
        )
        self.bloch_mods = nn.ModuleList([_BlochModulationLayer(f) for f in features])

    @property
    def name(self) -> str:
        return "bloch_aware_unet"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for BlochAwareUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        skips: list[torch.Tensor] = []
        h = self.inc(x)
        h = self.bloch_mods[0](h)
        skips.append(h)
        for i, down in enumerate(self.downs):
            h = down(h)
            h = self.bloch_mods[i + 1](h)
            if i < len(self.downs) - 1:
                skips.append(h)
        for up in self.ups:
            h = up(h, skips.pop())
        return self.outc(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Cascade Residual Network
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _ResBlock(nn.Module):
    """Residual block: Conv-BN-ReLU-Conv-BN + identity."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return F.relu(self.block(x) + x, inplace=True)


class _CascadeStage(nn.Module):
    """One cascade stage: N residual blocks + channel projection."""

    def __init__(self, channels: int, num_blocks: int = 3) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[_ResBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _CascadeStage.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.blocks(x)


@register_model(name="cascade_residual", training_mode="reconstruction")
class CascadeResidualGenerator(_VFGeneratorMixin, IGenerator, nn.Module):
    """Cascaded residual network for iterative EPI/artifact correction.

    Each cascade stage refines the reconstruction residual, enabling
    progressive multi-step correction of phase/ghosting artifacts.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_cascades: Number of cascade stages.
        features: Hidden feature channels per cascade.
        num_blocks: Residual blocks per cascade stage.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_cascades: int = 5,
        features: list[int] | None = None,
        num_blocks: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        hidden = (features or [64])[0]
        self.stem = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.cascades = nn.ModuleList(
            [_CascadeStage(hidden, num_blocks) for _ in range(num_cascades)]
        )
        self.head = nn.Conv2d(hidden, out_channels, 1)

    @property
    def name(self) -> str:
        return "cascade_residual"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for CascadeResidualGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        identity = x
        h = self.stem(x)
        for cascade in self.cascades:
            h = h + cascade(h)  # residual cascade connection
        out = self.head(h)
        return out + identity  # global residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Conditional Instance Normalization U-Net
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _CINLayer(nn.Module):
    """Conditional Instance Normalization.

    Learns condition-dependent affine parameters (gamma, beta)
    for instance normalization.
    """

    def __init__(self, num_features: int, num_conditions: int = 2) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma_fc = nn.Linear(num_conditions, num_features)
        self.beta_fc = nn.Linear(num_conditions, num_features)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """forward method for _CINLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            condition (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        normed = self.norm(x)
        gamma = self.gamma_fc(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma * normed + beta


@register_model(name="cin_unet", training_mode="reconstruction")
class CINUNet(_VFGeneratorMixin, IGenerator, nn.Module):
    """U-Net with Conditional Instance Normalization for B1/flip-angle estimation.

    The CIN layers allow the network to adapt its normalization
    statistics based on acquisition-specific conditioning vectors
    (e.g., flip angle ratio, contrast type).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        features: Channel counts per encoder level.
        num_conditioning_channels: Size of the conditioning vector.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] | None = None,
        num_conditioning_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.inc, self.downs, self.ups, self.outc = _build_encoder_decoder(
            in_channels, out_channels, features
        )
        self.cin_layers = nn.ModuleList([_CINLayer(f, num_conditioning_channels) for f in features])
        self._cond_dim = num_conditioning_channels

    @property
    def name(self) -> str:
        return "cin_unet"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for CINUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Extract or create conditioning vector
        cond = kwargs.get("condition")
        if cond is None:
            cond = torch.zeros(x.shape[0], self._cond_dim, device=x.device)

        skips: list[torch.Tensor] = []
        h = self.inc(x)
        h = self.cin_layers[0](h, cond)
        skips.append(h)
        for i, down in enumerate(self.downs):
            h = down(h)
            h = self.cin_layers[i + 1](h, cond)
            if i < len(self.downs) - 1:
                skips.append(h)
        for up in self.ups:
            h = up(h, skips.pop())
        return self.outc(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Cross-Attention Fusion Network
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _CrossAttention(nn.Module):
    """Multi-head cross-attention between image and conditioning features."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """forward method for _CrossAttention.

        Executes PyTorch tensor operations.

        Args:
            query (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            kv (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b, c, h, w = query.shape
        q = query.flatten(2).transpose(1, 2)  # [B, HW, C]
        k = kv.flatten(2).transpose(1, 2)
        out, _ = self.attn(q, k, k)
        out = self.norm(out + q)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_model(name="cross_attention_fusion", training_mode="reconstruction")
class CrossAttentionFusionGenerator(_VFGeneratorMixin, IGenerator, nn.Module):
    """Cross-attention fusion for multi-coil/multi-modal combination.

    Uses cross-attention between the main reconstruction stream
    and a conditioning branch (e.g., coil sensitivity maps, B1 fields)
    to fuse information from multiple sources.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        features: Channel counts per level.
        num_heads: Attention heads.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] | None = None,
        num_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.inc, self.downs, self.ups, self.outc = _build_encoder_decoder(
            in_channels, out_channels, features
        )
        # Cross-attention at the bottleneck
        self.cross_attn = _CrossAttention(features[-1], num_heads)
        # Conditioning encoder (same input projected to bottleneck dim)
        self.cond_enc = nn.Sequential(
            nn.Conv2d(in_channels, features[-1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(None),  # keeps spatial dims
        )
        self._bottleneck_pool = nn.AdaptiveAvgPool2d(None)

    @property
    def name(self) -> str:
        return "cross_attention_fusion"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for CrossAttentionFusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        skips: list[torch.Tensor] = []
        h = self.inc(x)
        skips.append(h)
        for i, down in enumerate(self.downs):
            h = down(h)
            if i < len(self.downs) - 1:
                skips.append(h)

        # Cross-attention at bottleneck
        cond = self.cond_enc(x)
        cond = F.adaptive_avg_pool2d(cond, h.shape[2:])
        h = self.cross_attn(h, cond)

        for up in self.ups:
            h = up(h, skips.pop())
        return self.outc(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Deep Unfolding Network
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _ProximalBlock(nn.Module):
    """Learned proximal operator (small U-Net-like CNN)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        mid = max(channels * 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ProximalBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return x + self.net(x)


@register_model(name="deep_unfolding", training_mode="reconstruction")
class DeepUnfoldingGenerator(_VFGeneratorMixin, IGenerator, nn.Module):
    """Deep unfolding network (learned ADMM).

    Unrolls N iterations of the proximal gradient descent algorithm
    with learnable step sizes and proximal operators. Each iteration:
        x_{n+1} = prox_theta(x_n - eta * grad)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_iterations: Number of unrolled iterations.
        features: Channel list for internal proximal blocks.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_iterations: int = 8,
        features: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        # Learned step sizes per iteration
        self.step_sizes = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_iterations)]
        )
        # Learned proximal operators (shared or per-iteration)
        self.prox_blocks = nn.ModuleList(
            [_ProximalBlock(in_channels) for _ in range(num_iterations)]
        )
        # Output projection
        self.output_proj = nn.Conv2d(in_channels, out_channels, 1)

    @property
    def name(self) -> str:
        return "deep_unfolding"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for DeepUnfoldingGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_cur = x
        for i in range(self.num_iterations):
            # Gradient step (simplified: gradient of ||Ax - y||^2 ≈ x - input)
            grad = x_cur - x
            x_cur = x_cur - self.step_sizes[i] * grad
            # Proximal step (learned denoiser)
            x_cur = self.prox_blocks[i](x_cur)
        return self.output_proj(x_cur)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Learned Dictionary (LISTA) Network
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _LISTALayer(nn.Module):
    """One LISTA iteration: z = shrink(W1 * x + S * z_prev, theta)."""

    def __init__(self, signal_dim: int, dict_dim: int) -> None:
        super().__init__()
        self.W = nn.Conv2d(signal_dim, dict_dim, 1)
        self.S = nn.Conv2d(dict_dim, dict_dim, 1)
        self.theta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """forward method for _LISTALayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            z_prev (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        z = self.W(x) + self.S(z_prev)
        # Soft thresholding
        return torch.sign(z) * F.relu(torch.abs(z) - self.theta.abs())


@register_model(name="learned_dictionary", training_mode="reconstruction")
class LearnedDictionaryGenerator(_VFGeneratorMixin, IGenerator, nn.Module):
    """LISTA: Learned Iterative Shrinkage-Thresholding Algorithm.

    Unrolls sparse coding iterations with learned dictionaries and
    thresholds for artifact extraction and sparse signal separation.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_atoms: Dictionary size (number of atoms).
        atom_size: Spatial size of dictionary atoms (unused, for config compat).
        num_iterations: LISTA unrolling iterations.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_atoms: int = 256,
        atom_size: int = 8,
        num_iterations: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.lista_layers = nn.ModuleList(
            [_LISTALayer(in_channels, num_atoms) for _ in range(num_iterations)]
        )
        # Synthesis: sparse code -> reconstruction
        self.synthesis = nn.Sequential(
            nn.Conv2d(num_atoms, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )

    @property
    def name(self) -> str:
        return "learned_dictionary"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for LearnedDictionaryGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        z = torch.zeros(
            x.shape[0],
            self.lista_layers[0].W.out_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )
        for layer in self.lista_layers:
            z = layer(x, z)
        return self.synthesis(z) + x  # residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Physics-Informed Neural Network for MRI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _SinActivation(nn.Module):
    """Sinusoidal activation for PINN (like SIREN but simpler)."""

    def __init__(self, omega: float = 30.0) -> None:
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _SinActivation.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.sin(self.omega * x)


@register_model(name="pinn_mri", training_mode="reconstruction")
class PINNMRIGenerator(_VFGeneratorMixin, IGenerator, nn.Module):
    """Physics-Informed Neural Network for MRI reconstruction.

    Uses sinusoidal activations and a coordinate-like MLP backbone
    that natively supports embedding PDE constraints (Laplace, Helmholtz)
    into the loss function.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_layers: Number of hidden layers.
        hidden_dim: Hidden dimension width.
        omega_0: Frequency of sinusoidal activation.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 6,
        hidden_dim: int = 256,
        omega_0: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Feature extraction front-end (conv-based, not pure MLP)
        self.stem = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.stem_act = _SinActivation(omega_0)

        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))
            layers.append(_SinActivation(omega_0))
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Conv2d(hidden_dim, out_channels, 1)

        # SIREN-style weight initialization
        self._init_weights(omega_0)

    def _init_weights(self, omega_0: float) -> None:
        with torch.no_grad():
            nn.init.uniform_(
                self.stem.weight,
                -1.0 / self.stem.in_channels,
                1.0 / self.stem.in_channels,
            )
            for layer in self.backbone:
                if isinstance(layer, nn.Conv2d):
                    fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / omega_0
                    nn.init.uniform_(layer.weight, -bound, bound)

    @property
    def name(self) -> str:
        return "pinn_mri"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for PINNMRIGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.stem_act(self.stem(x))
        h = self.backbone(h)
        return self.head(h) + x  # residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Variational Network (VarNet)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _VarNetBlock(nn.Module):
    """One VarNet refinement cascade: CNN regulariser + DC-like step."""

    def __init__(self, channels: int, chans: int = 18) -> None:
        super().__init__()
        self.regulariser = nn.Sequential(
            nn.Conv2d(channels, chans, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, channels, 3, padding=1),
        )
        self.dc_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_cur: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """forward method for _VarNetBlock.

        Executes PyTorch tensor operations.

        Args:
            x_cur (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_ref (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Regulariser step
        reg = self.regulariser(x_cur)
        # DC step: pull towards reference (input)
        dc = self.dc_weight * (x_cur - x_ref)
        return x_cur - reg - dc


@register_model(name="variational_network", training_mode="reconstruction", spatial_dims=(2,))
class VariationalNetworkGenerator(_VFGeneratorMixin, IGenerator, nn.Module):
    """Variational Network (VarNet) for MRI reconstruction.

    Unrolls N cascades of CNN regularisation + data consistency steps,
    following the Hammernik et al. (MRM 2018) formulation.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_cascades: Number of VarNet cascades.
        chans: Internal CNN channels per cascade.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_cascades: int = 8,
        chans: int = 18,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cascades = nn.ModuleList(
            [_VarNetBlock(in_channels, chans) for _ in range(num_cascades)]
        )
        self.output_proj = nn.Conv2d(in_channels, out_channels, 1)

    @property
    def name(self) -> str:
        return "variational_network"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for VariationalNetworkGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_ref = x
        x_cur = x
        for cascade in self.cascades:
            x_cur = cascade(x_cur, x_ref)
        return self.output_proj(x_cur)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# __all__
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    "BlochAwareUNet",
    "CINUNet",
    "CascadeResidualGenerator",
    "CrossAttentionFusionGenerator",
    "DeepUnfoldingGenerator",
    "LearnedDictionaryGenerator",
    "PINNMRIGenerator",
    "VariationalNetworkGenerator",
]
