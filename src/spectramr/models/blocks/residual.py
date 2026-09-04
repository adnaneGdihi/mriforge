"""Residual Building Blocks
=======================

Common residual connection patterns used across different architectures.
"""

import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """__init__.

        Args:
            channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn1 = nn.GroupNorm(math.gcd(32, channels), channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            1,  # stride=1 for second conv
            padding,
            bias=False,
        )
        self.bn2 = nn.GroupNorm(math.gcd(32, channels), channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(math.gcd(32, channels), channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class PreActResidualBlock(nn.Module):
    """Pre-activation residual block (ResNet v2 style)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """__init__.

        Args:
            channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
        """
        super().__init__()
        self.bn1 = nn.GroupNorm(math.gcd(32, channels), channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn2 = nn.GroupNorm(math.gcd(32, channels), channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            1,  # stride=1 for second conv
            padding,
            bias=False,
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(math.gcd(32, channels), channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for PreActResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out


class UnifiedResidualBlock(nn.Module):
    """Unified residual block supporting multiple architectural variants.

    This block consolidates common residual patterns found across different
    generators, supporting various normalization, activation, and scaling
    options.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        use_group_norm: bool = True,
        activation: str = "leaky",
        skip_scale: float = 1.0,
        use_shortcut: bool = True,
        pre_activation: bool = False,
    ):
        """__init__.

        Args:
            channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (Optional[int]): Description.
            bias (bool): Description.
            use_group_norm (bool): Description.
            activation (str): Description.
            skip_scale (float): Description.
            use_shortcut (bool): Description.
            pre_activation (bool): Description.
        """
        super().__init__()
        self.skip_scale = skip_scale
        self.use_shortcut = use_shortcut
        self.pre_activation = pre_activation

        # Default padding for "same" convolution
        if padding is None:
            padding = kernel_size // 2
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(
                f"Unknown activation '{activation}'. Valid options: 'relu', 'leaky', 'gelu'."
            )

        # First convolution block
        conv_kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": bias,
        }

        if pre_activation and use_group_norm:
            self.bn1 = nn.GroupNorm(math.gcd(32, channels), channels)
            self.conv1 = nn.Conv2d(channels, channels, **conv_kwargs)
            self.bn2 = nn.GroupNorm(math.gcd(32, channels), channels)
            self.conv2 = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            )
        else:
            self.bn1 = (
                nn.GroupNorm(math.gcd(32, channels), channels) if use_group_norm else nn.Identity()
            )
            self.conv1 = nn.Conv2d(channels, channels, **conv_kwargs)
            self.bn2 = (
                nn.GroupNorm(math.gcd(32, channels), channels) if use_group_norm else nn.Identity()
            )
            self.conv2 = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            )

        # Shortcut connection for channel/dimension changes
        if use_shortcut:
            self.shortcut = nn.Sequential()
            if stride != 1 or channels != channels:  # Handle dimension changes
                shortcut_layers = []
                if stride != 1:
                    shortcut_layers.append(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=1,
                            stride=stride,
                            bias=bias,
                        )
                    )
                if use_group_norm:
                    shortcut_layers.append(nn.GroupNorm(math.gcd(32, channels), channels))
                self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through unified residual block.

        forward method for UnifiedResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = self.shortcut(x) if self.use_shortcut else x

        if self.pre_activation:
            # Pre-activation style (ResNet v2)
            out = self.bn1(x)
            out = self.act(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.act(out)
            out = self.conv2(out)
        else:
            # Standard residual style (ResNet v1)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)

            out = self.conv2(out)
            out = self.bn2(out)

        # Apply skip connection with optional scaling
        if self.skip_scale != 1.0:
            out = out * self.skip_scale

        out = out + residual

        # Final activation (only for non-pre-activation style)
        if not self.pre_activation:
            out = self.act(out)

        return out


__all__ = ["PreActResidualBlock", "ResidualBlock", "UnifiedResidualBlock"]
