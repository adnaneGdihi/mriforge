"""Upsampling Building Blocks
========================

Common upsampling patterns used across different architectures.
"""

import torch
from torch import nn


class UnifiedUpsampleBlock(nn.Module):
    """Unified upsampling block supporting multiple upsampling methods.

    Consolidates common upsampling patterns found across different generators,
    supporting bilinear, transposed conv, pixel shuffle, and nearest neighbor methods.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        method: str = "bilinear",
        kernel_size: int = 3,
        bias: bool = False,
        use_conv: bool = True,
        activation: str = "relu",
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            scale_factor (int): Description.
            method (str): Description.
            kernel_size (int): Description.
            bias (bool): Description.
            use_conv (bool): Description.
            activation (str): Description.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.method = method
        self.use_conv = use_conv

        # Activation function
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "none":
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(inplace=True)

        # Upsampling method
        if method == "bilinear":
            self.upsample = nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
        elif method == "nearest":
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        elif method == "trilinear":
            self.upsample = nn.Upsample(
                scale_factor=scale_factor, mode="trilinear", align_corners=False
            )
        elif method == "transpose":
            self.upsample = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=scale_factor,
                stride=scale_factor,
                bias=bias,
            )
        elif method == "pixel_shuffle":
            # Pixel shuffle requires specific channel arrangement
            hidden_channels = in_channels * (scale_factor**2)
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias),
                nn.PixelShuffle(scale_factor),
            )
        else:
            raise ValueError(f"Unknown upsampling method: {method}")

        # Optional convolution after upsampling
        if use_conv:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=bias,
            )
        else:
            self.conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through unified upsampling block.

        forward method for UnifiedUpsampleBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.upsample(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class ResidualUpsampleBlock(nn.Module):
    """Upsampling block with residual connection.

    Combines upsampling with a residual connection for better gradient flow
    during training, commonly used in super-resolution architectures.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        method: str = "bilinear",
        kernel_size: int = 3,
        bias: bool = False,
        use_residual: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            scale_factor (int): Description.
            method (str): Description.
            kernel_size (int): Description.
            bias (bool): Description.
            use_residual (bool): Description.
        """
        super().__init__()
        self.use_residual = use_residual

        # Main upsampling path
        self.upsample = UnifiedUpsampleBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            method=method,
            kernel_size=kernel_size,
            bias=bias,
            activation="relu",
        )

        # Residual connection
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            self.residual_upsample = nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
        elif use_residual:
            self.residual_conv = nn.Identity()
            self.residual_upsample = nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
        else:
            self.residual_conv = nn.Identity()
            self.residual_upsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection.

        forward method for ResidualUpsampleBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = self.residual_upsample(self.residual_conv(x))

        out = self.upsample(x)

        if self.use_residual:
            out = out + residual

        return out


__all__ = ["ResidualUpsampleBlock", "UnifiedUpsampleBlock"]
