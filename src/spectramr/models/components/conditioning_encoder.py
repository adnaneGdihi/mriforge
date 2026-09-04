"""Conditioning Encoder Implementation
===================================

This module implements a lightweight CNN encoder to compress ULF MRI images into
the LDM's latent space for conditioning.

It maps:
    Input: (B, C_in, H, W) -> e.g., (B, 3, 256, 256)
    Output: (B, C_latent, H_latent, W_latent) -> e.g., (B, 4, 32, 32) (for f=8)
"""

import torch
import torch.nn as nn

from spectramr.models.registry import register_model


class DownBlock(nn.Module):
    """Simple downsampling block: Conv -> GroupNorm -> SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        num_groups: int = 32,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            num_groups (int): Description.
        """
        super().__init__()
        # Use simple strided convolution for downsampling
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # GroupNorm is preferred over BatchNorm for LDM conditioning
        # Handle case where channels < num_groups
        groups = num_groups if out_channels >= num_groups else max(1, out_channels // 2)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DownBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.act(self.norm(self.conv(x)))


@register_model(name="conditioning_encoder", training_mode="diffusion")
class ConditioningEncoder(nn.Module):
    """Conditioning Encoder for Latent Diffusion Models.

    compresses ULF input images to match the dimensions of the VAE latent space.
    Used for 'concat-conditioning' in LDM training.

    Args:
        in_channels: Number of input channels (e.g., 3 for 2.5D slab)
        out_channels: Number of latent channels (e.g., 4 to match VAE)
        base_channels: Base number of filters (e.g., 64)
        num_levels: Number of downsampling levels (e.g., 3 for 8x downsampling: 256->32)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        base_channels: int = 64,
        num_levels: int = 3,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
            num_levels (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = []

        # Initial projection (maintain resolution)
        current_channels = base_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, current_channels, 3, padding=1, bias=False),
                nn.GroupNorm(
                    32 if current_channels % 32 == 0 else max(1, current_channels // 2),
                    current_channels,
                ),
                nn.SiLU(inplace=True),
            )
        )

        # Downsampling blocks
        # Each level reduces spatial dim by 2x
        for i in range(num_levels):
            # Double channels at each step, maxing out at reasonable limit (e.g., 512)
            next_channels = min(current_channels * 2, 512)
            layers.append(
                DownBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    stride=2,  # Downsample
                )
            )
            current_channels = next_channels

        # Final projection to exact latent channels (e.g., 4)
        # Use 3x3 conv to mix features one last time
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """_init_weights.

        Returns:
            Any: Description.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            Conditioning encodings (B, out_channels, H/2^levels, W/2^levels)
        """
        return self.net(x)

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Calculate output shape given input shape."""
        # Simple dry run for shape calculation
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            output = self.forward(dummy_input)
        return tuple(output.shape[1:])
