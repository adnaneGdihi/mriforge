import math

import torch
import torch.nn.functional as F
from torch import nn

# The import path is relative to the current file's location.
# The import path is relative to the current file's location.
from mriforge.models.layers.kan.kan_convs.fast_kan_conv import FastKANConv2DLayer
from mriforge.models.layers.kan.kan_convs.wav_kan import WavKANConv2DLayer


class DoubleConvWithKAN(nn.Module):
    """(convolution => [GroupNorm] => LeakyReLU) * 2, using KAN layers."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        time_emb_dim=None,
        kan_type="FastKAN",  # "FastKAN" or "WavKAN"
        **kan_kwargs,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            mid_channels (Any): Description.
            time_emb_dim (Any): Description.
            kan_type (Any): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Pop time_emb_dim from kan_kwargs to avoid passing it to KAN layers
        kan_kwargs.pop("time_emb_dim", None)

        if kan_type == "WavKAN":
            KANLayer = WavKANConv2DLayer
        else:
            KANLayer = FastKANConv2DLayer

        # Using GroupNorm as it is generally more stable for GANs/Diffusion than BatchNorm
        self.double_conv = nn.Sequential(
            KANLayer(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                **kan_kwargs,
            ),
            nn.GroupNorm(
                math.gcd(32, mid_channels) if mid_channels > 0 else 1,
                mid_channels,
            ),
            nn.LeakyReLU(0.2, inplace=False),
            KANLayer(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                **kan_kwargs,
            ),
            nn.GroupNorm(
                math.gcd(32, out_channels) if out_channels > 0 else 1,
                out_channels,
            ),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x, t_emb=None):
        """forward.

        Args:
            x (Any): Description.
            t_emb (Any): Description.
        Returns:
            Any: Description.

        forward method for DoubleConvWithKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.double_conv(x)


class DownWithKAN(nn.Module):
    """Downscaling with maxpool then DoubleConvWithKAN."""

    def __init__(self, in_channels, out_channels, time_emb_dim=None, **kan_kwargs):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            time_emb_dim (Any): Description.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWithKAN(
                in_channels,
                out_channels,
                time_emb_dim=time_emb_dim,
                **kan_kwargs,
            ),
        )

    def forward(self, x, t_emb=None):
        """forward.

        Args:
            x (Any): Description.
            t_emb (Any): Description.
        Returns:
            Any: Description.

        forward method for DownWithKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.maxpool_conv(x)


class UpWithKAN(nn.Module):
    """Upscaling with ConvTranspose2d then DoubleConvWithKAN."""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=False,
        time_emb_dim=None,
        **kan_kwargs,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            bilinear (Any): Description.
            time_emb_dim (Any): Description.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConvWithKAN(
            in_channels,
            out_channels,
            time_emb_dim=time_emb_dim,
            **kan_kwargs,
        )

    def forward(self, x1, x2, t_emb=None):
        """forward.

        Args:
            x1 (Any): Description.
            x2 (Any): Description.
            t_emb (Any): Description.
        Returns:
            Any: Description.

        forward method for UpWithKAN.

        Executes PyTorch tensor operations.

        Args:
            x1 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            x2 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t_emb)
