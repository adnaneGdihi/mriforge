"""IMDN (Information Multi-distillation Network) Generator
======================================================

Implementation of the Information Multi-distillation Network for image
super-resolution. IMDN uses information multi-distillation blocks for
efficient and effective feature learning.

Reference:
- "Lightweight Image Super-Resolution with Information Multi-distillation
  Network" by Hui Zou, Hongxing Gao, Wei Cao, Yuqian Zhou
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class IMDNBlock(nn.Module):
    """Information Multi-distillation Block."""

    def __init__(
        self,
        n_feat: int,
        kernel_size: int = 3,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        """__init__.

        Args:
            n_feat (int): Description.
            kernel_size (int): Description.
            act (nn.Module): Description.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2)
        self.conv4 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2)
        self.conv5 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2)
        self.confusion = nn.Conv2d(n_feat * 4, n_feat, 1, padding=0)

        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through IMDN block.

        forward method for IMDNBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # First branch
        branch1 = self.act(self.conv1(x))

        # Second branch
        branch2 = self.act(self.conv2(branch1))

        # Third branch
        branch3 = self.act(self.conv3(branch2))

        # Fourth branch
        branch4 = self.act(self.conv4(branch3))

        # Fifth branch
        branch5 = self.act(self.conv5(branch4))

        # Information distillation
        concat = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        distilled = self.confusion(concat)

        # Residual connection
        out = distilled + branch5

        return out + x


@register_model(name="imdn", training_mode="reconstruction")
class IMDNGenerator(nn.Module, IGenerator):
    """Information Multi-distillation Network Generator for image
    super-resolution.

    This implementation follows the IMDN architecture with information
    multi-distillation blocks for efficient feature learning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_feat: int = 64,
        n_blocks: int = 6,
        scale: int = 4,
    ):
        """Initialize IMDN generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_feat: Number of feature channels
            n_blocks: Number of IMDN blocks
            scale: Upscaling factor

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        # Feature extraction
        self.conv_first = nn.Conv2d(in_channels, n_feat, 3, padding=1)

        # IMDN blocks
        self.imdn_blocks = nn.ModuleList([IMDNBlock(n_feat) for _ in range(n_blocks)])

        # Feature fusion
        self.conv_after_body = nn.Conv2d(n_feat, n_feat, 3, padding=1)

        # Upsampling
        if scale == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feat, n_feat * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )
        elif scale == 2:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )
        elif scale == 1:
            self.upsampler = nn.Identity()
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

        # Output
        self.conv_last = nn.Conv2d(n_feat, out_channels, 3, padding=1)

        # Skip connection
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through IMDN generator.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Output tensor of shape [N, out_channels, H*scale, W*scale]

        forward method for IMDNGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Skip connection
        skip = self.skip_conv(x)

        # Feature extraction
        feat = self.conv_first(x)

        # IMDN blocks
        res = feat
        for block in self.imdn_blocks:
            res = block(res)

        # Feature fusion
        res = self.conv_after_body(res)
        feat = feat + res

        # Upsampling
        feat = self.upsampler(feat)

        # Output
        out = self.conv_last(feat)

        # Skip connection
        out = out + F.interpolate(
            skip,
            scale_factor=self.get_scale_factor(),
            mode="bicubic",
            align_corners=False,
        )

        return out

    def get_scale_factor(self) -> int:
        """Get the upscaling factor."""
        if hasattr(self, "upsampler"):
            if isinstance(self.upsampler, nn.Identity):
                return 1
            if len(self.upsampler) == 2:  # scale=2
                return 2
            if len(self.upsampler) == 4:  # scale=4
                return 4
        return 1

    @property
    def name(self) -> str:
        """Return the model name."""
        return "IMDNGenerator"

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for IMDN, z is the input image."""
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        n, _, h, w = input_shape
        return (n, self.out_channels, h * self.scale, w * self.scale)

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return self.num_parameters

    @property
    def num_parameters(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_imdn_generator(
    in_channels: int = 3,
    out_channels: int = 3,
    n_feat: int = 64,
    n_blocks: int = 6,
    scale: int = 4,
) -> IMDNGenerator:
    """Factory function to create IMDN generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_feat: Number of feature channels
        n_blocks: Number of IMDN blocks
        scale: Upscaling factor

    Returns:
        IMDNGenerator instance

    """
    return IMDNGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        n_feat=n_feat,
        n_blocks=n_blocks,
        scale=scale,
    )
