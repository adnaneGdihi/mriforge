"""EDSR (Enhanced Deep Super-Resolution) Generator
==============================================

EDSR generator implementation for super-resolution tasks.
Based on the paper "Enhanced Deep Residual Networks for Single
Image Super-Resolution" by Lim et al. (2017).

Features:
- Residual blocks with skip connections
- No batch normalization for better performance
- Efficient architecture for SR tasks
"""

import torch
from torch import nn

from spectramr.models.blocks import UnifiedResidualBlock
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="edsr", training_mode="reconstruction")
class EDSRGenerator(IGenerator, nn.Module):
    """EDSR generator for super-resolution tasks.

    Features:
    - Multiple residual blocks for deep feature extraction
    - Skip connections for gradient flow
    - No batch normalization for better performance
    - Efficient upsampling for SR tasks

    Args:
        in_channels: Number of input channels (default: 1 for grayscale MRI)
        out_channels: Number of output channels (default: 1)
        num_features: Number of feature maps in residual blocks (default: 64)
        num_blocks: Number of residual blocks (default: 16)
        scale_factor: Upsampling scale factor (default: 2)
        output_activation: Output activation function

    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 16,
        scale_factor: int = 2,
        skip_scale: float = 1.0,
        output_activation: nn.Module | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_features (int): Description.
            num_blocks (int): Description.
            scale_factor (int): Description.
            skip_scale (float): Description.
            output_activation (Optional[nn.Module]): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.scale_factor = scale_factor
        self.skip_scale = torch.clamp(torch.tensor(skip_scale), 0.1, 1.0).item()
        self.output_activation = output_activation or nn.Identity()

        # Initial feature extraction
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                UnifiedResidualBlock(
                    num_features,
                    use_group_norm=False,
                    skip_scale=self.skip_scale,
                    bias=True,
                )
                for _ in range(num_blocks)
            ],
        )

        # Feature fusion
        self.mid_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Upsampling
        if scale_factor == 2:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, 3, padding=1),
            )
        elif scale_factor == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, 3, padding=1),
            )
        else:
            # For other scale factors, use bilinear upsampling
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False,
                ),
                nn.Conv2d(num_features, out_channels, 3, padding=1),
            )

    @property
    def name(self) -> str:
        """Return model name."""
        return f"EDSRGenerator_{self.num_blocks}blocks_{self.scale_factor}x"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EDSR generator.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, out_channels, H*scale_factor, W*scale_factor]

        forward method for EDSRGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Initial feature extraction
        head = self.head(x)

        # Residual blocks with skip connections
        residual = head
        for block in self.residual_blocks:
            residual = block(residual)

        # Feature fusion
        residual = self.mid_conv(residual)
        residual += head  # Global skip connection

        # Upsampling
        out = self.upsampler(residual)

        # Apply output activation
        return self.output_activation(out)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate super-resolved image from low-resolution input.

        Args:
            z: Low-resolution input image
            **kwargs: Additional generation parameters

        Returns:
            Super-resolved output image

        """
        return self.forward(z)

    def get_output_shape(self, input_shape):
        """Calculate output shape for given input shape.

        Args:
            input_shape: Input tensor shape (N, C, H, W)

        Returns:
            Output tensor shape (N, out_channels, H*scale_factor,
            W*scale_factor)

        """
        n, _, h, w = input_shape
        return (n, self.out_channels, h * self.scale_factor, w * self.scale_factor)

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["EDSRGenerator"]
