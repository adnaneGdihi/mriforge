"""Anisotropic Voxel Handling Generator
====================================

Hybrid 2D/3D architecture for handling anisotropic voxels in clinical MRI.
Addresses the common scenario where in-plane resolution (x,y) differs from
through-plane resolution (z), which is problematic for standard 3D CNNs.

This generator uses:
- 2D convolutions for high-resolution in-plane processing
- 1D convolutions for low-resolution through-plane processing
- Anisotropic pooling and upsampling operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class AnisotropicConv3D(nn.Module):
    """3D convolution that handles anisotropic voxels by separating in-plane and through-plane operations.

    Updated to use explicit 3D convolution to avoid artifacts from orthogonal decomposition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (1, 1, 1),
        anisotropic_factor: float = 1.0,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (tuple[int, int, int]): Description.
            stride (tuple[int, int, int]): Description.
            padding (tuple[int, int, int]): Description.
            anisotropic_factor (float): Description.
        """
        super().__init__()
        self.anisotropic_factor = anisotropic_factor

        # Explicit 3D convolution allows for true volumetric correlation
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Standardize normalization/activation to follow the conv if needed,
        # but typically this is done in blocks.
        # The previous implementation had norm/act INSIDE the layer.
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            Output tensor (B, C_out, D_out, H_out, W_out)
        """
        x = self.conv3d(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class AnisotropicUpsample3D(nn.Module):
    """Anisotropic 3D upsampling that handles different scales in different dimensions."""

    def __init__(
        self,
        scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0),
        mode: str = "trilinear",
    ):
        """__init__.

        Args:
            scale_factor (tuple[float, float, float]): Description.
            mode (str): Description.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample with anisotropic scaling.

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            Upsampled tensor
        """
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False if self.mode == "trilinear" else None,
        )


@register_model(name="anisotropic_voxel_generator", training_mode="reconstruction")
class AnisotropicVoxelGenerator(nn.Module, IGenerator):
    """Hybrid 2D/3D generator for anisotropic voxel handling.

    This architecture separates processing for in-plane (high resolution)
    and through-plane (low resolution) dimensions to better handle
    anisotropic voxels common in clinical MRI.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
        anisotropic_factor: float = 5.0,  # Typical slice thickness / in-plane resolution ratio
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
            num_levels (int): Description.
            anisotropic_factor (float): Description.
            dropout (float): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.anisotropic_factor = anisotropic_factor

        # Encoder: downsampling path
        self.encoder = nn.ModuleList()
        current_channels = base_channels

        # Initial 2D conv for input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        for _level in range(num_levels):
            self.encoder.append(
                nn.Sequential(
                    AnisotropicConv3D(
                        current_channels,
                        current_channels * 2,
                        anisotropic_factor=anisotropic_factor,
                    ),
                    nn.Dropout3d(dropout),
                    AnisotropicConv3D(
                        current_channels * 2,
                        current_channels * 2,
                        anisotropic_factor=anisotropic_factor,
                    ),
                )
            )
            current_channels *= 2

        # Decoder: upsampling path
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for _level in range(num_levels):
            # Upsampler
            self.upsamplers.append(
                AnisotropicUpsample3D(
                    scale_factor=(anisotropic_factor, 2.0, 2.0),
                )
            )

            # Decoder block
            self.decoder.append(
                nn.Sequential(
                    AnisotropicConv3D(
                        current_channels,
                        current_channels // 2,
                        anisotropic_factor=anisotropic_factor,
                    ),
                    nn.Dropout3d(dropout),
                    AnisotropicConv3D(
                        current_channels // 2,
                        current_channels // 4,
                        anisotropic_factor=anisotropic_factor,
                    ),
                )
            )
            current_channels //= 2

        # Output convolution
        self.output_conv = nn.Sequential(
            nn.Conv3d(current_channels, out_channels, 1),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid 2D/3D network.

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            Output tensor (B, C_out, D, H, W)
        """
        # Initial 2D processing of input slices
        b, c, d, h, w = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, D, C, H, W)
        x_2d = x_2d.view(b * d, c, h, w)  # (B*D, C, H, W)

        x_2d = self.input_conv(x_2d)  # (B*D, base_channels, H, W)

        # Reshape back to 3D for encoder
        _, c_enc, h_enc, w_enc = x_2d.shape
        x_enc = x_2d.view(b, d, c_enc, h_enc, w_enc)
        x_enc = x_enc.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, D, H, W)

        # Encoder
        encoder_features = []
        for encoder_block in self.encoder:
            x_enc = encoder_block(x_enc)
            encoder_features.append(x_enc)

        # Decoder with skip connections
        x_dec = encoder_features[-1]
        for i, (upsampler, decoder_block) in enumerate(
            zip(self.upsamplers, self.decoder, strict=False)
        ):
            x_dec = upsampler(x_dec)
            # Skip connection
            if i < len(encoder_features) - 1:
                skip = encoder_features[-(i + 2)]
                x_dec = torch.cat([x_dec, skip], dim=1)
            x_dec = decoder_block(x_dec)

        # Output
        output = self.output_conv(x_dec)
        return output

    def generate(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate method for IGenerator interface."""
        return self.forward(input_tensor)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "anisotropic_voxel_generator"

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        # Assume same spatial dimensions (anisotropic upsampling handled internally)
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_anisotropic_voxel_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 64,
    num_levels: int = 4,
    anisotropic_factor: float = 5.0,
    dropout: float = 0.1,
) -> AnisotropicVoxelGenerator:
    """Factory function for anisotropic voxel generator."""
    return AnisotropicVoxelGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_levels=num_levels,
        anisotropic_factor=anisotropic_factor,
        dropout=dropout,
    )
