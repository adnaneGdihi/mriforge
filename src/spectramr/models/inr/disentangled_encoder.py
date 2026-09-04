"""Disentangled Encoders for UMR Pillar I.

This module implements the encoders responsible for separating anatomical structure
from physics/contrast information using contrastive learning.
"""

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spectramr.models.registry import register_model


class AnatomyEncoder(nn.Module):
    """3D CNN Encoder for Anatomical Features (Structure)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 256,
        base_filters: int = 32,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_dim (int): Description.
            base_filters (int): Description.
        """
        super().__init__()
        self.out_dim = out_dim

        # Simple 3D ResNet-like encoder structure
        self.conv_in = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm3d(base_filters)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.blocks = nn.Sequential(
            self._make_layer(base_filters, base_filters * 2),  # -> /2
            self._make_layer(base_filters * 2, base_filters * 4),  # -> /4
            self._make_layer(base_filters * 4, base_filters * 8),  # -> /8
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_filters * 8, out_dim)

    def _make_layer(self, in_c: int, out_c: int) -> nn.Module:
        """_make_layer.

        Args:
            in_c (int): Description.
            out_c (int): Description.
        Returns:
            nn.Module: Description.
        """
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Float[Tensor, "B C D H W"]) -> Float[Tensor, "B OutDim"]:
        """forward.

        Args:
            x (Float[Tensor, 'B C D H W']): Description.
        Returns:
            Float[Tensor, 'B OutDim']: Description.

        forward method for AnatomyEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B OutDim']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.act(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class PhysicsEncoder(nn.Module):
    """3D CNN Encoder for Physics Features (Contrast).

    Typically shallower or smaller than AnatomyEncoder as contrast
    is a global style property.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 64,
        base_filters: int = 16,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_dim (int): Description.
            base_filters (int): Description.
        """
        super().__init__()
        self.out_dim = out_dim

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, stride=2, padding=1),  # /2
            nn.InstanceNorm3d(base_filters),
            nn.LeakyReLU(0.2),
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),  # /4
            nn.InstanceNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1),  # /8
            nn.InstanceNorm3d(base_filters * 4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.fc = nn.Linear(base_filters * 4, out_dim)

    def forward(self, x: Float[Tensor, "B C D H W"]) -> Float[Tensor, "B OutDim"]:
        """forward.

        Args:
            x (Float[Tensor, 'B C D H W']): Description.
        Returns:
            Float[Tensor, 'B OutDim']: Description.

        forward method for PhysicsEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B OutDim']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


@register_model(name="disentangled_encoder", training_mode="encoder")
class DisentangledEncoder(nn.Module):
    """Wrapper combining Anatomy and Physics encoders."""

    def __init__(
        self,
        anatomy_dim: int = 256,
        physics_dim: int = 64,
        in_channels: int = 1,
    ):
        """__init__.

        Args:
            anatomy_dim (int): Description.
            physics_dim (int): Description.
            in_channels (int): Description.
        """
        super().__init__()
        self.encoder_geo = AnatomyEncoder(in_channels, anatomy_dim)
        self.encoder_phy = PhysicsEncoder(in_channels, physics_dim)

    def forward(
        self, x: Float[Tensor, "B C D H W"]
    ) -> tuple[Float[Tensor, "B D_geo"], Float[Tensor, "B D_phy"]]:
        """Encode input into disentangled representations.

        Args:
            x: Input 3D volume.

        Returns:
            Tuple of (z_geo, z_phy).

        forward method for DisentangledEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[Float[Tensor, 'B D_geo'], Float[Tensor, 'B D_phy']]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        z_geo = self.encoder_geo(x)
        z_phy = self.encoder_phy(x)
        return z_geo, z_phy
