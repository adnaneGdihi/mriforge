"""Base PatchGAN Discriminator
===========================

Abstract base class for PatchGAN style discriminators that provides
common architecture components and eliminates code duplication across
discriminator variants.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BasePatchGAN(nn.Module, ABC):
    """Abstract base class for PatchGAN discriminators.

    This class provides the common PatchGAN architecture components:
    - Multiple convolutional layers
    - Configurable depth and feature maps
    - Common utility methods

    Subclasses should implement the specific convolution block types.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512, 512),
        n_layers: int = 3,
        norm_layer: type | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            features (tuple[int, ...]): Description.
            n_layers (int): Description.
            norm_layer (Optional[type]): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.features = features
        self.n_layers = n_layers

        # Default normalization
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Build discriminator layers
        self.layers = nn.ModuleList()

        # First layer (no normalization)
        self.layers.append(
            self._make_conv_block(in_channels, features[0], normalize=False),
        )

        # Intermediate layers
        for i in range(1, n_layers + 1):
            in_feat = features[i - 1]
            out_feat = features[i] if i < len(features) else features[-1]
            self.layers.append(self._make_conv_block(in_feat, out_feat, normalize=True))

        # Final layer
        if n_layers < len(features):
            final_features = features[n_layers]
        else:
            final_features = features[-1]
        self.layers.append(
            nn.Conv2d(final_features, 1, kernel_size=4, stride=1, padding=1),
        )

    @abstractmethod
    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
    ) -> nn.Module:
        """Create a convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            normalize: Whether to include normalization

        Returns:
            Convolution block module

        """
        raise NotImplementedError(
            "Subclasses of BasePatchGAN must implement the `_make_conv_block` "
            "method. This method should return a sequential module "
            "representing a single convolutional block in the discriminator."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            x: Input tensor

        Returns:
            Discriminator output

        forward method for BasePatchGAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return self.__class__.__name__
