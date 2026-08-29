"""Base U-Net Generator
====================

Abstract base class for U-Net style generators that provides common
architecture components and eliminates code duplication across U-Net variants.
"""

from abc import ABC
from typing import Any

import torch
from torch import nn

from ..interfaces.models import IGenerator


class BaseUNetGenerator(nn.Module, IGenerator, ABC):
    """Abstract base class for U-Net style generators.

    This class provides the common U-Net architecture components:
    - Encoder blocks with downsampling
    - Decoder blocks with upsampling and skip connections
    - Configurable depth and feature maps
    - Common utility methods

    Subclasses should implement the specific block types and
    attention mechanisms.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512, 1024),
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
        time_dim: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (tuple[int, ...]): Description.
            depth (int): Description.
            bilinear (bool): Description.
            dropout (float): Description.
            time_dim (Optional[int]): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.depth = depth
        self.bilinear = bilinear
        self.dropout = dropout
        self.time_dim = time_dim

        # Validate depth and features
        if len(features) < depth + 1:
            raise ValueError(
                f"features tuple must have at least {depth + 1} elements, got {len(features)}",
            )

        # Initialize encoder and decoder as ModuleLists
        # These will be populated by subclasses
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Output convolution - can be overridden by subclasses
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Initialize the architecture
        self._build_architecture()

    def _build_architecture(self) -> None:
        """Build the U-Net architecture.

        This method should be called by subclasses after they have
        defined their specific block types. It creates the encoder
        and decoder stacks.
        """
        # Create encoder blocks
        for i in range(self.depth + 1):
            # First block has no downsampling
            if i == 0:
                encoder_block = self._make_encoder_block(
                    self.in_channels,
                    self.features[i],
                    include_pool=False,
                )
            else:
                encoder_block = self._make_encoder_block(
                    self.features[i - 1],
                    self.features[i],
                    include_pool=True,
                )
            self.encoder.append(encoder_block)

        # Create decoder blocks
        for i in range(self.depth, 0, -1):
            decoder_block = self._make_decoder_block(
                self.features[i],
                self.features[i - 1],
            )
            self.decoder.append(decoder_block)

    def _make_encoder_block(
        self,
        in_channels: int,
        out_channels: int,
        include_pool: bool = True,
    ) -> nn.Module:
        """Create an encoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            include_pool: Whether to include max pooling

        Returns:
            Encoder block module

        """
        raise NotImplementedError(
            "Subclasses of BaseUNetGenerator must implement the "
            "`_make_encoder_block` method. This method should return a "
            "sequential module for a single encoder block.",
        )

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a decoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            Decoder block module

        """
        raise NotImplementedError(
            "Subclasses of BaseUNetGenerator must implement the "
            "`_make_decoder_block` method. This method should return a "
            "sequential module for a single decoder block.",
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the U-Net.

        Args:
            x: Input tensor
            timestep: Optional timestep for diffusion models
            cond: Optional conditioning tensor

        Returns:
            Output tensor

        forward method for BaseUNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timestep (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Store skip connections
        skip_connections = []

        # Encoder pass
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            if i < len(self.encoder) - 1:  # Don't store skip for last layer
                skip_connections.append(x)

        # Decoder pass with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, skip)

        # Final output convolution
        return self.out_conv(x)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate samples - for GAN compatibility."""
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        batch_size, _, height, width = input_shape
        return (batch_size, self.out_channels, height, width)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return self.__class__.__name__
