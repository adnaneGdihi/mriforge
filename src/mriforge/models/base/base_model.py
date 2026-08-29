# src/models/base_model.py

from typing import Any

import torch
from torch import nn

from ..interfaces.models import IDiscriminator, IGenerator


class BaseGenerator(nn.Module, IGenerator):
    """Abstract base class for generator models.
    Ensures a common interface for all generators and provides
    common functionality.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for BaseGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        raise NotImplementedError(
            "The `forward` method must be implemented by the subclass. "
            "This method defines the core logic of the generator model.",
        )

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Default generate method - can be overridden by subclasses."""
        return self.forward(x, **kwargs)

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        # Default implementation - subclasses should override for
        # specific behavior
        batch_size, _, *spatial_dims = input_shape
        return (batch_size, self.out_channels, *spatial_dims)

    @property
    def name(self) -> str:
        """Return the class name as default model name."""
        return self.__class__.__name__


class BaseDiscriminator(nn.Module, IDiscriminator):
    """Abstract base class for discriminator models.
    Ensures a common interface for all discriminators.
    """

    def __init__(self, in_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for BaseDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        raise NotImplementedError(
            "The `forward` method must be implemented by the subclass. "
            "This method defines the core logic of the discriminator model.",
        )

    def discriminate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Discriminate method for IDiscriminator interface."""
        return self.forward(x, **kwargs)

    def get_feature_maps(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Return feature maps (default empty dict if not supported)."""
        return {}

    @property
    def name(self) -> str:
        """Return the class name as default model name."""
        return self.__class__.__name__
