"""Base Blocks Module
==================

Contains abstract base classes for architectural blocks.
"""

import torch
from torch import nn

from mriforge.models.interfaces.models import IModel


class BaseGANBlock(nn.Module, IModel):
    """Base class for all GAN blocks following SOLID principles."""

    def __init__(self):
        """__init__."""
        super().__init__()

    @property
    def name(self) -> str:
        """Returns the block name."""
        return self.__class__.__name__

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the block."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GAN block.

        This method must be implemented by all subclasses to define the
        block's specific behavior.

        forward method for BaseGANBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        raise NotImplementedError(
            "Subclasses of BaseGANBlock must implement the `forward` method.",
        )
