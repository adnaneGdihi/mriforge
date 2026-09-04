"""Block interfaces and abstract base classes."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class IBlockStrategy(nn.Module, ABC):
    """Abstract base for all neural network blocks.

    This interface defines the contract that all blocks must satisfy.
    Implementations should inherit from this class and implement the
    abstract forward method.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass for the block.

        Args:
            x: Input tensor
            *args: Positional arguments for the block
            **kwargs: Keyword arguments for block
                (e.g., 'context' for cross-attention)

        Returns:
            Output tensor from the block

        forward method for IBlockStrategy.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""


class IFusionStrategy(IBlockStrategy, ABC):
    """Abstract base for fusion blocks that combine multiple inputs.

    This interface extends IBlockStrategy for blocks that need to fuse
    multiple input tensors rather than process a single tensor.
    """

    @abstractmethod
    def forward(
        self,
        inputs: list[torch.Tensor],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass for fusion block.

        Args:
            inputs: List of input tensors to fuse
            weights: Optional weights for each input (learned or provided)
            **kwargs: Additional keyword arguments

        Returns:
            Fused output tensor
        """
