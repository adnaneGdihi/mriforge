"""Tensor Train Decomposition Layers

Implements Tensor Train (TT) decomposition for neural network compression.
"""

import torch
import torch.nn as nn


class TTConv2d(nn.Module):
    """Conv2d layer with Tensor Train decomposition for compression."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        tt_ranks: list[int] | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            tt_ranks (Optional[list[int]]): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Default TT ranks for moderate compression
        if tt_ranks is None:
            rank = max(4, min(in_channels, out_channels) // 4)
            tt_ranks = [1, rank, rank, 1]

        self.tt_ranks = tt_ranks

        # TT cores (Low-rank decomposition of matricized kernel)
        # Core 1: (In * K * K, Rank)
        rank = tt_ranks[1]
        self.core1 = nn.Parameter(torch.randn(in_channels * kernel_size * kernel_size, rank))
        # Core 2: (Rank, Out)
        self.core2 = nn.Parameter(torch.randn(rank, out_channels))

        self.cached_weight = None
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize TT cores."""
        nn.init.xavier_normal_(self.core1)
        nn.init.xavier_normal_(self.core2)

    def train(self, mode: bool = True):
        """Set training mode and invalidate cache."""
        super().train(mode)
        self.cached_weight = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using unfolded convolution.

        forward method for TTConv2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Reconstruct approximate weight tensor
        if self.training or self.cached_weight is None:
            # (In*K*K, Rank) @ (Rank, Out) -> (In*K*K, Out)
            weight = torch.mm(self.core1, self.core2)

            # Reshape to (In, K, K, Out)
            weight = weight.view(
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                self.out_channels,
            )

            # Permute to (Out, In, K, K)
            weight = weight.permute(3, 0, 1, 2).contiguous()

            self.cached_weight = weight

        return nn.functional.conv2d(
            x,
            self.cached_weight,
            None,
            self.stride,
            self.padding,
        )

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_params = self.out_channels * self.in_channels * self.kernel_size * self.kernel_size
        compressed_params = self.core1.numel() + self.core2.numel()
        return original_params / compressed_params


class TTLinear(nn.Module):
    """Linear layer with Tensor Train decomposition."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tt_rank: int = 8,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            tt_rank (int): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank

        # TT cores
        self.core1 = nn.Parameter(torch.randn(in_features, tt_rank))
        self.core2 = nn.Parameter(torch.randn(tt_rank, out_features))

        self._initialize_weights()

    def _initialize_weights(self):
        """_initialize_weights.

        Returns:
            Any: Description.
        """
        nn.init.xavier_normal_(self.core1)
        nn.init.xavier_normal_(self.core2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        forward method for TTLinear.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # x @ core1 @ core2 = x @ W
        return torch.mm(torch.mm(x, self.core1), self.core2)

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_params = self.in_features * self.out_features
        compressed_params = self.core1.numel() + self.core2.numel()
        return original_params / compressed_params
