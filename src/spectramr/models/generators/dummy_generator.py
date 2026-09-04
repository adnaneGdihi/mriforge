"""Dummy Generator for Testing
===========================

Minimal implementation of IGenerator for fast smoke testing.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="dummy", training_mode="experimental")
class DummyGenerator(IGenerator, nn.Module):
    """Minimal generator that projects channels using 1x1 convolution.

    Used for:
    - Smoke tests
    - Pipeline validation
    - CI/CD checks
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Simple 1x1 convolution to match channel dimensions
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "dummy_generator"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Handle 2D and 3D inputs (treat 3D as batch of 2D or use Conv3d if strictly needed)
        # For simplicity in "dummy", we assume 2D (B, C, H, W) mostly.
        # If input is (B, C, D, H, W), we might need to handle it.

        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DummyGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.ndim == 5:
            # (B, C, D, H, W) -> Proj (B, C, D, H, W) via Conv3d or reshaping
            # Let's just use Conv3d for 5D input dynamically if needed, or simple projection
            if not hasattr(self, "proj3d"):
                self.proj3d = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1).to(
                    x.device
                )
            return self.proj3d(x)

        return self.proj(x)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from input (alias for forward in dummy)."""
        return self.forward(x, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        # Preserves spatial dims, changes channels
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        dims = list(input_shape)
        dims[1] = self.out_channels
        return tuple(dims)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
