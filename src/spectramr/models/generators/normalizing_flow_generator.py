"""Normalizing Flow Generator
==========================

A generator based on normalizing flows, providing invertible transformations
and exact likelihood estimation.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="normalizing_flow", training_mode="reconstruction")
class NormalizingFlowGenerator(nn.Module, IGenerator):
    """Normalizing Flow Generator with Affine Coupling Layers.

    This model implements a sequence of invertible transformations to map
    between a simple distribution (latent space) and the data distribution.
    Uses Real-NVP style affine coupling layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        num_flows: int = 4,
        **kwargs,
    ):
        """Initialize Normalizing Flow Generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (must match in_channels for flows usually)
            hidden_channels: Hidden dimension
            num_flows: Number of flow steps
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Affine Coupling layers (Real-NVP style)
        # Each layer splits channels, transforms half conditioned on other half
        self.flows = nn.ModuleList()
        for i in range(num_flows):
            # Alternate which half is transformed
            self.flows.append(
                AffineCouplingLayer(in_channels, hidden_channels, reverse_split=(i % 2 == 1))
            )

        # Final projection if out_channels differs (though flows usually preserve dim)
        if in_channels != out_channels:
            self.final_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x: torch.Tensor, reverse: bool = False, *args, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor
            reverse: Whether to run in reverse (generation) or forward (inference)

        Returns:
            Transformed tensor

        forward method for NormalizingFlowGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            reverse (bool): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        log_det_sum = 0.0

        if not reverse:
            # Forward: Data -> Latent
            for flow in self.flows:
                x, log_det = flow(x, reverse=False)
                log_det_sum = log_det_sum + log_det
        else:
            # Reverse: Latent -> Data
            for flow in reversed(self.flows):
                x, log_det = flow(x, reverse=True)
                log_det_sum = log_det_sum + log_det

        return self.final_conv(x)


class AffineCouplingLayer(nn.Module):
    """Affine Coupling Layer (Real-NVP style).

    Splits input channels, transforms one half conditioned on the other.
    Exactly invertible with tractable log-determinant.
    """

    def __init__(self, channels: int, hidden_channels: int, reverse_split: bool = False):
        """__init__.

        Args:
            channels (int): Description.
            hidden_channels (int): Description.
            reverse_split (bool): Description.
        """
        super().__init__()
        self.reverse_split = reverse_split
        half_channels = max(channels // 2, 1)
        other_half = channels - half_channels

        # Network to compute scale and shift from conditioning half
        self.net = nn.Sequential(
            nn.Conv2d(other_half, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, half_channels * 2, 3, padding=1),  # scale + shift
        )

        # Initialize to identity transform
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

        self.half_channels = half_channels
        self.other_half = other_half

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple:
        """Apply affine coupling.

        Returns:
            (transformed_x, log_det_jacobian)

        forward method for AffineCouplingLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            reverse (bool): Expected input tensor.

        Returns:
            tuple: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.reverse_split:
            x1, x2 = x[:, self.half_channels :], x[:, : self.half_channels]
        else:
            x1, x2 = x[:, : self.other_half], x[:, self.other_half :]

        # Compute scale and shift from x1
        params = self.net(x1)
        log_scale, shift = params.chunk(2, dim=1)
        log_scale = torch.tanh(log_scale)  # Bound scale for stability

        if not reverse:
            # Forward: y2 = x2 * exp(log_scale) + shift
            y2 = x2 * torch.exp(log_scale) + shift
            log_det = log_scale.sum(dim=[1, 2, 3])
        else:
            # Reverse: x2 = (y2 - shift) * exp(-log_scale)
            y2 = (x2 - shift) * torch.exp(-log_scale)
            log_det = -log_scale.sum(dim=[1, 2, 3])

        if self.reverse_split:
            y = torch.cat([y2, x1], dim=1)
        else:
            y = torch.cat([x1, y2], dim=1)

        return y, log_det

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        # Usually generation involves sampling from latent and running reverse
        # Here we assume x is the latent input
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x, reverse=True)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "normalizing_flow"
