#!/usr/bin/env python3
"""Latent Input Wrapper
Wrapper class that allows VAE/VQ-VAE models to accept latent vectors directly.
"""

import torch
from torch import nn

from spectramr.models.interfaces import IGenerator
from spectramr.models.registry import register_model
from spectramr.shared.utils.metaclass import ModuleABCMeta


@register_model(name="latent_input_wrapper", training_mode="wrapper")
class LatentInputWrapper(nn.Module, IGenerator, metaclass=ModuleABCMeta):
    """Wrapper that allows VAE/VQ-VAE models to accept latent vectors as input.

    This wrapper takes a latent vector z ∈ R^{N×D} and passes it directly
    to the model's decode method, bypassing the encoder path.
    """

    def __init__(self, model: nn.Module, latent_dim: int):
        """Initialize the latent input wrapper.

        Args:
            model: The underlying VAE/VQ-VAE model with a decode method
            latent_dim: Dimension of the latent space

        """
        super().__init__()
        self.model = model
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            z: Latent vector [N, latent_dim]

        Returns:
            Reconstructed output [N, C, H, W]

        forward method for LatentInputWrapper.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.model.decode(z)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate output from latent vector (alias for forward).

        Args:
            z: Latent vector [N, latent_dim]

        Returns:
            Generated output [N, C, H, W]

        """
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given latent input shape.

        Args:
            input_shape: Shape of latent input (N, latent_dim)

        Returns:
            Output shape (N, C, H, W)

        """
        # For latent input, we need to infer the spatial dimensions
        # This is a simplified assumption - in practice, you'd need to # IMPL
        # query the model's decoder for the actual output shape
        batch_size = input_shape[0]
        # Assume standard output dimensions for MRI
        return (batch_size, 1, 256, 256)

    def get_parameter_count(self) -> int:
        """Get total parameter count of the wrapped model."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def name(self) -> str:
        """Get the name of the wrapped model."""
        return f"LatentInputWrapper({self.model.__class__.__name__})"
