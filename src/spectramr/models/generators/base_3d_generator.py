"""Standardized 3D Generator Base Class.

Provides a lightweight default implementation that is easy to extend and
safe to instantiate in unit tests. Concrete subclasses can override the
latent sampling or forward pass while reusing the utility helpers here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="base_3d", training_mode="reconstruction")
class Base3DGenerator(nn.Module, IGenerator):
    """Standardized base class for 3D generators.

    Provides consistent parameter interface and common functionality
    for all 3D model implementations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        output_activation: nn.Module | None = None,
        latent_shape: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        **kwargs,
    ):
        """Initialize 3D generator with standardized parameters.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Feature map sizes for each level
            output_activation: Output activation function
            latent_shape: Optional spatial shape for sampled latents
            device: Device for latent sampling utilities
            **kwargs: Additional model-specific parameters

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.output_activation = output_activation or nn.Tanh()

        default_latent_shape = tuple(latent_shape or (16, 32, 32))
        if len(default_latent_shape) != 3:
            raise ValueError(
                "latent_shape must contain depth, height, and width",
            )
        self.latent_shape = default_latent_shape
        self.device = torch.device(device or "cpu")

        # Default output spatial dimensions mirror the latent shape
        self.output_dims: tuple[int, int, int] | None
        self.output_dims = kwargs.get(
            "output_dims",
            tuple(int(dim) for dim in self.latent_shape),
        )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Simple identity forward pass with optional activation.

        Subclasses are expected to override this with the actual
        generator logic. The default keeps unit tests lightweight.

        forward method for Base3DGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""

        if x.ndim != 5:
            raise ValueError(
                "Expected latent tensor with shape (N, C, D, H, W)",
            )

        output = x
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

    def generate(
        self,
        latents: torch.Tensor | None = None,
        *,
        num_samples: int = 1,
        latent_shape: Sequence[int] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Generate 3D samples, sampling latents if none are provided."""

        if latents is None:
            latents = self._sample_latents(
                num_samples=num_samples,
                latent_shape=latent_shape,
            )

        if latents is None:
            raise ValueError("Latent tensors are required for generation")

        return self.forward(latents)

    @property
    def name(self) -> str:
        """Get descriptive model name."""
        return f"{self.__class__.__name__}"

    def get_parameter_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(
        self,
        input_shape: Sequence[int] | None = None,
    ) -> tuple[int, int, int, int, int]:
        """Calculate output shape for the provided or default input."""

        if input_shape is None:
            if self.output_dims is None:
                raise AttributeError(
                    "output_dims has not been configured for this generator",
                )
            depth, height, width = self.output_dims
            return (1, self.out_channels, depth, height, width)

        if len(input_shape) != 5:
            raise ValueError("Expected input shape of length 5")

        batch_size, _, depth, height, width = input_shape
        return (batch_size, self.out_channels, depth, height, width)

    def _sample_latents(
        self,
        *,
        num_samples: int = 1,
        latent_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Sample latents from a normal distribution for 3D generation."""

        shape = tuple(latent_shape or self.latent_shape)
        if len(shape) != 3:
            raise ValueError("latent_shape must contain three spatial dims")

        latent_tensor = torch.randn(
            (num_samples, self.in_channels, *shape),
            device=self.device,
        )
        return latent_tensor


def standardize_3d_params(
    params: dict[str, Any] | None = None,
    **legacy_kwargs: Any,
) -> dict[str, Any]:
    """Standardize parameter names for 3D generators.

    Converts legacy parameter names to standardized ones.

    Args:
        params: Optional dictionary of parameters
        **legacy_kwargs: Additional parameters supplied via kwargs

    Returns:
        dict: Parameters with standardized names

    """
    combined: dict[str, Any] = {}
    if params is not None:
        if not isinstance(params, dict):
            raise TypeError("params must be a mapping of configuration values")
        combined.update(params)

    combined.update(legacy_kwargs)

    required_spatial_keys = {"depth", "height", "width"}
    missing = required_spatial_keys - combined.keys()
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise KeyError(f"Missing required 3D parameter(s): {missing_keys}")

    for key in required_spatial_keys:
        value = combined[key]
        if not isinstance(value, int) or value <= 0:
            raise TypeError(
                f"Parameter '{key}' must be a positive integer, got {value!r}",
            )

    return {key: combined[key] for key in combined}
