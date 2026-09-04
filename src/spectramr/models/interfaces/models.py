"""Model Interfaces - SOLID Design
==============================

This module defines the core interfaces for all model types following SOLID
principles.
All concrete implementations must implement these interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch


class IModel(ABC):
    """Interface for all neural network models.

    Note: This inherits from ABC to ensure ABC metaclass is used.
    When combining with nn.Module, use ModuleABCMeta as the metaclass
    to resolve potential conflicts.

    Concrete defaults are provided for the introspection helpers
    (``name``, ``get_parameter_count``) so subclasses only need to
    implement the data-flow methods (``forward``, plus ``generate`` /
    ``get_output_shape`` for IGenerator). Subclasses are still free to
    override.
    """

    @property
    def name(self) -> str:
        """Returns the model name (defaults to the class name)."""
        return type(self).__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        params = getattr(self, "parameters", None)
        if not callable(params):
            return 0
        return int(sum(p.numel() for p in params()))


class IGenerator(IModel):
    """Interface for generator models.

    ``generate`` defaults to ``forward`` (most generators are deterministic
    feed-forward in this codebase) and ``get_output_shape`` defaults to
    the identity (input shape == output shape) which matches the typical
    reconstruction case. Architectures that change the spatial extent —
    super-resolution, hierarchical decoders — should override.
    """

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate samples from latent space (defaults to forward)."""
        return self.forward(x, **kwargs)  # type: ignore[arg-type]

    def get_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Returns the output shape for a given input shape (identity by default)."""
        return tuple(input_shape)


class IDiscriminator(IModel):
    """Interface for discriminator models."""

    @abstractmethod
    def discriminate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Discriminate real vs fake samples."""

    @abstractmethod
    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract feature maps for feature matching."""


class ITransformer(IModel):
    """Interface for transformer-based models."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""


class IDiffusionModel(IModel):
    """Interface for diffusion models."""

    @abstractmethod
    def diffuse(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply diffusion process."""

    @abstractmethod
    def denoise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply denoising process."""

    @abstractmethod
    def sample(self, shape: tuple[int, ...], **kwargs: Any) -> torch.Tensor:
        """Sample from the diffusion model."""


class IModelFactory(ABC):
    """Interface for model factories."""

    @abstractmethod
    def create_generator(
        self,
        model_type: str,
        **kwargs: Any,
    ) -> IGenerator:
        """Create a generator instance."""

    @abstractmethod
    def create_discriminator(
        self,
        model_type: str,
        **kwargs: Any,
    ) -> IDiscriminator:
        """Create a discriminator instance."""

    @abstractmethod
    def create_model_pair(
        self,
        model_type: str,
        **kwargs: Any,
    ) -> tuple[IGenerator, IDiscriminator | None]:
        """Create a generator and discriminator instance as a pair."""

    @abstractmethod
    def get_available_types(self) -> dict[str, str]:
        """Get available model types and descriptions."""


class IConfigurable(ABC):
    """Interface for configurable components."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""

    @abstractmethod
    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration."""
