"""Generator Protocols - Static Polymorphism
==========================================

This module defines Protocol classes for generators to enable static
polymorphism, avoiding vtable lookup costs and enabling compile-time
type verification.

Compliance:
- perf_design.md: Static Polymorphism via Protocols
- restruct.md: Interface Segregation
- unit-test.md: Strict type annotations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    pass


@runtime_checkable
class IGeneratorProtocol(Protocol):
    """Protocol for generator models.

    Using `typing.Protocol` instead of ABC enables static polymorphism,
    avoiding virtual method dispatch overhead at runtime.

    Attributes:
        name: Human-readable identifier for the generator.

    Methods:
        forward: Core tensor transformation (training).
        generate: Inference-time generation.
        get_output_shape: Calculate output dimensions.
        get_parameter_count: Count trainable parameters.
    """

    @property
    def name(self) -> str:
        """Returns the model name."""
        ...

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, D, H, W).
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            Output tensor.
        """
        ...

    def generate(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Generate samples from input.

        Args:
            x: Input tensor (conditioning or latent).
            **kwargs: Generation-specific arguments.

        Returns:
            Generated tensor.
        """
        ...

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Shape of output tensor.
        """
        ...

    def get_parameter_count(self) -> int:
        """Returns the total number of trainable parameters."""
        ...


@runtime_checkable
class IDiffusionGeneratorProtocol(IGeneratorProtocol, Protocol):
    """Protocol for diffusion-based generators.

    Extends IGeneratorProtocol with diffusion-specific methods.
    This enables type-safe handling of diffusion models without
    inheritance overhead.
    """

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Forward diffusion process: q(x_t | x_0).

        Args:
            x_start: Clean input tensor of shape (B, C, H, W).
            t: Timestep tensor of shape (B,).
            noise: Optional pre-sampled noise tensor.

        Returns:
            Noised tensor x_t.
        """
        ...

    def p_sample(
        self,
        x: Tensor,
        t: Tensor,
        t_index: int,
        **kwargs: Any,
    ) -> Tensor:
        """Reverse diffusion step: p(x_{t-1} | x_t).

        Args:
            x: Noised tensor at timestep t.
            t: Timestep tensor of shape (B,).
            t_index: Integer index of timestep.
            **kwargs: Additional sampling arguments.

        Returns:
            Denoised tensor at timestep t-1.
        """
        ...

    def p_sample_loop(
        self,
        shape: tuple[int, ...],
        **kwargs: Any,
    ) -> Tensor:
        """Full reverse diffusion sampling loop.

        Args:
            shape: Shape of output tensor (B, C, H, W).
            **kwargs: Sampling arguments (e.g., cond, guidance_scale).

        Returns:
            Generated tensor.
        """
        ...


@runtime_checkable
class ILatentGeneratorProtocol(IGeneratorProtocol, Protocol):
    """Protocol for latent-space generators (VAE, VQ-VAE).

    Extends IGeneratorProtocol with encode/decode methods.
    """

    def encode(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Encode input to latent representation.

        Args:
            x: Input tensor.

        Returns:
            Latent tensor or tuple of (z, mu, logvar) for VAEs.
        """
        ...

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to output.

        Args:
            z: Latent tensor.

        Returns:
            Decoded tensor.
        """
        ...


@runtime_checkable
class IPhysicsGeneratorProtocol(IGeneratorProtocol, Protocol):
    """Protocol for physics-informed generators.

    Extends IGeneratorProtocol with data consistency methods.
    """

    def apply_data_consistency(
        self,
        prediction: Tensor,
        kspace_measured: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply k-space data consistency.

        Args:
            prediction: Predicted image tensor.
            kspace_measured: Measured k-space data.
            mask: Undersampling mask.

        Returns:
            Data-consistent image tensor.
        """
        ...


__all__ = [
    "IDiffusionGeneratorProtocol",
    "IGeneratorProtocol",
    "ILatentGeneratorProtocol",
    "IPhysicsGeneratorProtocol",
]
