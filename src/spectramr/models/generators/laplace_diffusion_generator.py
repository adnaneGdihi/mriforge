"""Laplace Diffusion Generator
==========================

Laplace diffusion generator implementing the IGenerator interface.
Uses Laplace noise instead of Gaussian noise for sharper feature modeling.
"""

import torch
from torch import nn

from spectramr.models.diffusion.laplace_diffusion import LaplaceDiffusion
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="laplace_diffusion_generator", training_mode="diffusion")
class LaplaceDiffusionGenerator(IGenerator, nn.Module):
    """Laplace diffusion generator implementing the IGenerator interface.
    Uses Laplace noise instead of Gaussian noise for modeling sharp features.
    """

    def __init__(
        self,
        denoising_model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str | None = None,
    ):
        """Initialize Laplace Diffusion Generator.

        Args:
            denoising_model: The neural network model that predicts noise
            timesteps: Number of diffusion timesteps
            beta_schedule: Schedule for beta values ("linear" or "cosine")
            device: Device to run on ("cuda" or "cpu")

        """
        super().__init__()
        self.denoising_model = denoising_model
        self.diffusion = LaplaceDiffusion(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            device=device,
        )

        # Move model to device
        self.denoising_model.to(self.diffusion.device)

    @property
    def name(self) -> str:
        """Return the model name."""
        return "LaplaceDiffusionGenerator"

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the denoising model.

        F7/E27 (2026-05-21 smoke audit): the wrapped
        ``EnhancedDeepDiffusionUNet`` requires a ``timesteps`` argument
        (forward signature ``(x, timesteps, cond=None)``). The earlier
        ``forward(self, x)`` signature dropped ``timesteps`` on the
        floor, crashing 2 experiments
        (experiment_73_laplace_diffusion + sibling) at runtime. The
        strategy's ``_call_generator_with_optional_kwargs`` path uses
        ``_callable_accepts_kwarg`` to introspect the signature; we
        must expose ``timesteps`` here for it to be passed through.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W].
            timesteps: Diffusion timestep tensor [B]. Required by
                EnhancedDeepDiffusionUNet; if None, fall back to a
                zeros vector so non-diffusion calls (e.g. EMA-style
                "generate at t=0") still work.
            **kwargs: Forwarded to the denoising model (e.g. cond).
        """
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.denoising_model(x, timesteps, **kwargs)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples using Laplace diffusion.

        Args:
            z: Input tensor (used to determine output shape)
            **kwargs: Additional arguments (unused for Laplace diffusion)

        Returns:
            Generated tensor

        """
        # Use z to determine the output shape
        shape = z.shape
        return self.diffusion.p_sample_loop(self.denoising_model, shape)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        # Laplace diffusion preserves the spatial dimensions
        return input_shape

    def get_parameter_count(self) -> int:
        """Count total parameters in the denoising model."""
        return sum(p.numel() for p in self.denoising_model.parameters() if p.requires_grad)
