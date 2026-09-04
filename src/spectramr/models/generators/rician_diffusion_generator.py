"""Rician Diffusion Generator
=========================

Rician diffusion generator implementing the IGenerator interface.
Uses Rician noise for modeling MRI-specific noise characteristics.
"""

import torch
from torch import nn

from spectramr.models.diffusion.rician_diffusion import RicianDiffusion
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="rician_diffusion", training_mode="diffusion")
class RicianDiffusionGenerator(IGenerator, nn.Module):
    """Rician diffusion generator implementing the IGenerator interface.
    Uses Rician noise which is characteristic of MRI magnitude images.
    """

    def __init__(
        self,
        denoising_model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str | None = None,
        v_min: float = 0.1,
        v_max: float = 10.0,
        v_schedule: str = "linear",
    ):
        """Initialize Rician Diffusion Generator.

        Args:
            denoising_model: The neural network model that predicts noise
            timesteps: Number of diffusion timesteps
            beta_schedule: Schedule for beta values ("linear" or "cosine")
            device: Device to run on ("cuda" or "cpu")
            v_min: Minimum non-centrality parameter for Rician distribution
            v_max: Maximum non-centrality parameter for Rician distribution
            v_schedule: Schedule for non-centrality parameter

        """
        super().__init__()
        self.denoising_model = denoising_model
        self.diffusion = RicianDiffusion(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            device=device,
            v_min=v_min,
            v_max=v_max,
            v_schedule=v_schedule,
        )

        # Move model to device
        self.denoising_model.to(self.diffusion.device)

    @property
    def name(self) -> str:
        """Return the model name."""
        return "RicianDiffusionGenerator"

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the denoising model.

        F7/E27 (2026-05-21 smoke audit): same fix as the Laplace
        generator — expose ``timesteps`` so
        ``_callable_accepts_kwarg`` in the diffusion strategy sees it
        and threads through. Crashed
        experiment_74_rician_diffusion_for_magnitude_mri without this.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W].
            timesteps: Diffusion timestep tensor [B]. Required by
                EnhancedDeepDiffusionUNet.
            **kwargs: Forwarded to the denoising model.
        """
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.denoising_model(x, timesteps, **kwargs)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples using Rician diffusion.

        Args:
            z: Input tensor (used to determine output shape)
            **kwargs: Additional arguments (unused for Rician diffusion)

        Returns:
            Generated tensor

        """
        # Use z to determine the output shape
        shape = z.shape
        return self.diffusion.p_sample_loop(self.denoising_model, shape)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        # Rician diffusion preserves the spatial dimensions
        return input_shape

    def get_parameter_count(self) -> int:
        """Count total parameters in the denoising model."""
        return sum(p.numel() for p in self.denoising_model.parameters() if p.requires_grad)
