"""Diffusion Strategy Mixin

This module provides a mixin class for shared diffusion behavior
across different diffusion-based training strategies.
"""

from abc import ABC

import torch
from torch import nn

from mriforge.infrastructure.training.schedulers.diffusion_scheduler import (
    DiffusionScheduler,
)


class DiffusionStrategyMixin(ABC):
    """Mixin class providing shared diffusion behavior.

    This mixin provides common functionality for diffusion-based training
    strategies, including noise scheduling, timestep sampling, and
    diffusion process utilities.
    """

    def __init__(self):
        """Initialize diffusion mixin."""
        self._diffusion_initialized = False

    def initialize_diffusion_parameters(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: torch.device | None = None,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """Initialize diffusion parameters.

        ``beta_start`` / ``beta_end`` are forwarded to the scheduler. They used
        to be omitted here, so ``DiffusionScheduler`` fell back to its own
        defaults and ``training.diffusion.{beta_start,beta_end}`` was an unread
        knob (pitfall #15): a linear-schedule arm that widened the beta range
        trained on 1e-4..0.02 regardless of what its YAML declared.

        Args:
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Type of beta schedule ('linear' or 'cosine')
            device: Device for tensor operations
            beta_start: First beta of the linear schedule (unused by 'cosine')
            beta_end: Last beta of the linear schedule (unused by 'cosine')

        """
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.device = device or torch.device("cpu")

        # Setup scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            device=self.device,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self._diffusion_initialized = True

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for diffusion.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Timestep tensor of shape [batch_size]

        """
        return self.scheduler.sample_timesteps(batch_size)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0).

        Args:
            x_start: Clean input tensor
            t: Timestep tensor
            noise: Optional noise tensor

        Returns:
            Noisy tensor at timestep t

        """
        return self.scheduler.q_sample(x_start, t, noise)

    def get_diffusion_schedule_info(self) -> dict[str, torch.Tensor]:
        """Get information about the diffusion schedule.

        Returns:
            Dictionary with schedule tensors

        """
        if not self._diffusion_initialized:
            raise RuntimeError("Diffusion parameters not initialized")

        return self.scheduler.get_diffusion_schedule_info()

    def compute_diffusion_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion loss.

        Args:
            model_output: Model prediction
            target: Target tensor
            timesteps: Timestep tensor
            noise: Original noise tensor

        Returns:
            Loss tensor

        """
        # Simple MSE loss for diffusion
        return nn.functional.mse_loss(model_output, noise)

    def get_noise_prediction(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Extract noise prediction from model output.

        Default implementation assumes model directly predicts noise.
        Subclasses can override for different prediction formats.

        Args:
            model_output: Raw model output
            x_t: Noisy input at timestep t
            timesteps: Timestep tensor

        Returns:
            Predicted noise tensor

        """
        # Default: assume model output is predicted noise
        return model_output

    def denoise_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step.

        Args:
            x_t: Noisy tensor at timestep t
            predicted_noise: Predicted noise
            timesteps: Timestep tensor

        Returns:
            Denoised tensor at timestep t-1

        """
        return self.scheduler.denoise_step(x_t, predicted_noise, timesteps)
