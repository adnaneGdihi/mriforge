"""Diffusion Mixin - Reusable Diffusion Logic
============================================

This mixin encapsulates shared diffusion methods (q_sample, p_sample,
p_sample_loop) to reduce code duplication across 15+ diffusion generators.

Compliance:
- soft_design.md: Composition over Inheritance (Mixins)
- perf_design.md: Stateless where possible
- perf.md: Pre-allocate buffers, torch.compile friendly (no graph breaks)
- unit-test.md: Strict type hints
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DiffusionMixin:
    """Mixin providing shared diffusion methods for diffusion generators.

    This mixin provides default implementations of:
    - Noise schedules (linear, cosine)
    - Forward diffusion q_sample
    - Reverse sampling p_sample, p_sample_loop
    - Utility extraction

    The mixin is designed to be stateless where possible (`perf_design.md`),
    with state stored via `register_buffer` for GPU transfer.

    Subclasses must:
    - Implement `_predict_x0_from_noise` or `_predict_noise`
    - Call `_initialize_diffusion_buffers` in `__init__`
    """

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------
    timesteps: int
    beta_schedule: str

    # ---------------------------------------------------------------------------
    # Buffers (to be registered via _initialize_diffusion_buffers)
    # ---------------------------------------------------------------------------
    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    sqrt_recip_alphas: Tensor
    posterior_variance: Tensor

    def _initialize_diffusion_buffers(
        self: nn.Module,  # type: ignore[arg-type]
        timesteps: int = 1000,
        beta_schedule: str = "linear",
    ) -> None:
        """Initialize diffusion schedule buffers.

        This method should be called in the subclass's `__init__` method.
        Buffers are registered to automatically move to GPU with the model.

        Args:
            timesteps: Number of diffusion timesteps.
            beta_schedule: Type of noise schedule ('linear', 'cosine').
        """
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule

        # Generate beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            msg = f"Unknown beta schedule: {beta_schedule}"
            raise ValueError(msg)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register buffers for automatic GPU transfer
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        logger.debug(
            "Initialized diffusion buffers: timesteps=%d, schedule=%s",
            timesteps,
            beta_schedule,
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
        """Cosine beta schedule as proposed in Nichol & Dhariwal (2021).

        Args:
            timesteps: Number of timesteps.
            s: Small offset to prevent singularity at t=0.

        Returns:
            Beta values tensor of shape (timesteps,).
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    # ---------------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------------
    def _extract(
        self,
        tensor: Tensor,
        t: Tensor,
        x_shape: tuple[int, ...],
    ) -> Tensor:
        """Extract values from tensor at given timesteps.

        Args:
            tensor: 1D tensor of shape (T,) containing schedule values.
            t: Timestep indices of shape (B,).
            x_shape: Shape of x for broadcasting (B, C, H, W).

        Returns:
            Extracted values broadcast to (B, 1, 1, 1) or matching x_shape.
        """
        batch_size = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # ---------------------------------------------------------------------------
    # Forward Diffusion (q_sample)
    # ---------------------------------------------------------------------------
    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Forward diffusion process: q(x_t | x_0).

        Adds noise to clean input according to the schedule at timestep t.

        Args:
            x_start: Clean input tensor of shape (B, C, H, W).
            t: Timestep tensor of shape (B,).
            noise: Optional pre-sampled noise tensor. If None, sampled here.

        Returns:
            Noised tensor x_t of same shape as x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod = self._extract(
            self.sqrt_alphas_cumprod,
            t,
            x_start.shape,
        )
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    # ---------------------------------------------------------------------------
    # Reverse Diffusion (p_sample)
    # ---------------------------------------------------------------------------
    @abstractmethod
    def _predict_start_from_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Predict x_0 from x_t and predicted noise.

        Subclasses must implement this based on their denoising model.
        """
        ...

    @abstractmethod
    def _model_forward(
        self,
        x: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Forward pass of the denoising model.

        Subclasses must implement this to call their neural network.
        """
        ...

    def p_sample(
        self,
        x: Tensor,
        t: Tensor,
        t_index: int,
        **kwargs: Any,
    ) -> Tensor:
        """Reverse diffusion sampling step: p(x_{t-1} | x_t).

        Args:
            x: Noised tensor at timestep t.
            t: Timestep tensor of shape (B,).
            t_index: Integer index of timestep.
            **kwargs: Additional arguments for model forward.

        Returns:
            Denoised tensor at timestep t-1.
        """
        # Predict noise
        predicted_noise = self._model_forward(x, t, **kwargs)

        # Get schedule values
        beta = self._extract(self.betas, t, x.shape)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x.shape,
        )

        # Predict mean
        model_mean = sqrt_recip_alpha * (x - beta * predicted_noise / sqrt_one_minus_alpha_cumprod)

        if t_index == 0:
            return model_mean

        # Add noise for t > 0
        posterior_variance = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance) * noise

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
        device = self.betas.device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion loop
        for i in reversed(range(self.timesteps)):
            t = torch.full(
                (batch_size,),
                i,
                device=device,
                dtype=torch.long,
            )
            x = self.p_sample(x, t, i, **kwargs)

        return x


__all__ = ["DiffusionMixin"]
