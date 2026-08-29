"""Cold Diffusion Generator (Refactored)
======================================

Cold Diffusion generator implementing the IGenerator interface.
Based on Cold Diffusion (https://arxiv.org/abs/2208.09392).

Refactored to use DiffusionMixin for shared diffusion logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from mriforge.config.schemas.enums import SchedulerTypes
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

from .mixins import DiffusionMixin

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ColdDiffusionGeneratorConfig:
    """Configuration for Cold Diffusion Generator parameters."""

    timesteps: int = 1000
    beta_schedule: str = "linear"
    device: str | None = None
    degradation_type: str = "noise"
    cold_schedule: str = "linear"
    restoration_steps: int = 1


@register_model(name="cold_diffusion", training_mode="diffusion")
class ColdDiffusionGenerator(DiffusionMixin, IGenerator, nn.Module):
    """Cold Diffusion generator using DiffusionMixin for shared logic.

    Instead of predicting noise, the model predicts the original image x_0.
    The sampling process then involves a degradation step.

    Uses DiffusionMixin for:
    - q_sample (forward diffusion)
    - p_sample (reverse diffusion step)
    - p_sample_loop (full sampling loop)
    - _extract (tensor indexing)
    - Schedule initialization
    """

    def __init__(
        self,
        denoising_model: nn.Module | None = None,
        config: ColdDiffusionGeneratorConfig | None = None,
        timesteps: int | None = None,
        beta_schedule: str | None = None,
        device: str | None = None,
        degradation_type: str | None = None,
        cold_schedule: str | None = None,
        restoration_steps: int | None = None,
    ) -> None:
        """Initialize Cold Diffusion Generator.

        Args:
            denoising_model: Neural network that predicts x_0 from x_t.
            config: Configuration object. If provided, overrides other params.
            timesteps: Number of diffusion timesteps.
            beta_schedule: Schedule type ('linear', 'cosine').
            device: Target device.
            degradation_type: Degradation type ('noise').
            cold_schedule: Cold schedule type ('linear', 'cosine').
            restoration_steps: Number of restoration steps.
        """
        super().__init__()

        # Build config
        if config is not None:
            self.config = config
        else:
            self.config = ColdDiffusionGeneratorConfig(
                timesteps=timesteps if timesteps is not None else 1000,
                beta_schedule=beta_schedule if beta_schedule is not None else "linear",
                device=device,
                degradation_type=degradation_type or "noise",
                cold_schedule=cold_schedule or "linear",
                restoration_steps=restoration_steps or 1,
            )

        # Set instance attributes
        self.degradation_type = self.config.degradation_type
        self.restoration_steps = self.config.restoration_steps

        # Create denoising model if not provided
        if denoising_model is None:
            from .unet_generator import UNetGenerator

            denoising_model = UNetGenerator(
                in_channels=1,
                out_channels=1,
                output_activation=None,
            )
        self.denoising_model = denoising_model

        # Set device
        # `self.config` is a ColdDiffusionGeneratorConfig, NOT TrainingSettings:
        # it carries `device` and has no `run:` block. Phase 4b's
        # `device -> run.device` rename was applied here by name rather than by
        # receiver, turning a working read into a hard AttributeError.
        self._device = self.config.device or "cpu"

        # Initialize diffusion buffers via DiffusionMixin
        self._initialize_diffusion_buffers(
            timesteps=self.config.timesteps,
            beta_schedule=self.config.beta_schedule,
        )

        # Initialize cold schedule
        self._initialize_cold_schedule()

        # Move to device
        self.to(self._device)

        logger.debug(
            "Initialized ColdDiffusionGenerator: timesteps=%d, schedule=%s",
            self.timesteps,
            self.config.beta_schedule,
        )

    def _initialize_cold_schedule(self) -> None:
        """Initialize cold-specific schedule buffers."""
        if self.config.cold_schedule == SchedulerTypes.LINEAR.value:
            cold_schedule = torch.linspace(1.0, 0.0, self.timesteps)
        elif self.config.cold_schedule == SchedulerTypes.COSINE.value:
            t = torch.linspace(0, self.timesteps, self.timesteps + 1)
            cold_schedule = torch.cos(0.5 * torch.pi * t / self.timesteps)[:-1]
        else:
            msg = f"Unknown cold schedule: {self.config.cold_schedule}"
            raise ValueError(msg)

        self.register_buffer("cold_schedule", cold_schedule)

    @property
    def name(self) -> str:
        """Return the model name."""
        return "ColdDiffusionGenerator"

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.betas.device

    # -------------------------------------------------------------------------
    # DiffusionMixin Required Methods
    # -------------------------------------------------------------------------
    def _predict_start_from_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Predict x_0 from x_t and predicted noise.

        For Cold Diffusion, this is not used directly since we predict x_0.
        """
        # Cold diffusion predicts x_0 directly, not noise
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_t.shape,
        )
        return (x_t - sqrt_one_minus_alpha_cumprod * noise) / sqrt_alpha_cumprod

    def _model_forward(self, x: Tensor, t: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass of the denoising model.

        For Cold Diffusion, we predict x_0 directly (not noise).
        """
        return self.denoising_model(x)

    # -------------------------------------------------------------------------
    # Cold Diffusion Specific Methods
    # -------------------------------------------------------------------------
    def _degrade(self, x_0_pred: Tensor, t: Tensor) -> Tensor:
        """Degrade predicted x_0 to x_t using the chosen degradation."""
        if self.degradation_type == "noise":
            return self.q_sample(x_start=x_0_pred, t=t)
        msg = f"Unknown degradation type: {self.degradation_type}"
        raise ValueError(msg)

    @torch.no_grad()
    def p_sample(
        self,
        x: Tensor,
        t: Tensor,
        t_index: int,
        **kwargs: Any,
    ) -> Tensor:
        """Cold diffusion sampling step.

        Overrides DiffusionMixin.p_sample with cold diffusion logic.
        """
        cond = kwargs.get("cond")
        guidance_scale = kwargs.get("guidance_scale", 1.0)

        # Predict x_0 directly
        if guidance_scale > 1.0 and cond is not None:
            uncond_pred = self.denoising_model(x)
            cond_pred = self.denoising_model(x)
            pred_x_start = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        else:
            pred_x_start = self.denoising_model(x)

        if t_index == 0:
            return pred_x_start

        # Degrade to t-1
        prev_t = torch.full_like(t, t_index - 1)
        return self._degrade(pred_x_start, prev_t)

    # -------------------------------------------------------------------------
    # IGenerator Interface
    # -------------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for training: add noise and predict x_0.

        Args:
            x: Input clean image of shape (B, C, H, W).

        Returns:
            Predicted clean image x_0 of shape (B, C, H, W).
        """
        # Shape guards (unit-test.md Directive 1.1)
        if x.dim() != 4:
            msg = f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D"
            raise ValueError(msg)

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)

        # Forward diffusion via DiffusionMixin
        x_t = self.q_sample(x, t)

        # Predict x_0
        return self.denoising_model(x_t)

    def generate(self, z: Tensor, **kwargs: Any) -> Tensor:
        """Generate samples using cold diffusion.

        Args:
            z: Input tensor (used to determine output shape).
            **kwargs: Additional arguments (cond, guidance_scale).

        Returns:
            Generated tensor.
        """
        return self.p_sample_loop(z.shape, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        return input_shape

    def get_parameter_count(self) -> int:
        """Count total parameters in the denoising model."""
        return sum(p.numel() for p in self.denoising_model.parameters() if p.requires_grad)
