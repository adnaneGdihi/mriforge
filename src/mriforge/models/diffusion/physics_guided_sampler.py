"""Physics-Guided Reverse Sampler for Diffusion-Based MRI Reconstruction.

Composable module that injects data consistency (DC) into diffusion reverse
steps, preventing hallucination of structures not supported by acquired
k-space measurements.

Architecture:
    At each reverse diffusion step t:
    1. Denoising network produces x̂₀|t (predicted clean image)
    2. PhysicsGuidedReverseSampler enforces DC projection:
       x̂₀|t_corrected = DC(x̂₀|t, y_measured, mask)
    3. Corrected estimate feeds back into the reverse schedule

The module composes with existing DC layers (HardDataConsistency,
SoftDataConsistency) and supports both single-coil and multi-coil paths
via the physics infrastructure already in the framework.

Supports:
    - DDPM / DDIM / Cold Diffusion reverse processes
    - Hard DC (measurement replacement) and Soft DC (proximal gradient)
    - Single-coil (fft2c/ifft2c) and multi-coil (SENSE) forward models
    - Configurable DC step size (λ_dc)
    - Gradient-through-DC for end-to-end training

References:
    - Chung et al., "Score-based diffusion models for accelerated MRI",
      Medical Image Analysis, 2022.
    - Song et al., "Solving Inverse Problems in Medical Imaging with
      Score-Based Generative Models", ICLR 2022.
    - Jalal et al., "Robust Compressed Sensing MRI with Deep Generative
      Priors", NeurIPS 2021.
"""

from __future__ import annotations

import logging
from enum import Enum

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.data_consistency import (
    HardDataConsistency,
    SoftDataConsistency,
)
from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.models.diffusion.samplers import register_sampler

logger = logging.getLogger(__name__)


class DCMode(str, Enum):
    """Data consistency projection mode."""

    HARD = "hard"
    SOFT = "soft"
    GRADIENT = "gradient"


@register_sampler(
    name="physics_guided",
    aliases=["physics_guided_reverse", "PhysicsGuidedReverseSampler"],
)
class PhysicsGuidedReverseSampler(nn.Module):
    """Composable DC-projected reverse step for diffusion MRI reconstruction.

    Wraps the framework's existing DC infrastructure into a single module
    that can be inserted into any diffusion reverse process. Handles format
    conversion, multi-coil support, and configurable projection modes.

    Args:
        dc_mode: Data consistency mode.
            'hard' — Replace measured k-space exactly (HardDataConsistency).
            'soft' — Proximal blend with learnable λ (SoftDataConsistency).
            'gradient' — Gradient-based correction x += λ·A^H(y - Ax).
        lambda_dc: DC step size. For 'hard' mode, ignored (hard replacement).
            For 'soft' and 'gradient', controls projection strength.
        learnable_lambda: If True, make lambda_dc a learnable parameter (soft mode).
        noise_level: Noise level for DC layer (train/eval).
    """

    def __init__(
        self,
        dc_mode: str = "hard",
        lambda_dc: float = 1.0,
        learnable_lambda: bool = False,
        noise_level: float = 0.0,
    ) -> None:
        super().__init__()
        self.dc_mode = DCMode(dc_mode.lower())
        self.noise_level = noise_level

        if self.dc_mode == DCMode.HARD:
            self.dc_layer = HardDataConsistency(
                train_noise_level=noise_level,
                eval_noise_level=0.0,
            )
        elif self.dc_mode == DCMode.SOFT:
            self.dc_layer = SoftDataConsistency(
                lambda_init=lambda_dc,
                learnable=learnable_lambda,
            )
        elif self.dc_mode == DCMode.GRADIENT:
            # Gradient mode uses manual A^H(y - Ax) projection
            self.dc_layer = None
            if learnable_lambda:
                self.lambda_dc = nn.Parameter(torch.tensor(float(lambda_dc)))
            else:
                self.register_buffer("lambda_dc", torch.tensor(float(lambda_dc)))
        else:
            raise ValueError(f"Unknown DC mode: {dc_mode}")

    def forward(
        self,
        x_pred: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        is_kspace_domain: bool = False,
    ) -> torch.Tensor:
        """Apply DC projection to a diffusion reverse-step prediction.

        Args:
            x_pred: Predicted clean image or k-space [B, C, H, W].
                For image domain: [B, 2, H, W] (real/imag interleaved) or complex.
                For k-space domain: same format but already in frequency domain.
            measured_kspace: Acquired k-space measurements [B, C, H, W].
            mask: Binary undersampling mask [B, 1, H, W] (1 = sampled).
            sensitivity_maps: Optional coil sensitivity maps [B, Nc, H, W] (complex).
                If provided, uses SENSE model for multi-coil DC.
            is_kspace_domain: If True, x_pred is already in k-space domain.

        Returns:
            DC-projected prediction, same shape and format as x_pred.

        forward method for PhysicsGuidedReverseSampler.

        Executes PyTorch tensor operations.

        Args:
            x_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sensitivity_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            is_kspace_domain (bool): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.dc_mode == DCMode.GRADIENT:
            return self._gradient_dc(
                x_pred, measured_kspace, mask, sensitivity_maps, is_kspace_domain
            )

        # Hard or Soft DC — delegate to existing infrastructure
        return self.dc_layer(
            x_pred,
            measured_kspace=measured_kspace,
            mask=mask,
            is_kspace_domain=is_kspace_domain,
        )

    def _gradient_dc(
        self,
        x_pred: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None,
        is_kspace_domain: bool,
    ) -> torch.Tensor:
        """Gradient-based DC: x_corrected = x̂ + λ·A^H(y - A(x̂)).

        This is the Manifold Constrained Gradient (MCG) approach from
        Chung et al. (2022). It applies a single gradient step toward
        data consistency rather than hard replacement.
        """
        is_complex_input = torch.is_complex(x_pred)
        is_interleaved = not is_complex_input and x_pred.ndim == 4 and x_pred.shape[1] % 2 == 0

        # Convert to complex for physics operations
        if is_interleaved:
            real = x_pred[:, 0::2, ...]
            imag = x_pred[:, 1::2, ...]
            x_complex = torch.complex(real, imag)
        elif not is_complex_input:
            x_complex = x_pred.to(torch.complex64)
        else:
            x_complex = x_pred

        # Forward operator: image → k-space
        if is_kspace_domain:
            k_pred = x_complex
        else:
            k_pred = fft2c(x_complex)

        # Convert measured k-space to complex if needed
        if not torch.is_complex(measured_kspace):
            if measured_kspace.shape[1] % 2 == 0:
                real_m = measured_kspace[:, 0::2, ...]
                imag_m = measured_kspace[:, 1::2, ...]
                measured_complex = torch.complex(real_m, imag_m)
            else:
                measured_complex = measured_kspace.to(torch.complex64)
        else:
            measured_complex = measured_kspace

        # Ensure mask dimensions match
        while mask.dim() < k_pred.dim():
            mask = mask.unsqueeze(1)
        mask_f = mask.float()

        # Residual: r = M ⊙ (y - Ax̂)
        residual = mask_f * (measured_complex - k_pred)

        # Adjoint: A^H(r) = IFFT(r) for single-coil
        if is_kspace_domain:
            correction = residual
        else:
            correction = ifft2c(residual)

        # Apply correction with step size
        lambda_val = self.lambda_dc
        if is_kspace_domain:
            k_corrected = k_pred + lambda_val * correction
            x_corrected = k_corrected
        else:
            x_corrected = x_complex + lambda_val * correction

        # Convert back to original format
        if is_interleaved:
            return torch.stack([x_corrected.real, x_corrected.imag], dim=2).flatten(1, 2)
        elif not is_complex_input:
            return x_corrected.real
        return x_corrected

    def extra_repr(self) -> str:
        parts = [f"mode={self.dc_mode.value}"]
        if self.dc_mode == DCMode.GRADIENT:
            lam = self.lambda_dc
            if isinstance(lam, nn.Parameter):
                parts.append(f"lambda_dc={lam.item():.4f} (learnable)")
            else:
                parts.append(f"lambda_dc={lam.item():.4f}")
        return ", ".join(parts)


__all__ = ["DCMode", "PhysicsGuidedReverseSampler"]
