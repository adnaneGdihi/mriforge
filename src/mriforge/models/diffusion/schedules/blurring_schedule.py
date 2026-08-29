r"""Heat-equation (DCT) blurring forward process.

Implements the forward process of *Blurring Diffusion Models* (Hoogeboom
& Salimans, ICLR 2023). Instead of destroying image content with
isotropic Gaussian noise, the forward process attenuates high-frequency
content via the heat equation :math:`\partial_t u = \Delta u`, whose
closed-form solution on :math:`[0, L]^2` under Neumann (reflecting)
boundary conditions is diagonal in the DCT-II basis:

.. math::

    q(x_t \mid x_0) = \mathcal{N}\!\Bigl(\mathcal{C}^{-1}
        \bigl(\alpha_t(k)\,\mathcal{C}(x_0)\bigr),\; \sigma_t^2 I\Bigr),
    \qquad \alpha_t(k) = e^{-\lambda(k)\,\tau_t},

with eigenvalues :math:`\lambda_{k_x,k_y} = \pi^2 (k_x^2 + k_y^2) / L^2`
and a monotone-increasing blur schedule :math:`\{\tau_t\}`.

References:
    E. Hoogeboom and T. Salimans, "Blurring diffusion models," ICLR 2023.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mriforge.infrastructure.physics.dct_ops import dct2_neumann, idct2_neumann

__all__ = ["BlurringSchedule"]


class BlurringSchedule(nn.Module):
    r"""Frequency-dependent blur + residual-noise forward process.

    Registers the per-frequency dissipation eigenvalues ``lambda_k``, the
    monotone blur times ``tau``, and the residual-noise std ``sigma`` as
    buffers so they move with ``.to(device)`` / survive ``state_dict``
    round-trips.

    Args:
        num_timesteps: Number of discrete diffusion steps :math:`T`.
        image_size: Spatial side length :math:`L` (square images).
        sigma_min: Residual Gaussian std at :math:`t = 0`.
        sigma_max: Residual Gaussian std at :math:`t = T - 1`.
        blur_sigma_max: Controls the maximum blur time so that the
            highest-frequency mode is fully attenuated by the final step.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        image_size: int = 256,
        sigma_min: float = 0.01,
        sigma_max: float = 0.1,
        blur_sigma_max: float = 20.0,
    ) -> None:
        super().__init__()
        if num_timesteps < 1:
            raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        if sigma_min <= 0 or sigma_max <= 0:
            raise ValueError(f"sigma_min/sigma_max must be > 0, got {sigma_min}, {sigma_max}")
        if sigma_max < sigma_min:
            raise ValueError(f"sigma_max ({sigma_max}) must be >= sigma_min ({sigma_min})")

        self.num_timesteps = int(num_timesteps)
        self.image_size = int(image_size)

        # Per-frequency Laplacian eigenvalues under Neumann BCs.
        kx = torch.arange(image_size, dtype=torch.float32)
        ky = torch.arange(image_size, dtype=torch.float32)
        kxx, kyy = torch.meshgrid(kx, ky, indexing="ij")
        lambda_k = (math.pi**2) * (kxx**2 + kyy**2) / float(image_size**2)

        # Monotone-increasing blur-time schedule. tau grows as a cosine
        # ramp from ~0 to blur_sigma_max so alpha_t(k) = exp(-lambda_k * tau_t)
        # is strictly decreasing in t for every non-DC mode (k > 0).
        steps = torch.linspace(0.0, 1.0, self.num_timesteps)
        tau = blur_sigma_max * (1.0 - torch.cos(steps * math.pi / 2.0))

        sigma = torch.linspace(sigma_min, sigma_max, self.num_timesteps)

        self.register_buffer("lambda_k", lambda_k)
        self.register_buffer("tau", tau)
        self.register_buffer("sigma", sigma)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""Per-frequency dissipation factor :math:`\alpha_t(k)`.

        Args:
            t: Long tensor of timestep indices, shape ``[B]``.

        Returns:
            Tensor of shape ``[B, 1, H, W]`` with values in ``(0, 1]``.
        """
        tau_t = self.tau.index_select(0, t).view(-1, 1, 1, 1)
        return torch.exp(-self.lambda_k.unsqueeze(0).unsqueeze(0) * tau_t)

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Residual-noise std broadcast to ``[B, 1, 1, 1]``."""
        return self.sigma.index_select(0, t).view(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        r"""Sample :math:`x_t \sim q(x_t \mid x_0)`.

        Blurs ``x0`` in the DCT domain by :math:`\alpha_t(k)` then adds
        the residual Gaussian noise :math:`\sigma_t\,\epsilon`.

        Args:
            x0: Clean image batch ``[B, C, H, W]``.
            t: Long timestep indices ``[B]``.
            noise: Gaussian noise broadcastable to ``x0``.

        Returns:
            Blurred + noised image ``x_t`` of the same shape as ``x0``.
        """
        alpha_t = self.alpha(t)  # [B, 1, H, W]
        blurred = idct2_neumann(alpha_t * dct2_neumann(x0))
        return blurred + self.sigma_t(t) * noise

    def v_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Velocity (v-parameterisation) target.

        :math:`v_t = \alpha_t\,\epsilon - \sigma_t\,x_0`. The
        frequency-dependent dissipation :math:`\alpha_t(k)` weights the
        noise term in the DCT domain (the same operator used to blur the
        signal), while the residual-noise std :math:`\sigma_t` weights
        the clean-image term. This v-parameterisation is numerically
        stable across the full SNR range, unlike the
        :math:`\epsilon`-parameterisation which loses signal as
        :math:`t \to 0` in the blurring regime.

        Args:
            x0: Clean image batch ``[B, C, H, W]``.
            noise: Gaussian noise of the same shape as ``x0``.
            t: Long timestep indices ``[B]``.

        Returns:
            v-target tensor of the same shape as ``x0``.
        """
        alpha_t = self.alpha(t)  # [B, 1, H, W]
        alpha_noise = idct2_neumann(alpha_t * dct2_neumann(noise))
        return alpha_noise - self.sigma_t(t) * x0
