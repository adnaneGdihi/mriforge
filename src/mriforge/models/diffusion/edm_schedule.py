r"""EDM noise schedule + preconditioning (paradigm-expansion PR-10 / M2).

Implements Karras et al. *NeurIPS* 2022 [Elucidated Diffusion Models]:

- :class:`EDMNoiseSchedule` — logarithmic σ-spacing per the paper:

  .. math::

      \sigma_i = \bigl(\sigma_{max}^{1/\rho}
                + i/(N-1)\,(\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho})\bigr)^{\rho},
      \quad i = 0, \ldots, N-1.

- :func:`edm_preconditioning_coefficients` — the four scalar
  multipliers :math:`(c_{skip}, c_{out}, c_{in}, c_{noise})` Karras
  derives so that the network sees a unit-variance input and emits
  the residual:

  .. math::

      c_{skip} &= \frac{\sigma_{data}^2}{\sigma^2 + \sigma_{data}^2}, \\
      c_{out}  &= \sigma \cdot \sigma_{data} / \sqrt{\sigma_{data}^2 + \sigma^2}, \\
      c_{in}   &= 1 / \sqrt{\sigma^2 + \sigma_{data}^2}, \\
      c_{noise} &= \tfrac{1}{4}\log \sigma.

- :func:`edm_loss_weight` — :math:`\lambda(\sigma) =
  (\sigma^2 + \sigma_{data}^2) / (\sigma\,\sigma_{data})^2` gives the
  per-sample denoising-loss weight that makes training stable across
  the full σ range.

Reference
---------
Karras T., Aittala M., Aila T., Laine S. (2022). *Elucidating the
Design Space of Diffusion-Based Generative Models*. NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EDMSchedulePoint:
    """A single (σ, t) entry on the EDM schedule.

    Attributes:
        sigma: Noise level σ.
        t: Continuous-time index in ``[0, 1]`` (informational —
           the denoiser receives σ, not t).
    """

    sigma: float
    t: float


class EDMNoiseSchedule:
    r"""Logarithmic σ-spacing per Karras et al. 2022.

    Args:
        sigma_min: Smallest noise level (default 0.002).
        sigma_max: Largest noise level (default 80.0).
        rho: Schedule exponent (default 7.0 — Karras' choice).
        num_steps: Number of σ values produced by :py:meth:`sigmas`.

    Raises:
        ValueError: invalid arguments.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_steps: int = 18,
    ) -> None:
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be > 0; got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must exceed sigma_min; got sigma_max={sigma_max}, sigma_min={sigma_min}"
            )
        if rho <= 0:
            raise ValueError(f"rho must be > 0; got {rho}")
        if num_steps < 2:
            raise ValueError(f"num_steps must be ≥ 2; got {num_steps}")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.num_steps = int(num_steps)

    def sigmas(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the schedule's ``num_steps`` σ values, descending."""
        i = torch.arange(self.num_steps, device=device, dtype=dtype)
        u = i / (self.num_steps - 1)
        sigma_min_r = self.sigma_min ** (1.0 / self.rho)
        sigma_max_r = self.sigma_max ** (1.0 / self.rho)
        # Note: schedule is descending — sigmas() goes from σ_max down to σ_min.
        return (sigma_max_r + u * (sigma_min_r - sigma_max_r)) ** self.rho

    def sample_sigma(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""Karras log-uniform σ sampler (training-time).

        Samples log σ ~ N(P_mean, P_std²); the paper's defaults are
        ``P_mean = -1.2`` and ``P_std = 1.2``. The resulting σ values
        are clamped to ``[sigma_min, sigma_max]`` so they never escape
        the schedule.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1; got {batch_size}")
        # Karras' canonical training-time σ distribution:
        #   ln σ ~ N(P_mean, P_std²),   P_mean = -1.2, P_std = 1.2.
        log_sigma = torch.randn(batch_size, device=device, generator=generator) * 1.2 - 1.2
        sigma = log_sigma.exp()
        return sigma.clamp(self.sigma_min, self.sigma_max)


def edm_preconditioning_coefficients(
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Karras :math:`(c_{skip}, c_{out}, c_{in}, c_{noise})` coefficients.

    Args:
        sigma: ``[B]`` per-sample noise level.
        sigma_data: Standard deviation of the clean data distribution.

    Returns:
        ``(c_skip, c_out, c_in, c_noise)`` — all ``[B]`` tensors.

    Raises:
        ValueError: ``sigma_data <= 0`` or any ``sigma <= 0``.
    """
    if sigma_data <= 0:
        raise ValueError(f"sigma_data must be > 0; got {sigma_data}")
    if (sigma <= 0).any():
        raise ValueError("All sigma values must be > 0")
    sigma_sq = sigma * sigma
    data_sq = sigma_data * sigma_data
    denom = (sigma_sq + data_sq).clamp_min(1e-12)
    c_skip = data_sq / denom
    c_out = sigma * sigma_data / torch.sqrt(denom)
    c_in = 1.0 / torch.sqrt(denom)
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def edm_loss_weight(
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    r"""Karras per-sample loss weight :math:`\lambda(\sigma)`.

    .. math::

        \lambda(\sigma) = (\sigma^2 + \sigma_{data}^2) / (\sigma \sigma_{data})^2.

    This makes the per-σ training loss have approximately unit
    variance, so the optimiser sees comparable signal at every noise
    level.
    """
    if sigma_data <= 0:
        raise ValueError(f"sigma_data must be > 0; got {sigma_data}")
    if (sigma <= 0).any():
        raise ValueError("All sigma values must be > 0")
    return (sigma * sigma + sigma_data * sigma_data) / (sigma * sigma_data) ** 2


__all__ = [
    "EDMNoiseSchedule",
    "EDMSchedulePoint",
    "edm_loss_weight",
    "edm_preconditioning_coefficients",
]
