r"""Classifier-Free Guidance (CFG) — stand-alone sampler module.

CFG (Ho & Salimans 2021) merges a conditional and an unconditional
diffusion prediction without ever training a separate classifier:

.. math::
    \tilde{\boldsymbol{\epsilon}}(\mathbf{x}_t, t, \mathbf{c})
    = (1 + w)\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})
    - w\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset).

The framework already implements CFG in two specific places:

* `DDIMSampler` ([src/models/diffusion/ddim_sampler.py](ddim_sampler.py))
  applies CFG when ``guidance_scale != 1.0`` and ``cond is not None``.
* `MultiAxisCFGMixin` ([src/models/blocks/multi_axis_cfg.py](../blocks/multi_axis_cfg.py))
  extends it to per-axis guidance for multi-conditional models.

This module provides the same formula as a *stand-alone* composable
sampler matching the API of `ClassifierGuidanceSampler`,
`PhysicsGuidedReverseSampler`, `MCGSampler`, and `DPSMRISampler`. Use
it in any custom reverse loop that does not go through DDIM.

Reference:
    [59] J. Ho, T. Salimans, "Classifier-free diffusion guidance,"
        *NeurIPS Workshop*, 2021.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class ClassifierFreeGuidanceSampler(nn.Module):
    r"""CFG corrector for a noise/score prediction.

    The denoiser is invoked twice — once with the requested condition
    :math:`\mathbf{c}` and once with the model's null token
    :math:`\emptyset` (empty / zero / drop-flag — strategy-defined).

    Args:
        denoiser: Callable ``denoiser(x_t, t, cond) -> noise_pred``.
            Must accept ``cond=None`` to return the unconditional
            prediction (or wrap a model whose forward already
            interprets ``None`` as null).
        guidance_scale: ``w`` in the formula. ``0`` reduces to the
            unconditional prediction; ``1`` is identity-conditional;
            typical text-to-image numbers are 5–10.
        unconditional_token: Optional explicit "null" token to pass to
            the denoiser instead of ``None``. Useful for embedding-based
            conditioners that need a learned null vector.
    """

    def __init__(
        self,
        denoiser: Callable[..., torch.Tensor],
        guidance_scale: float = 1.0,
        unconditional_token: object | None = None,
    ) -> None:
        super().__init__()
        if not callable(denoiser):
            raise TypeError("denoiser must be callable.")
        if guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0, got {guidance_scale}")
        self.denoiser = denoiser
        self.guidance_scale = guidance_scale
        self.unconditional_token = unconditional_token

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: object | None,
    ) -> torch.Tensor:
        r"""Return the guidance-corrected prediction.

        Convention (matches `DDIMSampler.guidance_scale` in this codebase):

        * ``cond is None`` → unconditional pass only.
        * ``guidance_scale == 1`` → purely conditional pass.
        * ``guidance_scale == 0`` → unconditional pass (purely
          marginal — no conditioning influence).
        * ``guidance_scale > 1`` → standard CFG extrapolation past the
          conditional.
        """
        if cond is None:
            return self.denoiser(x_t, t, self.unconditional_token)
        if self.guidance_scale == 1.0:
            return self.denoiser(x_t, t, cond)
        if self.guidance_scale == 0.0:
            return self.denoiser(x_t, t, self.unconditional_token)

        eps_cond = self.denoiser(x_t, t, cond)
        eps_uncond = self.denoiser(x_t, t, self.unconditional_token)
        if eps_cond.shape != eps_uncond.shape:
            raise ValueError(
                f"conditional shape {tuple(eps_cond.shape)} != "
                f"unconditional shape {tuple(eps_uncond.shape)}"
            )
        return eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
