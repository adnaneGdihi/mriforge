r"""DWI monoexponential ADC (diffusion) consistency loss (audit 2026-07 I2).

Diffusion-weighted MRI follows the monoexponential model

.. math::

    S(b) = S_0 \, e^{-b \cdot \mathrm{ADC}},

where ``b`` is the diffusion weighting (s/mm²) and ``ADC`` the apparent
diffusion coefficient (mm²/s). Given a stack of predicted b-value images and the
per-acquisition ``b_values``, this loss fits ``(S_0, ADC)`` per pixel by
log-linear least squares and penalises the deviation of the images from the
fitted monoexponential signal — a physics prior that the DWI output obeys the
diffusion model. Optionally it also penalises the fitted ADC map against a
reference ADC.

This is the **real consumer** of the ``b_values`` batch key produced by
:class:`spectramr.data.transforms.dwi_metadata.LoadDWIMetadata`, which previously
had no consumer (an inert facade — audit finding F3).

The physics itself lives in
:mod:`spectramr.infrastructure.physics.signal_models.diffusion_models` (canonical
homes, CLAUDE.md #6) and is registered there as the ``adc_monoexp`` signal model
— the forward map backing ``mri_diffusion_weighted``. ``fit_adc_loglinear`` is
re-exported here for the callers that already import it from this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.signal_models.diffusion_models import (
    fit_adc_loglinear,
)
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="dwi_adc_monoexp",
    aliases=["DWIADCMonoexpLoss", "adc_monoexp"],
    domain="image",
    workflows=frozenset({Regime.DIFFUSION_WEIGHTED}),
    tasks=frozenset({Task.RECONSTRUCTION, Task.PARAMETER_MAPPING}),
)
class DWIADCMonoexpLoss(nn.Module):
    r"""Monoexponential diffusion (ADC) consistency loss.

    Args:
        weight: overall ``λ ≥ 0`` multiplier.
        lambda_adc: weight of the optional ADC-vs-reference term (needs
            ``adc_reference`` in ``forward``). ``0`` disables it.
        eps: numerical floor on the signal / variance.

    ``forward(pred, b_values, adc_reference=None)``:
        - ``pred``: ``[B, n_b, H, W]`` predicted b-value images.
        - ``b_values``: ``[n_b]`` diffusion weightings (s/mm²).
        - ``adc_reference``: optional ``[B, 1, H, W]`` reference ADC (mm²/s).
    """

    def __init__(self, weight: float = 1.0, lambda_adc: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be ≥ 0; got {weight}")
        if lambda_adc < 0:
            raise ValueError(f"lambda_adc must be ≥ 0; got {lambda_adc}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0; got {eps}")
        self.weight = float(weight)
        self.lambda_adc = float(lambda_adc)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        b_values: torch.Tensor,
        adc_reference: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        adc, log_s0 = fit_adc_loglinear(pred, b_values, eps=self.eps)
        b = b_values.reshape(-1).to(pred.dtype).to(pred.device)
        b_col = b.reshape(1, -1, 1, 1)
        # Reconstructed log-signal from the fit, and the log-space residual: the
        # network output should obey ln S = ln S0 - b·ADC (monoexp structure).
        y = torch.log(pred.clamp_min(self.eps))
        y_fit = log_s0 - adc * b_col
        consistency = ((y - y_fit) ** 2).mean()

        total = consistency
        if self.lambda_adc > 0.0:
            if adc_reference is None:
                raise ValueError(
                    "DWIADCMonoexpLoss.lambda_adc>0 requires an adc_reference "
                    "[B, 1, H, W] map (mm²/s)."
                )
            total = total + self.lambda_adc * ((adc - adc_reference) ** 2).mean()
        return self.weight * total


__all__ = ["DWIADCMonoexpLoss", "fit_adc_loglinear"]
