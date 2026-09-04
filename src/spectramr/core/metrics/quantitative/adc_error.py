"""ADC mean-absolute-error metric (audit 2026-07 I2).

Grades an estimated apparent-diffusion-coefficient (ADC) map against a reference
in mm²/s. An ADC map is real-valued, so a complex input (an image, not an ADC
map) or a shape mismatch returns ``NaN`` (skip) rather than a silent grade —
mirroring the anti-facade guard used by ``b0_field_rmse``.

The ADC map itself is produced by the log-linear monoexponential fit
``spectramr.infrastructure.physics.signal_models.diffusion_models.fit_adc_loglinear``
from a stack of b-value images + the acquisition ``b_values`` — the estimator
paired with the ``adc_monoexp`` signal model that backs this regime.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.core.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    "adc_mae",
    aliases=["ADCMeanAbsoluteError", "adc_error"],
    direction="lower",
    workflows=frozenset({Regime.DIFFUSION_WEIGHTED}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class ADCMeanAbsoluteError:
    """Mean absolute error (mm²/s) between an estimated ADC map and a reference.

    Tagged ``mri_diffusion_weighted``: an ADC map is the parameter of the
    ``adc_monoexp`` signal model, so this is what grades that regime on its own
    terms rather than as a generic image.

    Lower is better. Both inputs must be real ADC maps of the same shape
    (``[B, 1, H, W]`` or ``[B, H, W]``). Returns ``NaN`` (skip) for a complex
    input or a shape mismatch — an ADC map is a real physical quantity.
    """

    @property
    def name(self) -> str:
        return "adc_mae"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        if prediction is None or target is None:
            return float("nan")
        if prediction.is_complex() or target.is_complex():
            logger.debug("adc_mae skipped: complex input is an image, not an ADC map.")
            return float("nan")
        if prediction.shape != target.shape:
            logger.debug(
                "adc_mae skipped: shape mismatch %s vs %s.",
                tuple(prediction.shape),
                tuple(target.shape),
            )
            return float("nan")
        return float((prediction.float() - target.float()).abs().mean())


__all__ = ["ADCMeanAbsoluteError"]
