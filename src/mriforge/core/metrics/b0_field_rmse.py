"""Absolute B0-field RMSE (Hz) — the headline metric for exp_vf_29.

Grades a model's estimated off-resonance field against an independently known
B0 reference in Hz. The point of the metric is *absolute* accuracy, so it must
refuse to grade anything that is not an Hz field: the runtime half of the
two-lock parametrization guard (the static half is the Tier-1 check that the
model declares ``output_field_units == "Hz"``).

A complex tensor is an *image*, not a real field — grading it as a B0 map is the
``bssfp_peak_finder`` facade (returns a corrected complex image, not a field;
pitfall #16). Such an input, or a shape mismatch, returns ``NaN`` (skip), never a
silent grade.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from mriforge.config.schemas.enums import Regime, Task
from mriforge.core.metrics._guards import field_comparability
from mriforge.core.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    "b0_field_rmse",
    aliases=["b0_rmse_hz"],
    direction="lower",
    workflows=frozenset({Regime.QUANTITATIVE}),
    tasks=frozenset({Task.PARAMETER_MAPPING, Task.FIELD_TRANSLATION}),
)
class B0FieldRMSE:
    """RMS error (Hz) between an estimated B0 field and a real B0 reference.

    Tagged ``mri_quantitative``: it grades a field map in Hz, a physical
    quantity, and it is what makes ``MultiEchoB0FitStrategy``'s claim checkable.

    Lower is better. Returns ``NaN`` (skip) when the estimate is not an Hz field
    comparable to the reference — a complex image, a shape mismatch, or a missing
    tensor — so the metric never produces a number for an incomparable quantity.
    """

    @property
    def name(self) -> str:
        return "b0_field_rmse"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        # A real B0 field is real-valued; a complex tensor is an image, not a
        # field (the facade the guard exists to block).
        est_kind = "image" if (prediction is not None and prediction.is_complex()) else "field"
        verdict = field_comparability(
            prediction,
            target,
            estimate_kind=est_kind,
            reference_kind="field",
            estimate_units="Hz",
            reference_units="Hz",
        )
        if not verdict.ok:
            logger.debug("b0_field_rmse skipped (not an Hz field): %s", verdict.reason)
            return float("nan")
        assert prediction is not None and target is not None  # guaranteed by verdict.ok
        diff = prediction.float() - target.float()
        return float(torch.sqrt(torch.mean(diff * diff)))
