"""SPECTRA test-time adaptation (TTA) configuration schema (plan §D).

The single training-flavoured SPECTRA paradigm: freeze the pre-trained backbone
and adapt only normalization / adapter parameters per subject, driven by a
self-supervised k-space-consistency objective. Keeping the adaptation surface
small is what bounds the train/infer memory multiplier (gradients + Adam moments
shrink from ``|W|`` to ``|W_adapter| ≪ |W|``), so it can run on the inference
datapath. The ``tta_adapter_only`` Tier-1 audit check enforces the freeze; the
:class:`SpectraTestTimeAdaptationStrategy` enforces it at runtime.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .base import BaseTrainingConfigSchema


class SpectraTTAConfig(BaseModel):
    """SPECTRA test-time adaptation hyperparameters."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    lambda_consistency: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Weight on the self-supervised k-space data-consistency objective "
            "(the only TTA loss; must be > 0 or there is nothing to adapt)."
        ),
    )
    num_inner_steps: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Per-subject adaptation steps before the (adapted) forward pass.",
    )
    learning_rate: float = Field(
        default=1.0e-4,
        gt=0.0,
        description="Inner-loop learning rate for the adapter/norm parameters.",
    )
    freeze_backbone: bool = Field(
        default=True,
        description=(
            "Freeze every non-norm/adapter parameter. Keep True: this is the "
            "bounded-memory guarantee. False is debug-only."
        ),
    )


class TrainingConfigSpectraTTA(BaseTrainingConfigSchema):
    """Configuration for the SPECTRA test-time-adaptation paradigm."""

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    training_mode: Literal["spectra_tta"] = Field(default="spectra_tta")

    spectra_tta: SpectraTTAConfig = Field(
        default_factory=SpectraTTAConfig,
        description="SPECTRA test-time-adaptation specific configuration.",
    )


__all__ = ["SpectraTTAConfig", "TrainingConfigSpectraTTA"]
