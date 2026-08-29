r"""Schema for B-1 Equivariance-Defect Conformal Prediction (Idea 1).

Defines the ``training.equivariance_conformal`` sub-block and the top-level
``TrainingConfigEquivarianceConformal`` that mounts it. The strategy
(:class:`~mriforge.infrastructure.training.strategies.equivariance_conformal_strategy.EquivarianceConformalCalibrationStrategy`)
is a degenerate, no-gradient calibration pass: it loads a frozen reconstructor,
computes the equivariance-defect score over ``num_rotations_K`` rotations per
calibration sample, and emits the split-conformal quantile :math:`q_\alpha`.

Mounting (lead wires this — not edited here): add to
``mriforge/config/schemas/training/__init__.py``::

    from .equivariance_conformal import TrainingConfigEquivarianceConformal
    _MODE_DISPATCH["equivariance_conformal"] = "TrainingConfigEquivarianceConformal"

and re-export the name in ``__all__``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import BaseTrainingConfigSchema


class EquivarianceConformalConfig(BaseModel):
    """Hyperparameters for the equivariance-defect conformal calibration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Target miscoverage rate; conformal coverage is >= 1 - alpha.",
    )
    num_rotations_K: int = Field(
        default=8,
        ge=1,
        description="Monte-Carlo group elements per equivariance-defect score.",
    )
    group: Literal["SO2", "SE2"] = Field(
        default="SO2",
        description="Rotation group: SO2 (planar) or SE2 (rotation + integer shift).",
    )
    max_shift_px: int = Field(
        default=4,
        ge=0,
        description="Max integer pixel shift for the SE2 translation part.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Base seed for the per-sample (exchangeable) rotation draws.",
    )
    exchangeability_test: Literal["ks_test", "none"] = Field(
        default="ks_test",
        description=(
            "Post-hoc check that calibration and a held-out split share a "
            "score distribution (a borderline-exchangeability safety net)."
        ),
    )
    exchangeability_ks_pvalue_threshold: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Minimum KS p-value below which exchangeability is flagged.",
    )
    pretrained_reconstructor_checkpoint: str = Field(
        ...,
        description="Path to the frozen reconstructor checkpoint G_phi.",
    )
    output_artefact: str | None = Field(
        default=None,
        description="Optional path to write the conformal-quantile JSON report.",
    )


class TrainingConfigEquivarianceConformal(BaseTrainingConfigSchema):
    """Top-level training config when ``training_mode == 'equivariance_conformal'``."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    equivariance_conformal: EquivarianceConformalConfig


__all__ = [
    "EquivarianceConformalConfig",
    "TrainingConfigEquivarianceConformal",
]
