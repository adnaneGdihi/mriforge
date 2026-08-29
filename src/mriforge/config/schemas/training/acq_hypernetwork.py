r"""Schema for LCAH: Lipschitz-Certified Acquisition Hypernetwork (M3).

Defines the ``training.acq_hypernetwork`` sub-block and the top-level
``TrainingConfigAcqHypernetwork`` that mounts it. The strategy
(:class:`~mriforge.infrastructure.training.strategies.hypernetwork_strategy.AcquisitionHypernetworkStrategy`)
trains :class:`~mriforge.models.encoders.lcah_encoder.LCAHEncoder` -- a
spectral-normalised hypernetwork emitting FiLM modulation for a target network,
conditioned on the continuous acquisition vector
:math:`\boldsymbol\varphi=(\mathrm{TE},\mathrm{TR},\mathrm{TI},\alpha,B_0)`.

The discriminating claim is the *certificate*, not the conditioning: with both
networks spectral-normalised,

.. math::

   \bigl\|f_{h_\psi(\boldsymbol\varphi)}(\mathbf x)
       - f_{h_\psi(\boldsymbol\varphi^\ast)}(\mathbf x)\bigr\|
   \;\le\; L_w L_h\,\lVert\boldsymbol\varphi-\boldsymbol\varphi^\ast\rVert ,

so the certified extrapolation radius is reportable at inference at no training
cost. ``spectral_norm: false`` is permitted (it is a legitimate ablation) but the
Tier-1 ``spectral_norm_enabled`` check warns, because the certificate is void
without it.

Mounting (mirrors M1 phys_residual_conformal): the typed wrapper is registered in
``mriforge/config/schemas/training/__init__.py``::

    from .acq_hypernetwork import TrainingConfigAcqHypernetwork
    _MODE_DISPATCH["acq_hypernetwork"] = "TrainingConfigAcqHypernetwork"
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .base import BaseTrainingConfigSchema

#: Canonical acquisition-vector layout, matching the SSOT
#: ``ConditioningContext.acquisition`` ordering. Kept here so the schema, the
#: strategy and the ``acq_vector_present`` audit check agree on one spelling.
ACQUISITION_VECTOR_FIELDS: tuple[str, ...] = ("te", "tr", "ti", "fa", "b0")


class AcqHypernetworkConfig(BaseModel):
    """Hyperparameters for the LCAH acquisition hypernetwork."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hyper_layers: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Depth of the hypernetwork h_psi mapping phi -> FiLM (gamma, beta).",
    )
    hyper_hidden: int = Field(default=32, ge=1, description="Hypernetwork hidden width.")
    target_backbone: str = Field(
        default="lcah_encoder",
        description="Registered model name for the FiLM-modulated target f_theta.",
    )
    spectral_norm: bool = Field(
        default=True,
        description=(
            "Spectral-normalise both networks. Required for the certificate to "
            "be valid; false is an ablation only (Tier-1 warns)."
        ),
    )
    lipschitz_target: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Optional target for the product L_w*L_h. When set, the strategy "
            "penalises the empirical product above this value."
        ),
    )
    lipschitz_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight on the Lipschitz-budget penalty (0 disables it).",
    )
    lambda_l1: float = Field(
        default=1.0, ge=0.0, description="Weight on the L1 reconstruction term."
    )
    acquisition_key: str = Field(
        default="acquisition",
        description=(
            "Batch key carrying the [B, 5] acquisition vector. The pipeline must "
            "actually emit it -- via data.acquisition_metadata.enabled (per-sample) "
            "or data.multi_contrast.acquisition_params (fixed per contrast). "
            "Tier-1 acq_vector_present enforces one of the two."
        ),
    )
    report_certified_radius: bool = Field(
        default=True,
        description="Emit the certified extrapolation radius during validation.",
    )


class TrainingConfigAcqHypernetwork(BaseTrainingConfigSchema):
    """Top-level training config wrapper that mounts the LCAH sub-block."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    acq_hypernetwork: AcqHypernetworkConfig


__all__ = [
    "ACQUISITION_VECTOR_FIELDS",
    "AcqHypernetworkConfig",
    "TrainingConfigAcqHypernetwork",
]
