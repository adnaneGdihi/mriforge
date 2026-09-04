"""Strategy-mechanics config for the phase-contrast and perfusion verticals.

Both blocks carry *mechanics only*. One knob, one home:

- the **physics** lives on ``data.phase_contrast`` / ``data.perfusion`` (venc,
  encoding scheme, time axis, AIF source) — those are properties of the
  acquisition, not of the trainer;
- the **weights** live on ``losses.physics.lambda_*``, resolved through the
  loss-weight SSOT (:mod:`spectramr.models.losses.weights`).

Every field here is read by its strategy in the same change that declares it
(CLAUDE.md #15) — an unread knob would be a facade in config form.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class PhaseContrastFlowConfig(StrictSchema):
    """Mechanics for :class:`PhaseContrastFlowStrategy` (regime: ``mri_flow``)."""

    validate_encode_roundtrip: bool = Field(
        default=False,
        description="On the first training step, assert that decoding the batch's "
        "4-point phases at the declared venc reproduces the reference velocity. "
        "A cheap guard that the dataset really is four-point encoded at THIS venc "
        "— a venc mismatch between config and acquisition silently rescales every "
        "velocity, and the loss would happily converge to the wrong field. Off by "
        "default (it costs one extra decode on step 0).",
    )


class PerfusionKineticConfig(StrictSchema):
    """Mechanics for :class:`PerfusionKineticMappingStrategy` (``mri_perfusion``)."""

    parameter_activation: Literal["softplus", "none"] = Field(
        default="softplus",
        description="Activation applied to the generator's (Ktrans, ve, vp) maps. "
        "``softplus`` (default) is a NUMERICAL REQUIREMENT for a network-driven "
        "fit, not a preference: the extended-Tofts kernel is "
        "exp(-(Ktrans/ve)*t), so the moment a raw output makes Ktrans negative "
        "the exponent turns positive and the curve overflows to inf — and an "
        "untrained generator emits negative values immediately. Softplus forces "
        "Ktrans >= 0, hence kep >= 0 and exp(...) in (0, 1]. It also happens to "
        "be the physical truth (negative Ktrans/ve/vp are meaningless). "
        "TRADE-OFF: it makes PerfusionPhysiologicalBoxLoss's >=0 terms redundant "
        "— they can never fire — so do not read that loss as evidence the "
        "non-negativity constraint is being learned; its ve+vp<=1 term still "
        "does real work. ``none`` passes the maps through raw and is safe ONLY "
        "when they are already constrained upstream (e.g. a direct optimisation "
        "started from a physical initialisation); on a generator it produces inf.",
    )


class MRSQuantificationConfig(StrictSchema):
    """Mechanics for :class:`MRSQuantificationStrategy` (``mri_spectroscopy``)."""

    parameter_activation: Literal["softplus", "none"] = Field(
        default="softplus",
        description="Activation applied to the generator's AMPLITUDE and "
        "LINEWIDTH maps (frequency and phase are passed through raw — both are "
        "legitimately signed). ``softplus`` (default) is a NUMERICAL REQUIREMENT, "
        "not a preference, and for exactly the reason Ktrans needs it in "
        "perfusion: the FID's damping is exp(-pi*linewidth*t), so the moment a "
        "raw output makes the linewidth negative the exponent turns positive and "
        "the FID grows without bound to inf — and an untrained generator emits "
        "negatives from step 0. It is also the physical truth (a negative "
        "concentration or linewidth is meaningless). TRADE-OFF: it makes "
        "MRSPriorKnowledgeLoss's amplitude>=0 term redundant — it can never fire "
        "— so do not read that loss as evidence non-negativity is being learned; "
        "its linewidth bounds still do real work. ``none`` passes the parameters "
        "through raw and is safe ONLY when they are constrained upstream; on a "
        "generator it produces inf.",
    )
