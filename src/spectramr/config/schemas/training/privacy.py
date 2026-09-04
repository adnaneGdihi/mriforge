"""Differential-privacy configuration schema (PR-7 H5).

Defines the ``privacy`` sub-block consumed by DP-SGD training strategies —
notably
:class:`spectramr.infrastructure.training.strategies.federated_dp_conformal_strategy.FederatedDPConformalULFStrategy`.

Before this schema existed, :class:`~spectramr.config.settings.TrainingSettings`
carried **no** ``privacy`` field at all, so a YAML ``privacy:`` block was
rejected outright by ``extra="forbid"`` — and the strategy's
``getattr(config, "privacy", None)`` always returned ``None``, collapsing
every DP knob to a hardcoded literal (``noise_multiplier=1.1`` etc.). A
YAML-set ``privacy.noise_multiplier`` therefore had **zero effect**, which is
pitfall #15 (an advertised-but-unwired knob) compounding pitfall #9 (a silent
fallback to a default privacy budget).

Typing the block here makes the knobs real, readable attributes and lets
Pydantic reject an out-of-range value at load time (so the audit fails the
config instead of training with a meaningless ``(ε, δ)`` budget).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PrivacyConfig(BaseModel):
    """DP-SGD privacy knobs (Abadi et al. 2016 Gaussian mechanism).

    Every field is range-validated at load time so an invalid knob fails
    fast rather than silently producing a meaningless privacy guarantee.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    noise_multiplier: float = Field(
        default=1.1,
        gt=0.0,
        description="σ of the Gaussian mechanism N(0, σ²Δ²). Must be > 0.",
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "L2 clip threshold C bounding the per-step gradient sensitivity. Must be > 0."
        ),
    )
    target_delta: float = Field(
        default=1e-5,
        gt=0.0,
        lt=1.0,
        description="Target δ of the (ε, δ)-DP guarantee. Must lie in (0, 1).",
    )
    dataset_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of training samples used to derive the DP sample rate "
            "q = batch_size / dataset_size. Consulted only when the live "
            "train-dataloader length is unavailable; None means 'derive from "
            "the dataloader (preferred) or fall back to data.dataset_size'."
        ),
    )
