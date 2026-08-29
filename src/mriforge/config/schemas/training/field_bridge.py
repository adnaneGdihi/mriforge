"""Config for FieldBridgeStrategy (MICCAI MRIxFields2026, B-3.3).

A field-conditioned **Schrödinger bridge** for any-to-any cross-field translation:
one velocity field, conditioned on the field coordinate along the path, transports
between any two field marginals. Distinct from the deterministic field-flow (B-3.1)
by its **stochastic** entropic interpolant (a Brownian-bridge noise term during
training) and a stochastic reverse sampler — ``sigma_bridge=0`` recovers the
deterministic flow, so the knob directly tests the value of the stochastic bridge.
Read via ``config.training.field_bridge``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class FieldBridgeConfig(StrictSchema):
    """Field-conditioned Schrödinger-bridge knobs (B-3.3)."""

    sigma_bridge: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Entropic noise scale of the Brownian-bridge interpolant + stochastic "
            "sampler (0 = deterministic flow; >0 = stochastic Schrödinger bridge)."
        ),
    )
    n_steps: int = Field(
        default=20,
        ge=1,
        description="Reverse-sampler step count along the field axis.",
    )
    velocity_norm: Literal["l1", "l2"] = Field(
        default="l2",
        description="Norm of the velocity-matching loss (reuses FieldFlowVelocityLoss).",
    )
    lambda_endpoint_l1: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of an endpoint L1 anchor |x_hat_t - x_t| on the one-step endpoint estimate "
            "x_hat_t = x_path + v*(tau_t - tau_now). Trains the denoiser property the stochastic "
            "sampler relies on and grounds the transport in the true target — the "
            "anti-degeneracy guard (#20) preventing a noise-dominated bridge output. Applied "
            "identically in both arms so the one-knob stays sigma_bridge. 0 = pure velocity "
            "matching (pre-guard behaviour)."
        ),
    )
