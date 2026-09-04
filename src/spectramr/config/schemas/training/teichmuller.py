"""Teichmüller-scheduled cold-diffusion hyperparameters (idea 6).

Typed sub-block mounted at ``training.teichmuller`` so the schedule-head
and Beltrami-endpoint knobs are validated by Pydantic instead of being
read with bare ``getattr`` fallbacks inside the strategy (which would
silently accept typos and out-of-range values — CLAUDE.md pitfalls
#3 / #15). Read by
:class:`~spectramr.infrastructure.training.strategies.teichmuller_cold_diffusion_strategy.TeichmullerColdDiffusionStrategy`
at construction time, which RAISES when the block is absent (no silent
fallback, pitfall #9).

Plan: TODO/audit_plan_novel.md §6 (Teichmüller-scheduled cold diffusion
over space-filling curves).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TeichmullerTrainingConfigSchema(BaseModel):
    """Hyperparameters for the Teichmüller cold-diffusion strategy.

    The schedule head walks the closed-form Teichmüller geodesic between
    two Beltrami endpoints ``μ_0`` (trivial / identity SFC) and ``μ_T``
    (a maximally-warped SFC). ``r_max`` bounds the radial schedule away
    from the degenerate unit-ball boundary; ``mu_t_max`` bounds the
    spatially-varying ``μ_T`` field strictly inside the unit ball so the
    induced quasi-conformal degradation stays a homeomorphism (Ahlfors).
    """

    model_config = {"extra": "forbid", "frozen": True}

    r_max: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description=(
            "Upper bound on the radial schedule r(t) ∈ [0, r_max]. Must be "
            "< 1 — the geodesic is degenerate at the unit-ball boundary."
        ),
    )
    spatial_size: int = Field(
        default=16,
        ge=2,
        description=(
            "Grid resolution H=W of the Beltrami-coefficient maps μ_0 / μ_T "
            "used to parameterise the SFC degradation."
        ),
    )
    n_steps: int = Field(
        default=8,
        ge=2,
        description="Number of cold-diffusion schedule steps t ∈ linspace(0, 1, n_steps).",
    )
    lambda_teichmuller: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight on the geodesic constant-speed (log-dilatation) regulariser.",
    )
    mu_t_max: float = Field(
        default=0.7,
        gt=0.0,
        lt=1.0,
        description=(
            "Magnitude bound on the spatially-varying μ_T endpoint field "
            "(|μ_T| ≤ mu_t_max < 1). The endpoint is a non-uniform radial "
            "Beltrami field — a degenerate spatially-constant endpoint would "
            "carry no spatial structure."
        ),
    )
    lambda_degrade: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight on the per-step SFC degrade→restore data-fidelity term "
            "(the cold-diffusion restoration objective)."
        ),
    )


__all__ = ["TeichmullerTrainingConfigSchema"]
