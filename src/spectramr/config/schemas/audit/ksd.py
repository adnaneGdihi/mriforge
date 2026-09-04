"""Pydantic schema for the Tier-3 KSD defensibility audit (Idea 7)."""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrategySchema


class KSDAuditConfigSchema(StrategySchema):
    """Tier-3 audit configuration for the kernelised Stein discrepancy.

    Lives at ``audit.ksd`` in the v6.0 reference schema. Opt-in: when
    ``enabled`` is False the Tier-3 hook is not invoked.
    """

    enabled: bool = Field(
        default=False,
        description="Run the KSD defensibility audit at Tier 3.",
    )
    samples: int = Field(
        default=256,
        ge=4,
        description="Number of held-out samples used for the test.",
    )
    kernel: str = Field(
        default="imq",
        description="Kernel family (only 'imq' supported today).",
    )
    threshold: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Wild-bootstrap p-value below which the audit fails.",
    )
    bootstrap_replications: int = Field(
        default=500,
        ge=50,
        description="Number of wild-bootstrap replications.",
    )
    bandwidth: float | None = Field(
        default=None,
        gt=0.0,
        description="Kernel bandwidth; None ⇒ median heuristic.",
    )
