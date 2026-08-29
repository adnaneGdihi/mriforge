"""Audit-tier configuration sub-schemas (Idea 7 onwards)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema

from .ksd import KSDAuditConfigSchema


class AuditConfigSchema(StrictSchema):
    """Top-level container for the optional ``audit:`` YAML block.

    Mounts the per-audit sub-schemas (currently only KSD; expandable).
    Declared as a TrainingSettings field so YAMLs that opt into
    ``audit.ksd`` are accepted under ``extra="forbid"``.
    """

    ksd: KSDAuditConfigSchema | None = Field(
        default=None,
        description=(
            "Tier-3 kernelised Stein discrepancy defensibility audit "
            "(audit_plan_novel.md Idea 7). Run via "
            "`python -m mriforge.cli audit-ksd <config>`."
        ),
    )


__all__ = ["AuditConfigSchema", "KSDAuditConfigSchema"]
