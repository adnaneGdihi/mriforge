"""Compiler-style recommendation + tag system for the audit.

The audit's job grew in three stages:

1. **Pass/fail checks** (Phase 0–6): the existing ``HealthCheckResult``
   fires on hard breakage.
2. **Recommendations** (this module): suggestions for design improvements
   that aren't failures — "declare ``adapters.post_model: [...]`` for
   round-trip symmetry", "this experiment looks multi-contrast; consider
   adding ``data.multi_contrast.acquisition_params`` so contrast-aware
   models can read the per-contrast vectors". Diagnostic-code style:
   ``AUDIT-R001``, ``AUDIT-R002``, …
3. **Tags**: derived properties of the experiment (``multi_contrast``,
   ``paradigm=diffusion``, ``task=reconstruction``, ``operates_on=kspace``,
   …) that downstream tooling (campaign builders, model variant
   generators, dataset selectors) can read without re-parsing the YAML.

The CLI compiles all three into a single report — like a compiler emits
errors, warnings, info, and lints. See
``docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RecommendationLevel(str, Enum):
    """Severity of a recommendation. Strictly weaker than HealthCheckResult."""

    HINT = "hint"  # Nice-to-have improvement.
    SUGGESTION = "suggestion"  # Recommended for clarity / robustness.
    LINT = "lint"  # YAML-style cleanup (redundant declarations, etc.).


@dataclass(frozen=True)
class Recommendation:
    """One recommendation emitted by an inference check.

    Diagnostic-code format: ``AUDIT-R<NNN>`` where NNN is a stable
    three-digit identifier. Codes never get reused after deprecation.
    """

    code: str  # e.g. "AUDIT-R001"
    title: str  # one-line summary
    detail: str  # multi-line explanation
    level: RecommendationLevel = RecommendationLevel.SUGGESTION
    yaml_keys: tuple[str, ...] = ()  # paths in config that motivated this
    suggested_yaml: str | None = None  # concrete YAML snippet to apply
    references: tuple[str, ...] = ()  # links to docs/specs


@dataclass(frozen=True)
class Tag:
    """Derived property of the experiment.

    Tags are pure functions of the YAML — same YAML in, same tags out.
    Downstream tooling (campaign builders, variant generators) can
    filter / route on tags without re-parsing the config.
    """

    name: str  # e.g. "multi_contrast", "paradigm"
    value: Any  # bool / str / list — type-free for YAML cleanliness
    derived_from: tuple[str, ...] = ()  # YAML paths that contributed
    note: str | None = None  # one-line human-readable context


@dataclass
class CompilerReport:
    """Aggregate output of the audit's "compile" pass.

    Holds the same ``HealthCheckResult`` list the existing audit produces
    plus the new ``Recommendation`` and ``Tag`` lists. The CLI renders
    all three sections.
    """

    spec_card: str
    results: list = field(default_factory=list)  # list[HealthCheckResult]
    recommendations: list[Recommendation] = field(default_factory=list)
    tags: list[Tag] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(getattr(r, "severity", None) == "error" for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(getattr(r, "severity", None) == "warning" for r in self.results)

    def get_tag(self, name: str) -> Tag | None:
        for t in self.tags:
            if t.name == name:
                return t
        return None


# ---------------------------------------------------------------------------
# Stable diagnostic-code registry
# ---------------------------------------------------------------------------
# Add new codes here as recommendation checks are added. Never renumber.

DIAGNOSTIC_CODES: dict[str, str] = {
    "AUDIT-R001": "Recommend declaring inverse adapter for round-trip symmetry",
    "AUDIT-R002": "Multi-contrast data declared but model lacks contrast conditioning",
    "AUDIT-R003": "Multi-contrast acquisition_params missing for multi-contrast data",
    "AUDIT-R004": "Adapter declared but capability metadata says native — likely redundant",
    "AUDIT-R005": "Diffusion paradigm but no validation visualization configured",
    "AUDIT-R006": "K-space data + image-domain losses without explicit adapter chain",
    "AUDIT-R007": "Reporting/metrics list does not include all enabled validation metrics",
    "AUDIT-R008": "Strategy paradigm differs from model.training_mode",
    "AUDIT-R009": "Patch size suspiciously small for declared task — verify intent",
    "AUDIT-R010": "Model declares contrast conditioning but data has no multi_contrast block",
    "AUDIT-R011": "Multi-contrast metadata declared via tags but no data.multi_contrast block",
    # Phase 8 — Transforms + Physics contracts
    "AUDIT-R020": "SENSE-style loss declared but no coil_sensitivity source",
    "AUDIT-R021": "Bloch simulator model lacks acquisition_params for one or more contrasts",
    "AUDIT-R022": "scan_mode declared but model is unannotated (cannot cross-check)",
    "AUDIT-R023": "DC layer enabled with no acceleration block — DC has nothing to project against",
    "AUDIT-R024": "Mamba-family model without explicit scan_mode declared — silently uses default",
    "AUDIT-R999": "(Internal) recommender crashed — file a bug",
}


__all__ = [
    "DIAGNOSTIC_CODES",
    "CompilerReport",
    "Recommendation",
    "RecommendationLevel",
    "Tag",
]
