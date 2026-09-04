r"""Validation badge — clinical-pilot-eligibility gate (ULF-PR-26 / V.3).

Emits a single JSON document whose ``eligible`` flag is the audit
gate that ``validation_badge_gates_complete`` consults. Each gate
returns a pass/fail bool and a descriptor; the badge is *only*
``eligible`` if every required gate passes.

Implements ``TODO/backlog_ulf_radiologist_acceptance_roadmap.md`` §V.3.

Design
------
The badge is intentionally schema-light: each gate is a free-text
identifier mapping to a ``GateResult`` record. The audit consults
``required_gates`` to know which keys must be present, so adding a
new gate is purely additive — no breaking change to existing badges.

Required gates (from the plan)
------------------------------
- ``iqm`` — image-quality metric thresholds met
- ``downstream`` — downstream-task evaluation passed
- ``hallucination`` — calibrated hallucination FPR within bound
- ``conformal_coverage`` — empirical coverage ≥ ``1 - α``
- ``pathology_recall`` — recall certificate certified for every class
- ``pac_bayes`` — non-vacuous PAC-Bayes generalisation bound

The orchestrator may run extra gates; only the six above are *required*
for clinical-pilot eligibility.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The exact set the audit hook checks for.
DEFAULT_REQUIRED_GATES: tuple[str, ...] = (
    "iqm",
    "downstream",
    "hallucination",
    "conformal_coverage",
    "pathology_recall",
    "pac_bayes",
)


@dataclass(frozen=True)
class GateResult:
    """Per-gate result.

    Attributes:
        passed: ``True`` iff the gate cleared its threshold.
        details: Human-readable description (the raw certificate /
            bound / metric value, formatted for the JSON dump).
        threshold: Optional numeric threshold the gate evaluated against.
        observed: Optional observed value.
    """

    passed: bool
    details: str
    threshold: float | None = None
    observed: float | None = None

    def to_dict(self) -> dict:
        """JSON-friendly serialisation."""
        return asdict(self)


@dataclass(frozen=True)
class ValidationBadge:
    """The single JSON document gating clinical-pilot eligibility.

    Attributes:
        gates: Mapping ``gate_name → GateResult``.
        required_gates: Tuple of required gate names. The badge is
            ``eligible`` iff every required gate is present and passes.
        eligible: Final eligibility flag (computed at construction).
        missing_gates: Required gates absent from ``gates``.
        failed_gates: Required gates present but with ``passed = False``.
    """

    gates: dict[str, GateResult]
    required_gates: tuple[str, ...] = field(default_factory=lambda: DEFAULT_REQUIRED_GATES)
    eligible: bool = False
    missing_gates: tuple[str, ...] = ()
    failed_gates: tuple[str, ...] = ()

    @classmethod
    def from_gates(
        cls,
        gates: dict[str, GateResult],
        required_gates: Sequence[str] | None = None,
    ) -> ValidationBadge:
        """Construct a badge, computing ``eligible`` / ``missing`` / ``failed``."""
        req = tuple(required_gates) if required_gates is not None else DEFAULT_REQUIRED_GATES
        if not isinstance(gates, dict):
            raise TypeError("gates must be a dict[str, GateResult]")
        for name, result in gates.items():
            if not isinstance(result, GateResult):
                raise TypeError(
                    f"gates[{name!r}] must be a GateResult; got {type(result).__name__}"
                )
        missing = tuple(g for g in req if g not in gates)
        failed = tuple(g for g in req if g in gates and not gates[g].passed)
        eligible = not missing and not failed
        return cls(
            gates=gates,
            required_gates=req,
            eligible=eligible,
            missing_gates=missing,
            failed_gates=failed,
        )

    def to_dict(self) -> dict:
        """JSON-friendly serialisation."""
        return {
            "eligible": self.eligible,
            "missing_gates": list(self.missing_gates),
            "failed_gates": list(self.failed_gates),
            "required_gates": list(self.required_gates),
            "gates": {k: v.to_dict() for k, v in self.gates.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the badge to a JSON string ready for disk."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the badge to ``path`` as JSON.

        Parent directories are created on demand. Returns the absolute
        path of the written file so downstream tooling (the regulatory
        package CLI, ULF-PR-27) can list emitted artifacts.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(indent=indent))
        return out


@dataclass(frozen=True)
class BadgeAuditResult:
    """Outcome of the ``validation_badge_gates_complete`` audit hook.

    Attributes:
        passed: ``True`` iff the badge is clinical-pilot-eligible.
        missing_gates: Required gates absent from the badge.
        failed_gates: Required gates present but with ``passed = False``.
        message: Single human-readable summary line.
    """

    passed: bool
    missing_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    message: str


def validation_badge_gates_complete(badge: ValidationBadge) -> BadgeAuditResult:
    """Tier-1 audit hook — refuses clinical-pilot eligibility without all gates green.

    Implements the audit-hook contract from
    ``TODO/backlog_ulf_radiologist_acceptance_roadmap.md`` ULF-PR-26.
    A run cannot be marked "clinical-pilot-eligible" if any required
    gate is missing or failing — silent fallbacks at this gate would
    leak unsafe models into pilot deployment (CLAUDE.md pitfall #9).

    Args:
        badge: A :class:`ValidationBadge` produced by the nightly
            validation pipeline.

    Returns:
        :class:`BadgeAuditResult` with ``passed = badge.eligible`` plus
        diagnostic detail (which gates are missing or failing). The
        regulatory CLI (ULF-PR-27) consults this to decide whether to
        emit the bundle.
    """
    if badge.eligible:
        return BadgeAuditResult(
            passed=True,
            missing_gates=(),
            failed_gates=(),
            message=(
                f"All {len(badge.required_gates)} required gates passed — clinical-pilot eligible."
            ),
        )
    parts: list[str] = []
    if badge.missing_gates:
        parts.append(f"missing: {list(badge.missing_gates)}")
    if badge.failed_gates:
        parts.append(f"failed: {list(badge.failed_gates)}")
    return BadgeAuditResult(
        passed=False,
        missing_gates=badge.missing_gates,
        failed_gates=badge.failed_gates,
        message="Validation badge NOT eligible — " + "; ".join(parts) + ".",
    )


__all__ = [
    "DEFAULT_REQUIRED_GATES",
    "BadgeAuditResult",
    "GateResult",
    "ValidationBadge",
    "validation_badge_gates_complete",
]
