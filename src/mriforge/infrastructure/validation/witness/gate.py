"""Run the registered witnesses. The one entry point all four surfaces share.

The point of a single gate is that CI, ``mriforge audit``, the in-run self-report
and the in-run abort cannot disagree about what was checked. They differ only in
which tiers they request and what they do with the verdicts, never in the set of
detectors — which is what went wrong with the two pre-existing stacks.

A witness that raises is itself reported as a failed verdict rather than being
allowed to abort the sweep. A crashed detector is a finding: it means the arm is
unverified for that class, and turning it into a silent skip would hand back a
clean report for a check that never ran.
"""

from __future__ import annotations

import logging

from mriforge.infrastructure.validation.witness.registry import (
    Severity,
    Tier,
    WitnessVerdict,
    get_witness_registry,
)
from mriforge.infrastructure.validation.witness.subject import (
    WitnessSubject,
    WitnessSubjectUnavailableError,
)

logger = logging.getLogger(__name__)

__all__ = ["WitnessGateError", "run_witnesses", "scheduled_witnesses"]


class WitnessGateError(RuntimeError):
    """Error-severity verdicts were produced where they must block."""

    def __init__(self, verdicts: list[WitnessVerdict]) -> None:
        self.verdicts = verdicts
        detail = "\n".join(f"  - {v.witness_name}: {v.message}" for v in verdicts)
        super().__init__(f"{len(verdicts)} witness error(s):\n{detail}")


def scheduled_witnesses(subject: WitnessSubject, *, tiers: frozenset[Tier]):
    """The witnesses that both belong to a requested tier and can run here.

    Separated from :func:`run_witnesses` so a surface can report what it *will*
    check, and so the "did anything run at all?" question is answerable. A sweep
    that scheduled zero witnesses is not a pass.
    """
    for witness in get_witness_registry().all():
        if not (witness.tiers & tiers):
            continue
        if not witness.applicability.matches(subject.config):
            continue
        if not subject.provides(witness.subjects):
            continue
        yield witness


def run_witnesses(subject: WitnessSubject, *, tiers: frozenset[Tier]) -> list[WitnessVerdict]:
    """Run every applicable witness and collect verdicts."""
    verdicts: list[WitnessVerdict] = []
    for witness in scheduled_witnesses(subject, tiers=tiers):
        try:
            result = witness.fn(subject)
        except WitnessSubjectUnavailableError as exc:
            # A scheduling bug, not a config defect: `provides()` said yes and the
            # witness still asked for something else. Report it loudly instead of
            # letting the arm look verified.
            verdicts.append(
                WitnessVerdict(
                    witness_name=witness.name,
                    passed=False,
                    message=f"scheduling bug: {exc}",
                    severity=Severity.ERROR,
                    category="witness_scheduling",
                    stage=witness.stage,
                )
            )
            continue
        except Exception as exc:
            logger.exception("witness %s raised", witness.name)
            verdicts.append(
                WitnessVerdict(
                    witness_name=witness.name,
                    passed=False,
                    message=f"witness raised {type(exc).__name__}: {exc}",
                    severity=Severity.ERROR,
                    category="witness_crash",
                    stage=witness.stage,
                )
            )
            continue
        verdicts.extend(result if isinstance(result, list) else [result])
    return verdicts


def assert_no_errors(verdicts: list[WitnessVerdict]) -> None:
    """Raise when any verdict is a failing error. Used by blocking surfaces."""
    errors = [v for v in verdicts if not v.passed and v.severity == Severity.ERROR]
    if errors:
        raise WitnessGateError(errors)
