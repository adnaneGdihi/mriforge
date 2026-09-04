"""Held-out test-set witnesses (cohort review 2026-09-02, T0.3).

Measured across ``experiments/inprogress`` on 2026-09-02: **0 of 647 arms**
declare a held-out test set, so every reported number is computed on the set
that also selects the checkpoint. Two witnesses:

* ``held_out_test_declared`` -- advisory (INFO). An arm whose role is a
  baseline / headline / reference reports on its selection set unless it
  declares ``data.source.test_index_path`` or ``data.enable_test_split``.
  Advisory now, promoted once the Tier-1 cohorts adopt the protocol (the
  ratchet every gate here follows); a warning today would fail ``--strict``
  on every baseline arm at once.
* ``held_out_test_split_disjoint`` -- ERROR. When a test manifest is declared
  and present, no subject or file of it may appear in the training pool
  (train + validation manifests). The scan lives in
  ``spectramr.data.split_leakage.analyze_test_split_leakage`` (data SSOT).

Registration is by import (the witness package walk).
"""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = [
    "REPORTING_ROLES",
    "declares_held_out_test",
    "held_out_test_declared",
    "held_out_test_split_disjoint",
]

_CATEGORY = "protocol"
_DECLARED = "held_out_test_declared"
_DISJOINT = "held_out_test_split_disjoint"

#: ``metadata.role`` / ``metadata.tags.role`` values whose numbers are the ones
#: a paper quotes. Ablations, variants and comparisons inherit their reference's
#: protocol and are not nagged separately.
REPORTING_ROLES: frozenset[str] = frozenset({"baseline", "headline", "reference", "ssot"})


def declares_held_out_test(data: object) -> bool:
    """True when the raw ``data:`` block declares a held-out test set."""
    if not isinstance(data, dict):
        return False
    source = data.get("source") or {}
    if isinstance(source, dict) and source.get("test_index_path"):
        return True
    return bool(data.get("enable_test_split"))


def _role(raw_config: dict) -> str | None:
    metadata = raw_config.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None
    role = metadata.get("role")
    if role is None and isinstance(metadata.get("tags"), dict):
        role = metadata["tags"].get("role")
    return str(role).strip().lower() if role else None


@register_witness(
    _DECLARED,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.CONFIG,),
    severity=Severity.INFO,
    description="A baseline/headline arm declares the held-out set it reports on",
    fix_hint=(
        "Declare data.source.test_index_path (a subject-disjoint manifest) or "
        "data.enable_test_split: true, and report the headline table on it; "
        "validation stays the checkpoint-selection set."
    ),
)
def held_out_test_declared(subject: WitnessSubject) -> WitnessVerdict:
    """Advisory: reporting arms without a held-out set report on their selection set."""
    role = _role(subject.raw_config)
    data = subject.raw_config.get("data") or {}
    if role not in REPORTING_ROLES:
        return WitnessVerdict(
            witness_name=_DECLARED,
            passed=True,
            message=f"role={role!r}: inherits its reference arm's reporting protocol",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T0,
        )
    if declares_held_out_test(data):
        return WitnessVerdict(
            witness_name=_DECLARED,
            passed=True,
            message=f"role={role!r}: held-out test set declared",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T0,
        )
    return WitnessVerdict(
        witness_name=_DECLARED,
        passed=False,
        message=(
            f"role={role!r} but no held-out test set is declared: every number this "
            "arm reports is computed on the validation set that also selects its "
            "checkpoint (optimistic by construction). Advisory until the cohort "
            "adopts data.source.test_index_path."
        ),
        severity=Severity.INFO,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T0,
        yaml_keys=("data.source.test_index_path", "data.enable_test_split"),
        fix_hint="Declare data.source.test_index_path or data.enable_test_split: true.",
    )


@register_witness(
    _DISJOINT,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T1,),
    subjects=(Subject.SETTINGS,),
    severity=Severity.ERROR,
    description="A declared held-out test manifest shares no subject or file with train/val",
    fix_hint=(
        "Rebuild the test manifest so no subject (or on-disk file) of it appears in "
        "data.source.index_path or data.source.validation_index_path."
    ),
)
def held_out_test_split_disjoint(subject: WitnessSubject) -> WitnessVerdict:
    """Error when the declared test set overlaps the training pool."""
    from spectramr.data.split_leakage import analyze_test_split_leakage

    report = analyze_test_split_leakage(subject.settings)
    if report.status == "leak":
        sample = ", ".join(report.overlap[:8])
        return WitnessVerdict(
            witness_name=_DISJOINT,
            passed=False,
            message=(
                f"HELD-OUT LEAK ({report.key_kind}-level): {report.detail} "
                f"pool={report.n_train} test={report.n_val}. Overlap: {sample}"
            ),
            severity=Severity.ERROR,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
            yaml_keys=(
                "data.source.test_index_path",
                "data.source.index_path",
                "data.source.validation_index_path",
            ),
        )
    if report.status == "skipped":
        return WitnessVerdict(
            witness_name=_DISJOINT,
            passed=True,
            message=f"skipped: {report.detail}",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    return WitnessVerdict(
        witness_name=_DISJOINT,
        passed=True,
        message=f"{report.detail} pool={report.n_train} test={report.n_val}.",
        severity=Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T1,
    )
