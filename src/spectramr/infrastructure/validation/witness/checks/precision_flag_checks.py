"""``training.enable_mixed_precision`` is read by nothing (#887, cohort review 2026-09-02).

The live AMP switch is ``optimization.precision.enabled``. The ``training``
spelling was declared on no live schema and read by no code path, so an arm
that set it ran at whatever the live block said: 238 ``inprogress/`` arms
carried it, nine of them (hilbert_mamba) disagreeing with a live ``true``.
The ``inprogress/`` corpus is drained; ``active/``, ``validated/``,
``experiments/training/``, ``experiments/templates/`` and test fixtures still
carry 84 files with the line, and those trees are the owner's to migrate, so
the spelling cannot yet be retired with a ``raise`` rename record (which is
corpus-wide) and cannot be folded (a fold must stay inside its block). Until
those trees drain, this witness is the owner of the rule on the audit surface.

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

__all__ = ["DEAD_FLAG", "LIVE_FLAG", "dead_precision_flag_defect", "no_dead_precision_flag"]

_CATEGORY = "dead_knob"
_NAME = "no_dead_precision_flag"
DEAD_FLAG = "training.enable_mixed_precision"
LIVE_FLAG = "optimization.precision.enabled"


def dead_precision_flag_defect(raw: dict | None) -> str | None:
    """The finding, or None: the dead spelling declared, with what the live block says."""
    training = (raw or {}).get("training") or {}
    if "enable_mixed_precision" not in training:
        return None
    declared = training.get("enable_mixed_precision")
    precision = ((raw or {}).get("optimization") or {}).get("precision") or {}
    live = precision.get("enabled", False) if isinstance(precision, dict) else False
    verb = "agrees with" if bool(declared) == bool(live) else "DISAGREES with"
    return (
        f"{DEAD_FLAG}={declared!r} is read by nothing and {verb} the live "
        f"{LIVE_FLAG}={bool(live)!r}; the run uses the live value either way."
    )


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.CONFIG,),
    severity=Severity.ERROR,
    description="The arm does not declare the unread training.enable_mixed_precision flag",
    fix_hint=f"Delete {DEAD_FLAG}; the AMP switch is {LIVE_FLAG} (#887).",
)
def no_dead_precision_flag(subject: WitnessSubject) -> WitnessVerdict:
    """Error on the dead ``training.enable_mixed_precision`` spelling."""
    message = dead_precision_flag_defect(subject.raw_config)
    if message is None:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message=f"{DEAD_FLAG} not declared",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T0,
        )
    return WitnessVerdict(
        witness_name=_NAME,
        passed=False,
        message=message,
        severity=Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T0,
        yaml_keys=(DEAD_FLAG, LIVE_FLAG),
        fix_hint=f"Delete {DEAD_FLAG}; the AMP switch is {LIVE_FLAG} (#887).",
    )
