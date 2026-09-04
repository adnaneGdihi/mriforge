"""``physics.digital_twin.ood_acceleration_range`` has two readers and one condition (VF review 2026-09-03).

The range is consumed by the validation step of the strategies that undersample
through the digital twin (``virtual_fiducial``, ``vf_admm`` and their
subclasses, which declare ``reads_ood_acceleration_range``). The two conditions
that live inside the block -- the twin must be enabled and must undersample, so
an out-of-distribution rung has an in-distribution rate to be out of -- are
owned by ``DigitalTwinConfig``'s validator and fail at load; this witness asks
the one question the schema cannot: does the resolved strategy read it. Its
predecessor, ``undersampling.out_of_distribution_range``, was declared on 58
arms and read by nothing (the dropped-key baseline recorded every one), which
is the shape this witness exists to stop recurring.

Registration is by import (the witness package walk).
"""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.checks.undersampling_checks import (
    _resolve_strategy,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["KEY", "READER_FLAG", "ood_acceleration_range_is_read", "ood_range_defect"]

_CATEGORY = "dead_knob"
_NAME = "ood_acceleration_range_is_read"
KEY = "physics.digital_twin.ood_acceleration_range"
READER_FLAG = "reads_ood_acceleration_range"


def _strategy_reads_it(strategy_cls: type | None) -> bool:
    if strategy_cls is None:
        return False
    # MRO scan, as ``undersampling_consumers`` does: a subclass inherits the
    # declaration, a plain attribute read would miss a mixin-declared one.
    return any(base.__dict__.get(READER_FLAG, False) is True for base in strategy_cls.__mro__)


def ood_range_defect(settings: object, strategy_cls: type | None) -> str | None:
    """The finding, or None: a declared range that no validation pass will read."""
    twin = getattr(getattr(settings, "physics", None), "digital_twin", None)
    declared = getattr(twin, "ood_acceleration_range", None)
    if not declared:
        return None
    # The intra-block conditions (twin disabled, twin not undersampling) are
    # owned by ``DigitalTwinConfig``'s validator and fail at load; only the
    # cross-object question is asked here.
    problems: list[str] = []
    if strategy_cls is None:
        problems.append("the training strategy could not be resolved")
    elif not _strategy_reads_it(strategy_cls):
        problems.append(
            f"{strategy_cls.__name__} does not read it (no {READER_FLAG} in its MRO; the "
            "readers are the virtual_fiducial and vf_admm strategies)"
        )
    if not problems:
        return None
    return f"{KEY}={list(declared)} is declared but " + "; ".join(problems) + "."


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T1,),
    subjects=(Subject.SETTINGS,),
    severity=Severity.ERROR,
    description="A declared out-of-distribution acceleration range is read by the validation pass",
    fix_hint=(
        f"Delete {KEY}, or move the arm onto a virtual_fiducial / vf_admm strategy, whose "
        "validation step re-scores every rung as val_ood_{R}x_<metric>."
    ),
)
def ood_acceleration_range_is_read(subject: WitnessSubject) -> WitnessVerdict:
    """Error on a range nothing will read; pass (INFO) otherwise."""
    settings = subject.settings
    twin = getattr(getattr(settings, "physics", None), "digital_twin", None)
    declared = getattr(twin, "ood_acceleration_range", None)
    if not declared:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="no out-of-distribution acceleration range declared",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    strategy_cls = _resolve_strategy(settings)
    message = ood_range_defect(settings, strategy_cls)
    if message is None:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message=(
                f"{KEY}={list(declared)} is re-scored by {strategy_cls.__name__}.validation_step "
                f"at {len(declared)} rung(s) beside the in-distribution "
                f"{getattr(twin, 'acceleration', '?')}x"
            ),
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    return WitnessVerdict(
        witness_name=_NAME,
        passed=False,
        message=message,
        severity=Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T1,
        yaml_keys=(KEY,),
    )
