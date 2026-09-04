"""Every declared ``losses.image_losses`` entry reaches the objective (mrixfields review 2026-09-03).

A strategy declares its loss ownership on the class: ``inline_losses`` (the
names it computes itself) and ``folds_image_losses`` (whether the other
declared entries reach the objective through the parent's builder path or the
fold). A declared name that is neither inline nor folded-and-registered is
dropped at runtime without a trace; on the mrixfields cohort 26 arms declared
an ``l1`` on score-matching and velocity strategies that never computed one,
and the audit's loss census read it as the objective. The rule lives in
``loss_folding.unreachable_image_losses``; the ratchet polarity is the usual
one: ERROR for a strategy that has declared, an INFO census line for one that
has not.

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

__all__ = ["declared_image_loss_names", "image_losses_reach_the_objective", "loss_ownership_defect"]

_CATEGORY = "dead_knob"
_NAME = "image_losses_reach_the_objective"
KEY = "losses.image_losses"


def declared_image_loss_names(settings: object) -> list[str]:
    """The enabled ``losses.image_losses`` names, raw as written."""
    losses = getattr(settings, "losses", None)
    entries = getattr(losses, "image_losses", None) or []
    out: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            name, enabled = entry.get("name"), entry.get("enabled", True)
        else:
            name, enabled = getattr(entry, "name", None), getattr(entry, "enabled", True)
        if name and enabled:
            out.append(str(name))
    return out


def loss_ownership_defect(names: list[str], strategy_cls: type | None) -> tuple[str | None, bool]:
    """``(message, declared)``: the finding or None, and whether the strategy declared.

    ``declared`` False means the strategy has not stated its ownership, which the
    caller reports as UNVERIFIED rather than as a pass.
    """
    from spectramr.infrastructure.training.strategies.loss_folding import (
        declared_folds_image_losses,
        declared_inline_losses,
        unreachable_image_losses,
    )
    from spectramr.models.losses.registry import LossRegistry

    if strategy_cls is None:
        return None, False
    inline = declared_inline_losses(strategy_cls)
    folds = declared_folds_image_losses(strategy_cls)
    if inline is None or folds is None:
        return None, False

    def _registered(name: str) -> bool:
        return name in LossRegistry.list_available() or name in getattr(
            LossRegistry, "_aliases", {}
        )

    lost = unreachable_image_losses(names, inline, folds, _registered)
    if not lost:
        return None, True
    route = "folds the builder's registered entries" if folds else "folds nothing"
    return (
        f"{KEY} declares {lost} but {strategy_cls.__name__} computes inline only "
        f"{sorted(inline) or 'nothing'} and {route}: the entry is dropped at runtime and the "
        "audit's loss census reads it as the objective."
    ), True


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T1,),
    subjects=(Subject.SETTINGS,),
    severity=Severity.ERROR,
    description="Every declared image loss is computed inline or folded by the strategy",
    fix_hint=(
        "Delete the entry, declare the name the strategy computes inline, or move the arm "
        "onto a strategy that folds the builder's losses (inline_losses / folds_image_losses "
        "on the strategy class say which)."
    ),
)
def image_losses_reach_the_objective(subject: WitnessSubject) -> WitnessVerdict:
    """Error on a declared entry no route computes; UNVERIFIED on an undeclared strategy."""
    settings = subject.settings
    names = declared_image_loss_names(settings)
    if not names:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="no image losses declared",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    strategy_cls = _resolve_strategy(settings)
    message, declared = loss_ownership_defect(names, strategy_cls)
    if not declared:
        who = strategy_cls.__name__ if strategy_cls is not None else "an unresolved strategy"
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message=(
                f"UNVERIFIED: {who} has not declared inline_losses / folds_image_losses, so "
                f"whether {names} reach the objective is not known here"
            ),
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    if message is None:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message=f"all {len(names)} declared image loss(es) reach {strategy_cls.__name__}'s objective",
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
