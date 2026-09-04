"""Training-budget witnesses (cohort review 2026-09-02, T0.9).

Two shapes the census found across ``experiments/inprogress``:

* ``optimization.scheduler.warmup_steps >= training.max_iterations`` on four
  arms -- two of them with ``max_iterations: 1``, a smoke setting left in a
  research arm. The learning rate never leaves warm-up, so the run measures
  the warm-up ramp, not the method.
* ``training.epochs: 0`` with no ``max_iterations`` on seven arms: a budget of
  zero steps that trains nothing and still writes a run directory.

Both are decidable from the raw document (DECLARE stage). Registration is by
import (the witness package walk).
"""

from __future__ import annotations

from spectramr.config.training_budget import NO_TRAINING_MODES, zero_budget_defect
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["training_budget_is_positive", "warmup_shorter_than_budget"]

_CATEGORY = "training_budget"
_WARMUP = "warmup_shorter_than_budget"
_BUDGET = "training_budget_is_positive"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _verdict(name: str, passed: bool, message: str, **extra) -> WitnessVerdict:
    return WitnessVerdict(
        witness_name=name,
        passed=passed,
        message=message,
        severity=Severity.ERROR,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T0,
        **extra,
    )


@register_witness(
    _WARMUP,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.CONFIG,),
    severity=Severity.ERROR,
    description="The learning-rate warm-up ends before the iteration budget does",
    fix_hint=(
        "Raise training.max_iterations above optimization.scheduler.warmup_steps, "
        "or shorten the warm-up; max_iterations: 1 is a smoke setting."
    ),
)
def warmup_shorter_than_budget(subject: WitnessSubject) -> WitnessVerdict:
    """Error when the scheduler warm-up covers the whole (or more than the) run."""
    raw = subject.raw_config
    warmup = _number(((raw.get("optimization") or {}).get("scheduler") or {}).get("warmup_steps"))
    max_iter = _number((raw.get("training") or {}).get("max_iterations"))
    if warmup is None or max_iter is None or max_iter <= 0:
        return _verdict(
            _WARMUP, True, "no bounded warm-up/budget pair declared; nothing to compare"
        )
    if warmup >= max_iter:
        return _verdict(
            _WARMUP,
            False,
            f"optimization.scheduler.warmup_steps={int(warmup)} >= training.max_iterations="
            f"{int(max_iter)}: the learning rate never leaves warm-up, so the run measures "
            "the ramp rather than the method.",
            yaml_keys=("optimization.scheduler.warmup_steps", "training.max_iterations"),
            fix_hint="Raise training.max_iterations or shorten the warm-up.",
        )
    return _verdict(_WARMUP, True, f"warm-up {int(warmup)} < budget {int(max_iter)} iterations")


@register_witness(
    _BUDGET,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.CONFIG,),
    severity=Severity.ERROR,
    description="The arm declares a positive training budget (epochs or max_iterations)",
    fix_hint="Set training.epochs >= 1, or a positive training.max_iterations.",
)
def training_budget_is_positive(subject: WitnessSubject) -> WitnessVerdict:
    """Error on ``epochs: 0`` with no positive ``max_iterations``."""
    training = subject.raw_config.get("training") or {}
    if str(training.get("training_mode") or "").lower() in NO_TRAINING_MODES:
        return _verdict(
            _BUDGET,
            True,
            f"training_mode={training.get('training_mode')!r} optimises nothing; a zero budget is its contract",
        )
    message = zero_budget_defect(training)
    if message is not None:
        return _verdict(
            _BUDGET,
            False,
            message,
            yaml_keys=("training.epochs", "training.max_iterations"),
            fix_hint="Set training.epochs >= 1 or a positive training.max_iterations.",
        )
    return _verdict(_BUDGET, True, "a positive training budget is declared")
