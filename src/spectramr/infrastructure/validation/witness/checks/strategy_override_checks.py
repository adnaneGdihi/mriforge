"""``training.strategy_class`` versus ``training.training_mode`` (cohort review 2026-09-02, T0.10).

``get_strategy_class`` lets an explicit ``strategy_class`` win over the
``training_mode`` map. That is the right precedence for a specialisation
(``training_mode: diffusion`` with an explicit ``AmbientDiffusionStrategy``), and
it is the facade shape when the explicit class is a *generic base* while the
mode names a mechanism the map resolves to a specific class:
``training_mode: symplectic_bloch`` (mapped to ``CycleBlochStrategy``) with
``strategy_class: ReconstructionTrainingStrategy`` runs a plain reconstruction
under a mechanism's name. Measured on 2026-09-03: 642 of 647 arms declare an
explicit class, 23 disagree with the map, 4 are this shape.

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

__all__ = ["GENERIC_BASES", "strategy_class_matches_training_mode", "strategy_override_defect"]

_CATEGORY = "strategy_dispatch"
_NAME = "strategy_class_matches_training_mode"
#: Bases that carry no mechanism of their own: an explicit one of these under a
#: mechanism-naming mode is the facade shape. Compared by class name so the
#: witness needs no strategy import of its own.
GENERIC_BASES = frozenset(
    {
        "ReconstructionTrainingStrategy",
        "PhysicsDrivenTrainingStrategy",
        "DiffusionTrainingStrategy",
        "GANTrainingStrategy",
        "VAETrainingStrategy",
        "BaseTrainingStrategy",
    }
)
_OVERRIDE_TAG = "strategy_override_reason"


def _leaf(spec: object) -> str:
    return str(spec).rsplit(".", 1)[-1]


def strategy_override_defect(
    explicit_cls: type | None, mapped_cls: type | None, override_reason: object
) -> str | None:
    """The finding, or None: a generic explicit class under a mode that maps elsewhere."""
    if explicit_cls is None or mapped_cls is None or explicit_cls is mapped_cls:
        return None
    if issubclass(explicit_cls, mapped_cls):
        return None  # a specialisation of what the mode names
    if explicit_cls.__name__ not in GENERIC_BASES:
        return None  # a sibling mechanism, declared by name
    if override_reason:
        return None
    return (
        f"training.strategy_class={explicit_cls.__name__} is a generic base while "
        f"training.training_mode maps to {mapped_cls.__name__}: the arm runs without the "
        "mechanism its mode names"
    )


def _resolve(settings):
    """(explicit class or None, mapped class or None) without raising."""
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    training = getattr(settings, "training", None)
    explicit_spec = getattr(training, "strategy_class", None)
    mode = getattr(training, "training_mode", None)
    factory = TrainingStrategyFactory()
    explicit_cls = mapped_cls = None
    if explicit_spec:
        try:
            explicit_cls = factory._load_strategy_class(explicit_spec)
        except Exception:
            explicit_cls = None
    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS.get(str(mode)) if mode else None
    if path:
        try:
            mapped_cls = factory._load_strategy_class(path)
        except Exception:
            mapped_cls = None
    return explicit_cls, mapped_cls


def _override_reason(settings) -> object:
    metadata = getattr(settings, "metadata", None)
    tags = getattr(metadata, "tags", None)
    if isinstance(tags, dict):
        return tags.get(_OVERRIDE_TAG)
    return getattr(tags, _OVERRIDE_TAG, None)


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T0, Tier.T1),
    subjects=(Subject.SETTINGS,),
    severity=Severity.ERROR,
    description="An explicit generic strategy class does not hide the mechanism training_mode names",
    fix_hint=(
        "Set training.training_mode to what the arm runs (e.g. reconstruction), give the arm the "
        "strategy its mode names, or record metadata.tags.strategy_override_reason."
    ),
)
def strategy_class_matches_training_mode(subject: WitnessSubject) -> WitnessVerdict:
    settings = subject.settings
    explicit_cls, mapped_cls = _resolve(settings)
    message = strategy_override_defect(explicit_cls, mapped_cls, _override_reason(settings))
    if message is None:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="strategy_class and training_mode agree",
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
        yaml_keys=("training.strategy_class", "training.training_mode"),
        fix_hint=(
            "Set training.training_mode to what the arm runs, give the arm the strategy its "
            "mode names, or record metadata.tags.strategy_override_reason."
        ),
    )
