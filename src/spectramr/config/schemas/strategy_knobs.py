"""Per-``strategy_class`` knob registry — the seam that closes the facade hole.

The central ``training:`` block is ``StrategySchema`` (``extra='allow'``) so that
each training strategy can read its own knob vocabulary off the config via
``getattr``. The cost (pitfall #16, the largest class in the 2026-06 validation
sweep) is that a typo'd or unimplemented strategy knob is **accepted silently**:
the run smoke-PASSes while the advertised mechanism never engages.

This module is the **B1** mechanism from
``docs/superpowers/specs/2026-06-12-schema-strictness-and-strategy-knob-audit.rst``:
a registry mapping a fully-qualified ``strategy_class`` to the knob names it
actually reads, plus a validator that flags *undeclared* knobs present on a
config. ``register the advertised set and let the audit reject illegal YAML``
(CLAUDE.md #9/#15) — applied to the open strategy section.

.. important::

   This seam is intentionally **not** auto-wired into the default audit rule set.
   Under the smoke wrapper's ``--strict`` mode a warning exits 2 (pitfall #10), so
   activating the rule before every strategy has a complete, cluster-verified knob
   inventory would reject the ~30 ``inprogress`` configs that legitimately carry
   strategy knobs. The validator is **inert** for any ``strategy_class`` that has
   not registered its knobs. Populating the registry per strategy and wiring the
   rule (advisory first, then strict per strategy) is the Tier-B rollout.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# strategy_class (FQ path) -> frozenset of knob names that strategy reads.
_STRATEGY_KNOBS: dict[str, frozenset[str]] = {}


def register_strategy_knob_names(strategy_class: str, names: Iterable[str]) -> None:
    """Register the knob vocabulary for a strategy by explicit name set.

    Idempotent-union: re-registering merges (so a base class and a subclass can
    each contribute). Raises on an empty registration to keep the advertised set
    meaningful.
    """
    new = frozenset(names)
    if not new:
        raise ValueError(
            f"register_strategy_knob_names({strategy_class!r}, ...) got an empty "
            f"knob set — register the knobs the strategy actually reads, or do "
            f"not register it at all (the validator is inert for unregistered "
            f"strategies)."
        )
    _STRATEGY_KNOBS[strategy_class] = _STRATEGY_KNOBS.get(strategy_class, frozenset()) | new


def register_strategy_knobs(strategy_class: str):
    """Decorator: register a knob *schema* — its field names become the knob set.

    Use on a ``StrictSchema`` subclass that declares one field per knob the
    strategy reads (giving types + defaults for the eventual typed validation),
    e.g.::

        @register_strategy_knobs("....ColdDiffusionStrategy")
        class ColdDiffusionKnobs(StrictSchema):
            sampler_steps: int = 28
            multistep_cold_sampling: bool = False
    """

    def _decorator(schema_cls):
        register_strategy_knob_names(strategy_class, schema_cls.model_fields.keys())
        return schema_cls

    return _decorator


def known_knobs(strategy_class: str) -> frozenset[str] | None:
    """Return the registered knob set for ``strategy_class``, or ``None`` if it
    has not registered (the validator is inert in that case)."""
    return _STRATEGY_KNOBS.get(strategy_class)


def _declared_training_fields() -> frozenset[str]:
    """Field names declared on the central ``TrainingStrategyConfigSchema`` —
    always permitted (they are real schema fields, not ``extra`` knobs).

    Imported lazily to avoid any import-order coupling with ``training.base``.
    """
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    return frozenset(TrainingStrategyConfigSchema.model_fields.keys())


def validate_strategy_knobs(config: dict[str, Any]) -> list[str]:
    """Flag knobs present on ``training:`` that the resolved strategy does not
    declare. Pure function over a raw config dict; returns advisory messages.

    Inert (returns ``[]``) when there is no ``strategy_class`` or the strategy
    has not registered its knobs — so it can never false-reject an unmigrated arm.
    """
    training = config.get("training") or {}
    if not isinstance(training, dict):
        return []
    strategy_class = training.get("strategy_class")
    if not strategy_class:
        return []
    registered = known_knobs(strategy_class)
    if registered is None:
        return []  # opt-in: inert until this strategy registers its knobs

    allowed = registered | _declared_training_fields()
    undeclared = sorted(k for k in training if k not in allowed)
    return [
        f"training.{k}: knob is not declared by strategy {strategy_class!r} "
        f"(undeclared/typo'd knobs are silently swallowed by extra='allow' — "
        f"pitfall #16). Add it to the strategy's registered knob set if real."
        for k in undeclared
    ]


def build_strategy_knob_rule(severity: str = "warning"):
    """Wrap :func:`validate_strategy_knobs` as a ``ValidationRule`` for opt-in
    registration into a ``ValidatorRegistry`` (see the module note on why this is
    not auto-wired into the strict default set yet)."""
    from spectramr.config.schemas.validator_registry import ValidationRule

    return ValidationRule(
        name="strategy_knobs_are_declared",
        validator_fn=validate_strategy_knobs,
        category="training",
        description=(
            "Strategy-specific knobs on the extra='allow' training block must be "
            "declared in the per-strategy knob registry (closes the facade hole)."
        ),
        severity=severity,
    )


__all__ = [
    "build_strategy_knob_rule",
    "known_knobs",
    "register_strategy_knob_names",
    "register_strategy_knobs",
    "validate_strategy_knobs",
]
