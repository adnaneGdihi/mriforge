"""``validation_metric_names_resolve`` (cohort review 2026-09-02, T0.5).

``metrics.compute`` is gated against ``MetricsRegistry`` at strategy
construction (#173) and at audit (``check_metric_names_are_registered``);
``validation.scoring.compute``, ``metrics.best_metric_name`` and
``early_stopping.metric`` were not. Thirteen arms used the validation list to
name keys their strategy's own ``validation_step`` writes (``field_mse`` on
the qMRI arms) -- honest keys the registry does not know -- and one arm once
selected its checkpoint on ``val_nmse``, a key no path produced, so early
stopping never fired.

A strategy now declares the keys it writes itself in
``capabilities.emitted_metrics``. Every name here must resolve in the
registry, in that set, or in the universal set (``val_loss``); a selector may
also carry a ``val_`` prefix and a ``_<n>x`` / ``_mean`` cascade suffix.
Error when the strategy has declared its emitted set (opt-in strictness);
INFO census line when it has not.
"""

from __future__ import annotations

import re

from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["unresolved_metric_names", "validation_metric_names_resolve"]

_NAME = "validation_metric_names_resolve"
_CATEGORY = "metrics_misconfiguration"
_UNIVERSAL = frozenset({"loss", "val_loss"})
_SUFFIX = re.compile(r"_(mean|min|max|median|heldout_\d+x|\d+x)$")


def _bases(name: str) -> set[str]:
    raw = str(name)
    stripped = _SUFFIX.sub("", raw)
    out = {raw, stripped}
    for candidate in list(out):
        out.add(candidate.removeprefix("val_"))
        out.add(f"val_{candidate.removeprefix('val_')}")
    return out


def unresolved_metric_names(names: list[str], registered, emitted: frozenset[str]) -> list[str]:
    """Names with no owner: not registered, not emitted, not universal."""
    known = set(emitted) | _UNIVERSAL
    out = []
    for name in names:
        bases = _bases(name)
        if any(registered(b) for b in bases) or bases & known:
            continue
        out.append(str(name))
    return out


def _declared_names(settings) -> list[str]:
    names: list[str] = []
    validation = getattr(settings, "validation", None)
    scoring = getattr(validation, "scoring", None)
    names.extend(str(m) for m in (getattr(scoring, "compute", None) or []))
    metrics = getattr(settings, "metrics", None)
    best = getattr(metrics, "best_metric_name", None)
    if best:
        names.append(str(best))
    early = getattr(settings, "early_stopping", None)
    monitor = getattr(early, "metric", None)
    if monitor:
        names.append(str(monitor))
    return names


def _emitted_for(settings) -> tuple[frozenset[str], str]:
    try:
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        cls = TrainingStrategyFactory().get_strategy_class(settings)
    except Exception:
        return frozenset(), "unresolved"
    caps = getattr(cls, "capabilities", None)
    return frozenset(getattr(caps, "emitted_metrics", frozenset()) or ()), cls.__name__


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T1,),
    subjects=(Subject.SETTINGS,),
    severity=Severity.ERROR,
    description="Every validation metric and selector name has an owner (registry or strategy)",
    fix_hint=(
        "Fix the name, register the metric with @register_metric, or declare it in the "
        "strategy's capabilities.emitted_metrics if its validation_step writes it."
    ),
)
def validation_metric_names_resolve(subject: WitnessSubject) -> WitnessVerdict:
    """Error (declared strategy) / advisory (undeclared) on an ownerless metric name."""
    from spectramr.core.metrics.registry import MetricsRegistry

    settings = subject.settings
    names = _declared_names(settings)
    if not names:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="no validation metric or selector declared",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    emitted, strategy_name = _emitted_for(settings)
    unresolved = unresolved_metric_names(names, MetricsRegistry.is_registered, emitted)
    if not unresolved:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message=f"all {len(names)} validation/selector name(s) resolve (strategy {strategy_name})",
            severity=Severity.ERROR,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    if emitted:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=False,
            message=(
                f"{len(unresolved)} name(s) are neither registered nor emitted by "
                f"{strategy_name}: {sorted(unresolved)}. A selector no path produces means the "
                "best checkpoint is never chosen; a compute entry no path produces is a silently "
                "missing column."
            ),
            severity=Severity.ERROR,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
            yaml_keys=(
                "validation.scoring.compute",
                "metrics.best_metric_name",
                "early_stopping.metric",
            ),
        )
    return WitnessVerdict(
        witness_name=_NAME,
        passed=True,
        message=(
            f"UNVERIFIED: {sorted(unresolved)} are not registered and {strategy_name} declares "
            "no emitted_metrics; if its validation_step writes them, declare them."
        ),
        severity=Severity.INFO,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T1,
    )
