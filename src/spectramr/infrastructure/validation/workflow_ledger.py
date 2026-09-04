"""Registry-walk backing the workflow maturity ledger.

The frozen :class:`~spectramr.domain.workflows.profiles.WorkflowProfile` facts
*declare* how far the framework can honour each regime. This module *asserts*
those declarations against the live registries — so a regime cannot claim to be
``LIVE`` without a real, registered strategy tagged for it, and an ``EVAL_ONLY``
regime cannot claim metric coverage it does not have.

It lives in ``infrastructure`` (not ``domain``) on purpose: reading strategy
capabilities means importing strategy classes from ``infrastructure.training``,
which ``domain`` may not do (deps point inward only, CLAUDE.md #13). The
maturity-ledger test and the ``config_health_checker`` consume these helpers.

Two properties are load-bearing and were both violated before 2026-07-16:

* **Tags are read from the class's own ``__dict__``, never inherited.** The
  question this module answers is "did this strategy *explicitly opt into* the
  regime?", and inheritance cannot answer it. See :func:`_declared_workflows`.
* **A broken ``spectramr`` strategy import raises.** Swallowing it would silently
  drop the strategy from the walk and turn a real ``ImportError`` into a
  misleading "no strategy is tagged for this regime" (pitfall #9). Only a
  genuinely-absent *third-party* optional dependency is skipped, loudly.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator
from typing import Any

from spectramr.config.schemas.enums import Regime

logger = logging.getLogger(__name__)


def operator_registered(name: str | None) -> bool:
    """True if ``name`` is a key in the physics operator registry.

    ``None`` (no forward operator declared) is not "registered" and returns
    False — callers distinguish that case explicitly.
    """
    if name is None:
        return False
    from spectramr.infrastructure.physics.registry import list_operators

    return name in list_operators()


def signal_model_registered(name: str | None) -> bool:
    """True if ``name`` is a key in the signal-model registry.

    The nonlinear half of "declares a real forward model". A regime whose physics
    is a *linear measurement operator* names it in ``forward_operator``; a regime
    whose physics is a nonlinear ``parameters -> signal`` map with no adjoint
    (extended Tofts, ADC mono-exponential) names it in ``signal_model``. Both are
    real forward models; only the first has an adjoint, which is why they cannot
    share a registry (see
    :mod:`spectramr.infrastructure.physics.signal_models.registry`).

    ``None`` returns False, exactly like :func:`operator_registered`.
    """
    if name is None:
        return False
    from spectramr.infrastructure.physics.signal_models.registry import (
        list_signal_models,
    )

    return name in list_signal_models()


def forward_model_declared(profile: Any) -> bool:
    """True if ``profile`` declares a forward model that actually resolves.

    The LIVE rule's first clause. It accepts **either** a registered linear
    operator **or** a registered signal model, because those are the two honest
    shapes physics comes in here.

    Before this was widened the clause read ``forward_operator is not None``,
    which had two failure modes pointing in opposite directions. It was far too
    weak for most regimes — *every* MR profile declares ``fft2d`` because MR
    k-space is FFT-reconstructed, so the clause tested "is this MR?" rather than
    "is this regime's physics wired?". And it was impossible for
    ``mri_perfusion``, which has the richest regime-specific physics in the repo
    but whose tracer-kinetic map is nonlinear with no adjoint. The only way to
    satisfy it there was to register a fake operator — so the clause actively
    *rewarded* the facade it existed to prevent (pitfall #16).
    """
    return operator_registered(
        getattr(profile, "forward_operator", None)
    ) or signal_model_registered(getattr(profile, "signal_model", None))


def metrics_tagged(regime: Regime) -> bool:
    """True if any registered metric is tagged for ``regime``.

    Importing :mod:`spectramr.core.metrics` triggers the package's
    ``pkgutil.walk_packages`` auto-discovery, so every ``@register_metric``
    decorator has fired before we read the tag table.
    """
    import spectramr.core.metrics  # noqa: F401  (side-effect: populate registry)
    from spectramr.core.metrics.registry import MetricsRegistry

    for meta in MetricsRegistry._workflow_tags.values():
        workflows = meta.get("workflows")
        if workflows and regime in workflows:
            return True
    return False


def losses_tagged(regime: Regime) -> bool:
    """True if any registered loss is tagged for ``regime``."""
    import spectramr.models.losses  # noqa: F401  (side-effect: populate registry)
    from spectramr.models.losses.registry import LossRegistry

    for meta in LossRegistry._loss_domains.values():
        workflows = meta.get("workflows")
        if workflows and regime in workflows:
            return True
    return False


def _iter_strategy_classes() -> Iterator[type[Any]]:
    """Yield each strategy class named in ``STRATEGY_CLASS_PATHS``.

    Import failures are NOT swallowed (pitfall #9). A strategy that fails to
    import would vanish from the walk and silently downgrade a LIVE/PARTIAL
    claim into a misleading "no strategy is tagged for this regime" assertion,
    hiding the real ``ImportError``. Only a genuinely-absent *third-party*
    optional dependency is skipped, and then at WARNING — never DEBUG.
    """
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    seen: set[str] = set()
    for dotted in TrainingStrategyFactory.STRATEGY_CLASS_PATHS.values():
        if dotted in seen:
            continue
        seen.add(dotted)
        module_path, _, cls_name = dotted.rpartition(".")
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] == "spectramr":
                raise ImportError(
                    f"workflow_ledger: {dotted} names a spectramr module that "
                    f"does not import ({exc}). The ledger cannot verify a "
                    "regime's maturity claim against a broken strategy — fix "
                    "the import, do not skip it."
                ) from exc
            logger.warning(
                "workflow_ledger: skipping %s — optional third-party "
                "dependency %r is not installed. Any regime claim backed ONLY "
                "by this strategy is UNVERIFIED in this environment.",
                dotted,
                exc.name,
            )
            continue
        try:
            yield getattr(module, cls_name)
        except AttributeError as exc:
            raise AttributeError(
                f"workflow_ledger: STRATEGY_CLASS_PATHS names {dotted!r} but "
                f"{cls_name!r} does not exist in {module_path!r} — the factory "
                "registry points at a class that is not there."
            ) from exc


def _declared_workflows(cls: type[Any]) -> frozenset[Regime] | None:
    """Workflows ``cls`` *itself* declares — never an inherited tag.

    ``capabilities`` is a ClassVar, so a plain ``getattr`` walks the MRO and
    reports every subclass of a tagged parent as tagged too. Before this was
    fixed, 64 strategies reported ``workflows={mri_structural}`` while exactly
    one (``ReconstructionTrainingStrategy``) declared it — and the inherited
    claim was outright FALSE for ``QSpaceDiffusionStrategy`` (diffusion-
    weighted), ``LowRankSparseStrategy`` (dynamic) and
    ``OneShotMultiParameterStrategy`` (quantitative). The ledger asks "did this
    strategy explicitly opt into this regime?", so it reads the class's own
    ``__dict__``: tagging by inheritance is how allow-lists rot.

    Deliberately NOT applied to
    :meth:`TrainingStrategyFactory.get_strategy_capabilities`, which keeps MRO
    semantics — ``required_config_fields`` *should* inherit. Different question.
    """
    caps = cls.__dict__.get("capabilities")
    workflows = getattr(caps, "workflows", None)
    return workflows or None


def strategies_tagged(regime: Regime) -> bool:
    """True if any registered training strategy is tagged for ``regime``.

    Reads the ``capabilities: StrategyCapabilities`` ClassVar *declared on the
    class itself* (see :func:`_declared_workflows`) — an inherited tag does not
    count, because inheriting a parent's regime is not opting into it.
    """
    return any(
        regime in (workflows or ())
        for cls in _iter_strategy_classes()
        if (workflows := _declared_workflows(cls)) is not None
    )


__all__ = [
    "forward_model_declared",
    "losses_tagged",
    "metrics_tagged",
    "operator_registered",
    "signal_model_registered",
    "strategies_tagged",
]
