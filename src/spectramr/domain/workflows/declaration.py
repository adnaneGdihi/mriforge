"""Read what an arm *declared* in its ``workflow:`` block — one accessor per axis.

Eight call sites read the regime, every one of them spelled
``getattr(workflow, "name", None)``, and every one of them treats ``None`` as
"skip this check, ``passed=True``". That combination is a trap: rename the field
and miss a site, and the check does not go red — it goes **quiet**, reporting
``severity="info"`` forever. This repo has already shipped that exact failure
(``min_center_fraction`` dropped by ``extra="ignore"``, which made the #534 mask
fix inert while every audit stayed green).

So the attribute name is spelled **once**, here. A future rename is a one-line
change, and ``tests/unit/domain/workflows/test_declaration.py`` pins that each
accessor still finds a declared value rather than merely tolerating its absence.

Deliberately ``getattr``-based and ``None``-tolerant: ``workflow:`` is optional
on :class:`TrainingSettings` (absent is advisory, wrong is a hard error — the
polarity CLAUDE.md mandates), and these are called from checkers that must run
against half-built or legacy configs.
"""

from __future__ import annotations

from typing import Any

from spectramr.config.schemas.enums import Regime, SignalDomain, Task

#: Each declared axis, named ONCE, so a rename cannot half-land across the
#: checkers. ``tests/unit/domain/workflows/test_declaration.py`` asserts each
#: literal appears exactly once in this module.
_REGIME_ATTR = "regime"
_TASK_ATTR = "task"
_SIGNAL_DOMAIN_ATTR = "signal_domain"
_SPATIAL_RANK_ATTR = "spatial_rank"


def workflow_block(config: Any) -> Any | None:
    """The ``workflow:`` block, or ``None`` when the arm predates it."""
    return getattr(config, "workflow", None)


def declared_regime(config: Any) -> Regime | None:
    """The declared imaging regime, or ``None`` if the arm declares none."""
    return getattr(workflow_block(config), _REGIME_ATTR, None)


def declared_task(config: Any) -> Task | None:
    """The declared task, or ``None``. Optional even when a regime is declared."""
    return getattr(workflow_block(config), _TASK_ATTR, None)


def declared_signal_domain(config: Any) -> SignalDomain | None:
    """The domain this arm CONSUMES, out of the regime's several.

    A regime's ``signal_domains`` is a set, so the profile alone cannot say
    which one an arm is in. Declaring it narrows the set to one. Semantics match
    ``ModelCapabilities.input_domain`` — see the schema docstring for why *emits*
    would be the wrong reading.
    """
    return getattr(workflow_block(config), _SIGNAL_DOMAIN_ATTR, None)


def declared_spatial_rank(config: Any) -> int | None:
    """The declared spatial rank, or ``None`` to leave it inferred from the data."""
    return getattr(workflow_block(config), _SPATIAL_RANK_ATTR, None)


__all__ = [
    "declared_regime",
    "declared_signal_domain",
    "declared_spatial_rank",
    "declared_task",
    "workflow_block",
]
