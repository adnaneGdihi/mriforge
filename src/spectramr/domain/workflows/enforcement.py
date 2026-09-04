"""Runtime maturity enforcement — turn the declared ledger into behaviour.

The :class:`~spectramr.config.schemas.enums.Maturity` on a
:class:`~spectramr.domain.workflows.profiles.WorkflowProfile` is a *promise*
about what the framework can run. This module is where that promise becomes a
hard runtime gate: a ``STUB`` regime raises from every pipeline, and an
``EVAL_ONLY`` regime raises from ``train`` (but not ``evaluate``/``predict``).

Pure domain code: it reads profiles and raises the domain exception. Pipelines
(a higher layer) call it — never the reverse.
"""

from __future__ import annotations

from typing import Any, Literal

from spectramr.config.schemas.enums import Maturity, Regime
from spectramr.domain.exceptions import WorkflowNotImplementedError
from spectramr.domain.workflows.declaration import declared_regime
from spectramr.domain.workflows.profiles import WORKFLOW_PROFILES

PipelineKind = Literal["train", "evaluate", "predict"]


def enforce_pipeline_maturity(regime: Regime | None, pipeline: PipelineKind) -> None:
    """Raise :class:`WorkflowNotImplementedError` if ``regime`` cannot run ``pipeline``.

    Args:
        regime: The declared imaging regime, or ``None`` when the arm declares
            no ``workflow:`` block. ``None`` is a no-op here — a missing
            declaration is the *audit's* job (``check_workflow_declared``),
            not the runtime's.
        pipeline: Which pipeline is about to run.

    Behaviour by maturity:
        - ``LIVE`` / ``PARTIAL`` → always allowed.
        - ``EVAL_ONLY`` → ``evaluate`` / ``predict`` allowed; ``train`` raises.
        - ``STUB`` → every pipeline raises.
    """
    if regime is None:
        return
    profile = WORKFLOW_PROFILES.get(regime)
    if profile is None:  # pragma: no cover - enum ⇒ profile invariant
        return

    maturity = profile.maturity
    if maturity is Maturity.STUB:
        raise WorkflowNotImplementedError(
            f"Regime {regime.value!r} is a STUB: the framework has no forward "
            f"operator, losses or strategy for it, so the {pipeline!r} pipeline "
            "cannot run. Pick a LIVE/PARTIAL regime or implement the stub first."
        )
    if maturity is Maturity.EVAL_ONLY and pipeline == "train":
        raise WorkflowNotImplementedError(
            f"Regime {regime.value!r} is EVAL_ONLY: metrics exist but there are "
            "no losses or training strategy for it, so it cannot be trained. "
            "Run evaluate/predict instead, or implement the losses+strategy to "
            "promote it to LIVE."
        )


def enforce_pipeline_maturity_for_config(config: Any, pipeline: PipelineKind) -> None:
    """Convenience wrapper: read ``config.workflow.regime`` and enforce.

    Tolerant of configs that predate the ``workflow:`` block (``None`` regime →
    no-op), so it is safe to call unconditionally at every pipeline entry.
    """
    enforce_pipeline_maturity(declared_regime(config), pipeline)


__all__ = [
    "PipelineKind",
    "enforce_pipeline_maturity",
    "enforce_pipeline_maturity_for_config",
]
