"""Imaging-regime x task workflow facts.

This package holds the *pure* facts of the workflow contract — the frozen
:class:`WorkflowProfile` per :class:`~mriforge.config.schemas.enums.Regime`. It
imports nothing from ``infrastructure`` so the ``domain`` layer never depends
leftward. The registry-walk that verifies these facts against the live
strategy/loss/metric/operator registries lives in
:mod:`mriforge.infrastructure.validation.workflow_ledger`.
"""

from mriforge.domain.workflows.declaration import (
    declared_regime,
    declared_task,
    workflow_block,
)
from mriforge.domain.workflows.enforcement import (
    enforce_pipeline_maturity,
    enforce_pipeline_maturity_for_config,
)
from mriforge.domain.workflows.profiles import (
    WORKFLOW_PROFILES,
    WorkflowProfile,
    get_profile,
)

__all__ = [
    "WORKFLOW_PROFILES",
    "WorkflowProfile",
    "declared_regime",
    "declared_task",
    "enforce_pipeline_maturity",
    "enforce_pipeline_maturity_for_config",
    "get_profile",
    "workflow_block",
]
