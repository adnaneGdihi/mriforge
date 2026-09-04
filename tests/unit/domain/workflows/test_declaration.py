"""The workflow-declaration accessors, and the reason they exist.

Seven audit checks read the declared regime. Every one of them treats "absent"
as ``passed=True, severity="info"`` — the "optional now, required later"
polarity CLAUDE.md mandates. That is correct, and it is also a trap: a check
that cannot find the field does not go **red**, it goes **quiet**. Rename the
field, miss one call site, and that check reports green forever while asserting
nothing. This repo has already shipped that exact shape (``min_center_fraction``
dropped by ``extra="ignore"``, which made the #534 mask fix inert behind a wall
of passing audits).

So the attribute name is spelled once, in
:mod:`spectramr.domain.workflows.declaration`, and
:class:`TestEveryWorkflowCheckStillFires` asserts the property that a missed
site would break: **declaring a regime must change what a check answers.**
"Skipped" and "passed" are the same boolean, so comparing verdicts would not
catch it; comparing the whole result does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.config.schemas.enums import Regime, SignalDomain, Task
from spectramr.config.schemas.workflow import WorkflowConfigSchema
from spectramr.domain.workflows.declaration import (
    declared_regime,
    declared_signal_domain,
    declared_spatial_rank,
    declared_task,
    workflow_block,
)
from tests.utils.config_block_stub import block_stub


def _cfg(workflow):
    """A config shaped like the checkers' minimum surface."""
    return SimpleNamespace(
        workflow=workflow,
        data=SimpleNamespace(dataset_type="image"),
        model=SimpleNamespace(model_type="unet"),
        losses=None,
        # `validation.metrics` folded to `validation.scoring.compute`.
        validation=block_stub("validation", metrics=[]),
    )


class TestAccessors:
    def test_reads_the_declared_regime(self) -> None:
        wf = WorkflowConfigSchema(regime=Regime.STRUCTURAL, task=Task.RECONSTRUCTION)
        assert declared_regime(_cfg(wf)) is Regime.STRUCTURAL
        assert declared_task(_cfg(wf)) is Task.RECONSTRUCTION
        assert workflow_block(_cfg(wf)) is wf

    def test_an_absent_block_is_none_not_an_error(self) -> None:
        """``workflow:`` is optional on TrainingSettings, and these are called
        from checkers that must survive legacy and half-built configs."""
        assert declared_regime(_cfg(None)) is None
        assert declared_task(_cfg(None)) is None
        assert declared_regime(SimpleNamespace()) is None

    def test_a_declared_regime_without_a_task_is_none(self) -> None:
        assert declared_task(_cfg(WorkflowConfigSchema(regime=Regime.FLOW))) is None

    def test_reads_the_narrowing_axes(self) -> None:
        wf = WorkflowConfigSchema(
            regime=Regime.STRUCTURAL, signal_domain="kspace", spatial_rank=2
        )
        assert declared_signal_domain(_cfg(wf)) is SignalDomain.KSPACE
        assert declared_spatial_rank(_cfg(wf)) == 2

    def test_the_narrowing_axes_default_to_none(self) -> None:
        """Absent is advisory, wrong is a hard error — so an uncertain author
        omits, and omitting must not look like a claim."""
        wf = WorkflowConfigSchema(regime=Regime.STRUCTURAL)
        assert declared_signal_domain(_cfg(wf)) is None
        assert declared_spatial_rank(_cfg(wf)) is None

    @pytest.mark.parametrize(
        "attribute", ["regime", "task", "signal_domain", "spatial_rank"]
    )
    def test_each_attribute_name_is_spelled_exactly_once(self, attribute: str) -> None:
        """The point of the module. If a second literal appears, the next rename
        can half-land again — which is the failure this file exists to prevent.

        Parametrised over every axis rather than asserted on ``regime`` alone:
        a guard that covers one field silently stops covering the module the
        moment a field is added.
        """
        import inspect

        from spectramr.domain.workflows import declaration

        body = "".join(
            line
            for line in inspect.getsource(declaration).splitlines(keepends=True)
            if not line.lstrip().startswith("#")
        )
        assert body.count(f'"{attribute}"') == 1, (
            f"the {attribute!r} attribute name appears more than once in "
            "declaration.py; route the extra read through its accessor."
        )


#: Every checker method routed through :func:`declared_regime`. Kept explicit
#: rather than discovered, so *deleting* a call site is as visible as breaking
#: one — a discovered list would simply get shorter and stay green.
_CHECKS = [
    "check_workflow_declared",
    "check_workflow_required_axes",
    "check_workflow_spatial_rank",
    "check_workflow_signal_domain",
    "check_workflow_dataset_signal_domain",
    "check_workflow_component_regime",
    "check_knob_applicability",
]


def _describe(result) -> str:
    """Flatten a result (or list of them) to something comparable."""
    items = result if isinstance(result, list) else [result]
    return " | ".join(f"{r.passed}:{r.severity}:{r.message}" for r in items)


class TestEveryWorkflowCheckStillFires:
    """Each check must answer *differently* once a regime is declared.

    This is the assertion a silent-skip regression fails. It deliberately does
    not assert *what* each check concludes — that is each check's own test file
    — only that the declaration reached it at all.
    """

    @pytest.fixture
    def checker(self):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        return ConfigHealthChecker.__new__(ConfigHealthChecker)

    @pytest.mark.parametrize("method", _CHECKS)
    def test_declaring_a_regime_changes_the_answer(self, checker, method: str) -> None:
        undeclared = _describe(getattr(checker, method)(_cfg(None)))
        declared = _describe(
            getattr(checker, method)(
                _cfg(WorkflowConfigSchema(regime=Regime.SPECTROSCOPIC))
            )
        )
        assert declared != undeclared, (
            f"{method} gives the same answer with and without a declared "
            "regime, so it is not reading the declaration. A check that cannot "
            "find the field reports green, not red — which is why this is "
            "asserted rather than assumed."
        )

    @pytest.mark.parametrize("method", _CHECKS)
    def test_the_check_exists(self, checker, method: str) -> None:
        """A renamed or deleted check would otherwise fail above with a
        confusing AttributeError attributed to the accessor."""
        assert callable(getattr(checker, method, None))
