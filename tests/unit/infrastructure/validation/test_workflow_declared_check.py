"""Tests for ``ConfigHealthChecker.check_workflow_declared``.

The check is the "optional now, required later" audit seam. An ABSENT ``workflow:``
block is advisory (see #283): no experiment YAML on dev predates this feature with
one, so erroring would redden all 1,465 arms rather than enforce anything. A
declaration that IS present must be right, so a STUB regime or an unsupported
regime x task pair is a hard error.
"""

from __future__ import annotations

from types import SimpleNamespace

from mriforge.config.schemas.enums import Regime, Task
from mriforge.config.schemas.workflow import WorkflowConfigSchema
from mriforge.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


def _check(workflow) -> object:
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    return checker.check_workflow_declared(SimpleNamespace(workflow=workflow))


def test_missing_workflow_is_advisory_not_an_error() -> None:
    # Regression guard for the dev port: `absent => error` was authored against the
    # public branch, which has no experiments/ tree. On dev it would fail the audit
    # for every un-annotated arm. Ratchet to error once annotated (#283).
    r = _check(None)
    assert r.passed
    assert r.severity == "info"
    assert r.check_name == "workflow_declared"


def _an_unsupported_task(regime: Regime) -> Task:
    """A Task the regime genuinely does not support, DERIVED not hardcoded.

    The negative cases below used a task member that no profile supported --
    which is why it was removed from the enum on 2026-07-31. A hardcoded pairing
    is a latent trap either way: the day someone adds it to the profile, every
    "unsupported task errors" case starts asserting on a supported one and
    passes for the wrong reason, silently.
    """
    from mriforge.domain.workflows import WORKFLOW_PROFILES

    supported = WORKFLOW_PROFILES[regime].supported_tasks
    for task in Task:
        if task not in supported:
            return task
    raise AssertionError(
        f"{regime.value} now supports every Task, so these tests have no "
        "negative case left. Pick a different regime."
    )


def test_declared_workflow_still_hard_fails_when_wrong() -> None:
    # The advisory downgrade must not weaken the guards that carry real signal.
    stub = _check(WorkflowConfigSchema(regime=Regime.CT))
    bad_task = _check(
        WorkflowConfigSchema(
            regime=Regime.STRUCTURAL, task=_an_unsupported_task(Regime.STRUCTURAL)
        )
    )
    assert not stub.passed and stub.severity == "error"
    assert not bad_task.passed and bad_task.severity == "error"


def test_live_regime_passes() -> None:
    r = _check(WorkflowConfigSchema(regime=Regime.STRUCTURAL, task=Task.RECONSTRUCTION))
    assert r.passed
    assert r.severity == "info"


def test_eval_only_regime_passes() -> None:
    # EVAL_ONLY is a legitimate declaration; train-time gating happens elsewhere.
    r = _check(WorkflowConfigSchema(regime=Regime.PERFUSION, task=Task.PARAMETER_MAPPING))
    assert r.passed


def test_stub_regime_errors() -> None:
    r = _check(WorkflowConfigSchema(regime=Regime.CT))
    assert not r.passed
    assert "STUB" in r.message


def test_unsupported_task_errors() -> None:
    r = _check(
        WorkflowConfigSchema(
            regime=Regime.STRUCTURAL, task=_an_unsupported_task(Regime.STRUCTURAL)
        )
    )
    assert not r.passed
    assert "not supported" in r.message


def test_check_is_wired_into_run_all_checks() -> None:
    assert hasattr(ConfigHealthChecker, "check_workflow_declared")
    import inspect

    src = inspect.getsource(ConfigHealthChecker.run_all_checks)
    assert "check_workflow_declared" in src


def test_fix_hint_uses_the_key_the_loader_actually_accepts() -> None:
    """The advice must not be a Tier-0 failure.

    ``workflow.name`` was renamed to ``workflow.regime`` on 2026-07-31 with
    posture="raise" -- one of only 7 raise-posture records out of 168, so
    ``WorkflowConfigSchema(name=...)`` is a hard ValidationError rather than a
    fold. This hint is emitted on every arm that has no ``workflow:`` block,
    which is 1,348 of 1,497 in the corpus, and precisely during the standing
    migration that CLAUDE.md mandates. Teaching the retired spelling sends every
    one of those authors into a schema rejection.
    """
    import pytest
    from pydantic import ValidationError

    hint = _check(None).fix_hint or ""
    assert "regime:" in hint
    assert "name:" not in hint, (
        "the hint teaches `workflow.name`, which the schema now rejects outright"
    )

    # And the hint's own spelling must survive the loader it is advising about.
    with pytest.raises(ValidationError):
        WorkflowConfigSchema(name="mri_structural")
    assert WorkflowConfigSchema(regime="mri_structural").regime is Regime.STRUCTURAL
