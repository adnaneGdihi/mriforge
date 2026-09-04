"""Tier-0 tests for the ``workflow:`` schema and its vocabulary.

A typo in ``workflow.regime`` / ``workflow.task`` must fail at construction
(Pydantic ``ValidationError``) — that is the whole point of closed enums over a
free-form ``metadata`` dict.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.enums import Regime, Task
from spectramr.config.schemas.workflow import WorkflowConfigSchema
from spectramr.domain.workflows import WORKFLOW_PROFILES


def test_valid_regime_and_task() -> None:
    wf = WorkflowConfigSchema(regime="mri_structural", task="reconstruction")
    assert wf.regime is Regime.STRUCTURAL
    assert wf.task is Task.RECONSTRUCTION


def test_task_is_optional() -> None:
    wf = WorkflowConfigSchema(regime=Regime.FLOW)
    assert wf.task is None


def test_typo_regime_raises() -> None:
    with pytest.raises(ValidationError):
        WorkflowConfigSchema(regime="mri_strctural")


def test_typo_task_raises() -> None:
    with pytest.raises(ValidationError):
        WorkflowConfigSchema(regime="mri_structural", task="reconstuction")


def test_extra_key_forbidden() -> None:
    with pytest.raises(ValidationError):
        WorkflowConfigSchema(regime="mri_structural", modality="mr")


def test_frozen() -> None:
    wf = WorkflowConfigSchema(regime=Regime.STRUCTURAL)
    with pytest.raises(ValidationError):
        wf.regime = Regime.FLOW  # type: ignore[misc]


class TestTheRetiredSpelling:
    """``name`` was the regime field until 2026-07-31.

    ``extra="forbid"`` alone would reject it, but with "Extra inputs are not
    permitted" — a message that does not say the value is still wanted, merely
    under a different key. These pin that the shim speaks instead.
    """

    def test_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowConfigSchema(name="mri_structural")

    def test_the_error_names_the_replacement_and_the_fixer(self) -> None:
        with pytest.raises(ValidationError) as exc:
            WorkflowConfigSchema(name="mri_structural")
        msg = str(exc.value)
        assert "workflow.regime" in msg
        assert "migrate_config_keys.py" in msg
        assert "Extra inputs" not in msg, (
            "extra='forbid' answered first — the shim must be a mode='before' "
            "validator so it sees the key at all."
        )

    def test_both_spellings_raise_rather_than_one_silently_winning(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowConfigSchema(name="mri_structural", regime="mri_quantitative")


@pytest.mark.parametrize("regime", list(Regime))
def test_every_regime_constructs_and_has_a_profile(regime: Regime) -> None:
    wf = WorkflowConfigSchema(regime=regime)
    assert wf.regime is regime
    assert regime in WORKFLOW_PROFILES


@pytest.mark.parametrize("regime", list(Regime))
def test_supported_tasks_are_a_subset_of_the_task_enum(regime: Regime) -> None:
    profile = WORKFLOW_PROFILES[regime]
    assert profile.supported_tasks <= set(Task)


def test_every_task_is_supported_by_some_regime() -> None:
    """The other direction, and the one that was false.

    A ``Task`` member no profile supports is an advertised option that
    hard-errors on use (``check_workflow_declared`` rejects an unsupported
    task), so the enum promises something the framework cannot run — pitfall #9
    at the vocabulary layer. Two members were orphaned until 2026-07-31, and the
    two had opposite fixes:

    * ``acquisition_design`` was **wired**. PILOTStrategy and
      BALDAcquisitionStrategy really do design acquisitions, but both subclass
      ``ReconstructionTrainingStrategy`` and inherited its ``{RECONSTRUCTION}``
      tag, so the capability existed and never announced itself.
    * ``segmentation`` was **removed**. Nothing implemented it at all.

    Only the subset check existed before, which is why the gap survived: it
    proves no profile invents a task, never that every task has a home.
    """
    supported = {t for p in WORKFLOW_PROFILES.values() for t in p.supported_tasks}
    orphans = sorted(t.value for t in Task if t not in supported)
    assert not orphans, (
        f"{orphans} are Task members no regime supports. Declaring one is a hard "
        "audit error, so the enum advertises a task that cannot be run. Either "
        "add it to the profile of a regime that genuinely supports it, or drop "
        "the member."
    )
