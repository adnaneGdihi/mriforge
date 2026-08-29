"""The two acquisition-design strategies must ANNOUNCE what they do.

``Task.ACQUISITION_DESIGN`` read as a task no regime supported, which looks like
an unimplemented enum member. It was not: PILOT learns a k-space trajectory
under slew/gradient constraints, BALD picks the next lines to acquire from a
posterior-disagreement score, ``acquisition:`` is a wired v6.1 config block, and
both strategies are registered and constructible.

The capability simply never surfaced, because both subclass
``ReconstructionTrainingStrategy`` and **inherited** its ``{RECONSTRUCTION}``
tag. Inheritance is the failure mode worth pinning: a subclass that adds a
capability silently keeps advertising only its parent's, so the registry-walk
that decides what a regime supports cannot see the difference between "does not
do X" and "does X without saying so".
"""

from __future__ import annotations

import pytest

from mriforge.config.schemas.enums import Regime, Task
from mriforge.domain.workflows import WORKFLOW_PROFILES
from mriforge.infrastructure.training.strategies.bald_acquisition_strategy import (
    BALDAcquisitionStrategy,
)
from mriforge.infrastructure.training.strategies.pilot_strategy import PILOTStrategy
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)

_STRATEGIES = [PILOTStrategy, BALDAcquisitionStrategy]


@pytest.mark.parametrize("strategy", _STRATEGIES, ids=lambda c: c.__name__)
class TestTheCapabilityIsDeclaredNotInherited:
    def test_declares_its_own_capabilities(self, strategy) -> None:
        """``__dict__``, not ``getattr`` — the whole defect was that ``getattr``
        found the parent's and looked fine."""
        assert "capabilities" in strategy.__dict__, (
            f"{strategy.__name__} inherits ReconstructionTrainingStrategy's "
            "capabilities, so its acquisition-design half is invisible to every "
            "registry walk."
        )

    def test_claims_acquisition_design(self, strategy) -> None:
        assert Task.ACQUISITION_DESIGN in strategy.capabilities.tasks

    def test_keeps_reconstruction(self, strategy) -> None:
        """Both are joint: they design an acquisition *and* reconstruct from it.
        Dropping RECONSTRUCTION would un-tag the half that already worked."""
        assert Task.RECONSTRUCTION in strategy.capabilities.tasks

    def test_the_parent_does_not_claim_acquisition_design(self, strategy) -> None:
        """Guards the assertions above against becoming vacuous: if the base
        class ever claimed the task, they would pass by inheritance again."""
        parent = ReconstructionTrainingStrategy.capabilities
        assert Task.ACQUISITION_DESIGN not in parent.tasks

    def test_its_regime_supports_the_task_it_claims(self, strategy) -> None:
        """The half that closes the loop. A strategy claiming a task its own
        regime's profile does not list is the same orphan one level down."""
        for regime in strategy.capabilities.workflows:
            profile = WORKFLOW_PROFILES[regime]
            assert Task.ACQUISITION_DESIGN in profile.supported_tasks, (
                f"{strategy.__name__} claims acquisition_design under "
                f"{regime.value}, whose profile does not support it."
            )


def test_structural_supports_acquisition_design() -> None:
    """Stated directly, so the wiring is greppable from the profile side too."""
    assert (
        Task.ACQUISITION_DESIGN in WORKFLOW_PROFILES[Regime.STRUCTURAL].supported_tasks
    )
