r"""The contrast/field-agnostic bundle arms must resolve to a real strategy.

``check_experiment_configs_load.py`` proves an arm *constructs*. It does not
prove the arm can *start*: strategy dispatch happens later, and a
``training_mode`` with no ``STRATEGY_CLASS_PATHS`` key raises
``ConfigurationError`` at that point instead.

That gap is exactly the defect this bundle was landed to fix — ``lcah_encoder``
shipped registered under ``training_mode="acq_hypernetwork"`` with no such
strategy key — so it is worth a test rather than a convention. The MCGI arm hits
the same edge from the other side: ``supervised`` is a paradigm bucket, not a
registered strategy, so that arm must name its ``strategy_class`` explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

ARMS = {
    "experiments/inprogress/acq_hypernetwork/lcah_recon_multifield.yaml": (
        "AcquisitionHypernetworkStrategy"
    ),
    "experiments/inprogress/dispersion_bloch_ae/dlbae_brain_5field.yaml": (
        "DispersionBlochAEStrategy"
    ),
    "experiments/inprogress/supervised/mcgi_seg_multifield.yaml": (
        "ReconstructionTrainingStrategy"
    ),
}


@pytest.mark.parametrize(("rel_path", "expected"), sorted(ARMS.items()))
def test_arm_resolves_to_expected_strategy(rel_path: str, expected: str) -> None:
    """Each bundle arm must dispatch to the strategy it was written for."""
    config = TrainingSettings.from_yaml(str(_REPO_ROOT / rel_path))
    resolved = TrainingStrategyFactory().get_strategy_class(config)
    assert resolved.__name__ == expected


@pytest.mark.parametrize("rel_path", sorted(ARMS))
def test_arm_model_type_is_registered(rel_path: str) -> None:
    """The declared model_type must resolve through the populated registry."""
    from mriforge.models.init_registry import populate_model_registry
    from mriforge.models.registry import get_model_class

    populate_model_registry()
    config = TrainingSettings.from_yaml(str(_REPO_ROOT / rel_path))
    assert get_model_class(config.model.model_type) is not None
