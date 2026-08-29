"""Guard: the dead training-orchestration / use-case cluster stays deleted.

2026-06-11 infrastructure cleanup. The whole orchestration web
(``MultiStageTrainingService``, ``ExecutionEngine``, ``TrainingOrchestrationService``,
``DataLoaderManager``, the application use-case factory + ``TrainingUseCase`` /
``TransferLearningUseCase`` and the ``*_use_case`` stubs, plus the broken
``train_gan`` adapter consumers) was confirmed unreachable from every entry point
and removed. This guard fails loudly if any of it is resurrected without wiring a
real consumer (the previous doc claimed it was removed while the files still
existed — see docs/services_reference.rst).
"""

from __future__ import annotations

import importlib

import pytest

DELETED_MODULES = [
    "mriforge.infrastructure.services.multi_stage_training_service",
    "mriforge.infrastructure.services.unrolled_training_advisor",
    "mriforge.infrastructure.services.training_orchestration",
    "mriforge.infrastructure.services.training_orchestration.service",
    "mriforge.infrastructure.services.training_orchestration.execution_engine",
    "mriforge.infrastructure.services.training_orchestration.data_loader_manager",
    "mriforge.application.use_cases.factory",
    "mriforge.application.use_cases.training_use_case",
    "mriforge.application.use_cases.transfer_learning_use_case",
    "mriforge.application.use_cases.multi_stage_training_use_case",
    "mriforge.application.use_cases.use_cases",
    "mriforge.application.use_cases.use_cases_3d",
    "mriforge.application.use_cases.requests",
    "mriforge.application.use_cases.inference_use_case",
    "mriforge.application.use_cases.export_model_use_case",
    "mriforge.models.tuning.single_stage_hyperparameter_tuning",
    "mriforge.models.tuning.staged_hyperparameter_tuning",
    "mriforge.models.stability.model_stabilization",
    "mriforge.domain.repositories",
    # Dead parallel loss-config schema with zero production consumers and stale
    # Literal names (vanilla / wgan-gp / kspace_l2) that diverged from the loss
    # registry — a latent trap if ever wired. The live path is LossBuilder /
    # create_loss. Deleted 2026-07 (F1).
    "mriforge.models.losses.schemas",
]

LIVE_MODULES = [
    "mriforge.application.use_cases.hpo_use_case",
    "mriforge.application.use_cases.nr_metric_validation_use_case",
    "mriforge.application.use_cases.mrf_dictless_matching_use_case",
    "mriforge.models.tuning",
    "mriforge.models.stability",
    "mriforge.models.stability.manager",
    "mriforge.infrastructure.services",
    "mriforge.domain.services",
]


@pytest.mark.parametrize("mod", DELETED_MODULES)
def test_dead_module_stays_deleted(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


@pytest.mark.parametrize("mod", LIVE_MODULES)
def test_live_module_still_imports(mod):
    assert importlib.import_module(mod) is not None


def test_orchestration_interfaces_gone():
    import mriforge.domain.services as ds
    import mriforge.infrastructure.services as infra

    assert not hasattr(ds, "ITrainingOrchestrationService")
    assert not hasattr(ds, "TrainingOrchestrationService")
    assert not hasattr(infra, "ITrainingOrchestrationService")
