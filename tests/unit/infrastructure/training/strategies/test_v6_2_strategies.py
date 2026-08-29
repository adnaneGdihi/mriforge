"""v6.2 strategy registration smoke tests.

Plan: TODO/DONE/IMPLEMENTATION_STATUS_v6_2.md.

Mirrors :mod:`test_v6_3_strategies` for the ten strategies that
shipped in v6.2 and were previously missing dedicated coverage:

- ``kspace_inr``
- ``bloch_consistent_denoising``
- ``girf_aware``
- ``inverse_bloch_phase``
- ``low_rank_sparse``
- ``qsm_pipeline``
- ``qspace_diffusion``
- ``adversarial_robustness``
- ``stochastic_interpolants``
- ``jepa``

Each strategy must:

1. Import as a Python module (no missing-import or syntax errors).
2. Resolve through :class:`TrainingStrategyFactory.STRATEGY_CLASS_PATHS`.
3. Round-trip via the factory's lazy class loader.
"""

from __future__ import annotations

import importlib

import pytest

from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory


V6_2_STRATEGIES: list[tuple[str, str, str]] = [
    ("kspace_inr",
     "mriforge.infrastructure.training.strategies.kspace_inr_strategy",
     "KSpaceINRStrategy"),
    ("bloch_consistent_denoising",
     "mriforge.infrastructure.training.strategies.bloch_consistent_denoising_strategy",
     "BlochConsistentDenoisingStrategy"),
    ("girf_aware",
     "mriforge.infrastructure.training.strategies.girf_aware_strategy",
     "GIRFAwareStrategy"),
    ("inverse_bloch_phase",
     "mriforge.infrastructure.training.strategies.inverse_bloch_phase_strategy",
     "InverseBlochPhaseStrategy"),
    ("low_rank_sparse",
     "mriforge.infrastructure.training.strategies.low_rank_sparse_strategy",
     "LowRankSparseStrategy"),
    ("qsm_pipeline",
     "mriforge.infrastructure.training.strategies.qsm_pipeline_strategy",
     "QSMPipelineStrategy"),
    ("qspace_diffusion",
     "mriforge.infrastructure.training.strategies.qspace_diffusion_strategy",
     "QSpaceDiffusionStrategy"),
    ("adversarial_robustness",
     "mriforge.infrastructure.training.strategies.adversarial_robustness_strategy",
     "AdversarialRobustnessStrategy"),
    ("stochastic_interpolants",
     "mriforge.infrastructure.training.strategies.stochastic_interpolants_strategy",
     "StochasticInterpolantsStrategy"),
    ("jepa",
     "mriforge.infrastructure.training.strategies.jepa_strategy",
     "JEPAStrategy"),
]


@pytest.mark.parametrize("alias,module_path,cls_name", V6_2_STRATEGIES)
def test_module_imports(alias: str, module_path: str, cls_name: str) -> None:
    """The strategy module imports cleanly and exposes the expected class."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, cls_name), (
        f"{module_path} does not expose {cls_name}; check the strategy file."
    )


@pytest.mark.parametrize("alias,module_path,cls_name", V6_2_STRATEGIES)
def test_strategy_factory_resolves(alias: str, module_path: str, cls_name: str) -> None:
    """The alias resolves through STRATEGY_CLASS_PATHS to the right dotted path."""
    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert alias in paths, (
        f"alias {alias!r} missing from STRATEGY_CLASS_PATHS — re-register or "
        f"remove the corresponding YAML."
    )
    expected = f"{module_path}.{cls_name}"
    assert paths[alias] == expected, (
        f"alias {alias!r} resolves to {paths[alias]!r}, expected {expected!r}"
    )


@pytest.mark.parametrize("alias,module_path,cls_name", V6_2_STRATEGIES)
def test_strategy_class_loads(alias: str, module_path: str, cls_name: str) -> None:
    """The factory's lazy class loader returns the actual class object."""
    factory = TrainingStrategyFactory()
    cls = factory._load_strategy_class(alias)
    assert cls.__name__ == cls_name
    assert cls.__module__ == module_path
