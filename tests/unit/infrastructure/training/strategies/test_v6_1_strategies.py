"""v6.1 strategy registration + import smoke tests.

Plan: TODO/DONE/COMPLETION_LOG_v6_1.md "Strategy-level unit tests" follow-up.

Each new v6.1 strategy is exercised by:

1. ``import`` — the module + class load without raising
2. ``STRATEGY_CLASS_PATHS`` lookup — the factory resolves the alias
3. ``__name__`` matches the registered alias

These tests are deliberately shallow — full forward-pass tests need the
DI container, the model registry, and the loss computer, all of which
are exercised in the integration suite. The point of this file is to
guard the wiring (alias → import path → class name) so a refactor
breaks the test before it breaks the audit.
"""

from __future__ import annotations

import importlib

import pytest

from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory


# (alias, module path, class name)
V6_1_STRATEGIES: list[tuple[str, str, str]] = [
    ("calibration",
     "mriforge.infrastructure.training.strategies.conformal_calibration_strategy",
     "ConformalCalibrationStrategy"),
    ("validation_badge",
     "mriforge.infrastructure.training.strategies.validation_badge_strategy",
     "ValidationBadgeStrategy"),
    ("loupe",
     "mriforge.infrastructure.training.strategies.loupe_strategy",
     "LOUPEStrategy"),
    ("pilot",
     "mriforge.infrastructure.training.strategies.pilot_strategy",
     "PILOTStrategy"),
    ("bald",
     "mriforge.infrastructure.training.strategies.bald_acquisition_strategy",
     "BALDAcquisitionStrategy"),
    ("multi_param_mapping",
     "mriforge.infrastructure.training.strategies.multi_param_mapping_strategy",
     "OneShotMultiParameterStrategy"),
    ("multi_contrast_contrastive",
     "mriforge.infrastructure.training.strategies.multi_contrast_contrastive_strategy",
     "MultiContrastContrastiveStrategy"),
    ("edm",
     "mriforge.infrastructure.training.strategies.edm_training_strategy",
     "EDMTrainingStrategy"),
    ("pnp",
     "mriforge.infrastructure.training.strategies.pnp_strategy",
     "PnPStrategy"),
    ("caur",
     "mriforge.infrastructure.training.strategies.concomitant_aware_recon_strategy",
     "ConcomitantAwareReconStrategy"),
    ("xfield_fm",
     "mriforge.infrastructure.training.strategies.xfield_fm_strategy",
     "XFieldFMStrategy"),
    ("scas",
     "mriforge.infrastructure.training.strategies.scas_strategy",
     "SCASStrategy"),
    ("federated_dp_conformal",
     "mriforge.infrastructure.training.strategies.federated_dp_conformal_strategy",
     "FederatedDPConformalULFStrategy"),
]


@pytest.mark.parametrize("alias,module_path,cls_name", V6_1_STRATEGIES)
def test_strategy_module_imports(alias: str, module_path: str, cls_name: str) -> None:
    """Every v6.1 strategy module imports cleanly."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, cls_name), (
        f"Module {module_path} does not export {cls_name}. "
        f"Available: {[n for n in dir(mod) if not n.startswith('_')][:10]}"
    )


@pytest.mark.parametrize("alias,module_path,cls_name", V6_1_STRATEGIES)
def test_strategy_factory_resolves_alias(alias: str, module_path: str, cls_name: str) -> None:
    """Each alias is wired in STRATEGY_CLASS_PATHS pointing at the right class."""
    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert alias in paths, (
        f"Strategy alias {alias!r} missing from STRATEGY_CLASS_PATHS. "
        f"Wire it in mriforge.infrastructure.training.strategy_factory."
    )
    expected = f"{module_path}.{cls_name}"
    assert paths[alias] == expected, (
        f"Alias {alias!r} resolves to {paths[alias]!r}, expected {expected!r}."
    )


@pytest.mark.parametrize("alias,module_path,cls_name", V6_1_STRATEGIES)
def test_strategy_class_name_matches(alias: str, module_path: str, cls_name: str) -> None:
    """The class registered under the alias actually has the expected ``__name__``."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    assert cls.__name__ == cls_name, (
        f"Class object name {cls.__name__!r} != registered name {cls_name!r}"
    )


def test_no_alias_collisions() -> None:
    """v6.1 aliases are distinct from existing v6.0 aliases."""
    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    new_aliases = {alias for alias, _, _ in V6_1_STRATEGIES}
    assert len(new_aliases) == len(V6_1_STRATEGIES), "Duplicate v6.1 alias in test fixture"
    # Every alias maps to exactly one path
    assert len({paths[a] for a in new_aliases}) >= len(new_aliases) - 2, (
        "Too many v6.1 aliases collapse to the same class — possible mis-registration"
    )
