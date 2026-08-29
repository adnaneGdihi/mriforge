"""Unit tests for TrainingStrategyFactory dispatch and error-path behaviour.

Scope (TASK IV.E):
  - get_strategy_class raises (not returns None) for an unknown paradigm.
  - get_strategy_class returns a BaseTrainingStrategy subclass for known keys.
  - _load_strategy_class raises ValueError for an un-importable path.
  - STRATEGY_CLASS_PATHS registry contains expected core keys.
  - factory.create_strategy method signature accepts env + kwargs.

Coverage is intentionally *orthogonal* to tests/contracts/test_strategy_registry.py,
which parametrises over every key. Here we test the dispatch logic and error
paths, not the full registry membership.

Heavy instantiation (which needs a full TrainingEnvironment + GPU) is
marked @pytest.mark.slow and skipped in the default fast lane.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mriforge.domain.exceptions import ConfigurationError
from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_with_strategy_class(class_path: str) -> MagicMock:
    """Return a mock config that exposes config.training.strategy_class."""
    cfg = MagicMock()
    cfg.training.strategy_class = class_path
    return cfg


def _config_with_training_mode(mode: str) -> MagicMock:
    """Return a mock config that uses the training_mode path.

    Until 2026-07-19 this helper also had to ``del cfg.training.diffusion_type``
    and ``del cfg.training.timesteps``: a MagicMock auto-vivifies those as truthy,
    and ``_infer_from_schema`` -- which outranked training_mode -- would hijack
    every dispatch into DiffusionTrainingStrategy. That rung is gone, so the
    mock no longer has to be disarmed to test the rung below it.
    """
    cfg = MagicMock()
    cfg.training.strategy_class = None  # force fallback to training_mode
    cfg.training.training_mode = mode
    return cfg


def _config_with_no_strategy() -> MagicMock:
    """Return a mock config with no resolvable strategy information."""
    cfg = MagicMock()
    cfg.training.strategy_class = None
    cfg.training.training_mode = "__totally_unknown_xyz_9999__"
    return cfg


# ---------------------------------------------------------------------------
# 1. Known keys resolve to a BaseTrainingStrategy subclass
# ---------------------------------------------------------------------------

SAMPLE_KNOWN_KEYS = [
    "gan",
    "reconstruction",
    "diffusion",
    "vae",
]


@pytest.mark.unit
@pytest.mark.parametrize("mode", SAMPLE_KNOWN_KEYS)
def test_get_strategy_class_known_key_returns_base_subclass(mode: str) -> None:
    """get_strategy_class must return a BaseTrainingStrategy subclass for known modes.

    Checks only static import — does NOT instantiate the strategy (that needs
    a full TrainingEnvironment + GPU).
    """
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy

    factory = TrainingStrategyFactory()
    cfg = _config_with_training_mode(mode)

    strategy_cls = factory.get_strategy_class(cfg)

    assert isinstance(
        strategy_cls, type
    ), f"Expected a class for mode='{mode}', got {type(strategy_cls)}"
    assert issubclass(strategy_cls, BaseTrainingStrategy), (
        f"Strategy class for mode='{mode}' must be a subclass of BaseTrainingStrategy, "
        f"got {strategy_cls}"
    )


@pytest.mark.unit
def test_get_strategy_class_explicit_class_path_takes_priority() -> None:
    """Explicit training.strategy_class overrides training_mode."""
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy

    factory = TrainingStrategyFactory()
    # Use explicit full class path — reconstruction is a reliable concrete class
    full_path = (
        "mriforge.infrastructure.training.strategies.reconstruction"
        ".ReconstructionTrainingStrategy"
    )
    cfg = _config_with_strategy_class(full_path)

    strategy_cls = factory.get_strategy_class(cfg)

    assert issubclass(strategy_cls, BaseTrainingStrategy)


# ---------------------------------------------------------------------------
# 2. Unknown paradigm raises — no silent fallback (CLAUDE.md pitfall #9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_strategy_class_unknown_paradigm_raises() -> None:
    """get_strategy_class must raise (not return None) for an unrecognised key.

    CLAUDE.md pitfall #9: 'If a model receives an attention_type it doesn't know,
    raise instead of falling back.'  The factory obeys the same rule — an unknown
    strategy name must produce an explicit, loud error.
    """
    from mriforge.domain.exceptions import ConfigurationError

    factory = TrainingStrategyFactory()
    cfg = _config_with_no_strategy()

    with pytest.raises((ConfigurationError, ValueError, RuntimeError)):
        factory.get_strategy_class(cfg)


@pytest.mark.unit
def test_unresolvable_training_mode_is_named_in_the_error() -> None:
    """The old message said "No strategy specified" even when one plainly was.

    A user who typo'd training_mode was told to add the field they had already
    filled in.
    """
    from mriforge.domain.exceptions import ConfigurationError

    factory = TrainingStrategyFactory()

    with pytest.raises(ConfigurationError, match="__totally_unknown_xyz_9999__"):
        factory.get_strategy_class(_config_with_no_strategy())


@pytest.mark.unit
def test_declared_training_mode_beats_diffusion_lookalike_fields() -> None:
    """Regression (pitfall #9): inference must never override a declaration.

    ``_infer_from_schema`` returned DiffusionTrainingStrategy whenever
    ``training.timesteps`` or ``training.diffusion_type`` was truthy, and it ran
    *before* the training_mode lookup -- so an explicit ``training_mode: gan``
    resolved to diffusion. Unreachable in production (``reject_flat_keys`` bars
    both keys from the schema), but it silently inverted the priority order for
    every duck-typed caller. The rung is now gone.
    """
    from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy

    factory = TrainingStrategyFactory()
    cfg = _config_with_training_mode("gan")
    cfg.training.timesteps = 1000  # the old hijack trigger
    cfg.training.diffusion_type = "ddpm"

    assert factory.get_strategy_class(cfg) is GANTrainingStrategy


@pytest.mark.unit
def test_get_strategy_class_none_config_raises() -> None:
    """get_strategy_class must raise if config.training is falsy/None."""
    factory = TrainingStrategyFactory()

    cfg = MagicMock()
    cfg.training = None  # no training section at all

    # v6.0 dispatch raises the domain-specific ConfigurationError (a MRIForgeError,
    # NOT a builtin) when no strategy can be resolved — see
    # strategy_factory.get_strategy_class final raise.
    with pytest.raises(ConfigurationError):
        factory.get_strategy_class(cfg)


# ---------------------------------------------------------------------------
# 3. _load_strategy_class raises ValueError for un-importable path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_strategy_class_raises_on_bad_module() -> None:
    """_load_strategy_class raises ValueError when the module doesn't exist."""
    factory = TrainingStrategyFactory()

    with pytest.raises(ValueError, match="Failed to load strategy class"):
        factory._load_strategy_class("mriforge.nonexistent_module.ghost.GhostStrategy")


@pytest.mark.unit
def test_load_strategy_class_raises_on_missing_class() -> None:
    """_load_strategy_class raises ValueError when the class doesn't exist in the module."""
    factory = TrainingStrategyFactory()

    with pytest.raises(ValueError, match="Failed to load strategy class"):
        # Module exists, class does not
        factory._load_strategy_class(
            "mriforge.infrastructure.training.strategy_factory.NonExistentClass9999"
        )


# ---------------------------------------------------------------------------
# 4. STRATEGY_CLASS_PATHS registry sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strategy_class_paths_non_empty() -> None:
    """STRATEGY_CLASS_PATHS must be a non-empty dict."""
    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert isinstance(paths, dict)
    assert len(paths) > 0


@pytest.mark.unit
def test_strategy_class_paths_core_keys_present() -> None:
    """Core strategy keys must be in STRATEGY_CLASS_PATHS."""
    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    expected_keys = {"gan", "reconstruction", "diffusion", "vae"}
    missing = expected_keys - set(paths.keys())
    assert not missing, f"Core strategy keys missing from registry: {missing}"


@pytest.mark.unit
def test_strategy_class_paths_values_are_dotted_strings() -> None:
    """Every value in STRATEGY_CLASS_PATHS must be a dotted module.ClassName string."""
    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    for key, path in paths.items():
        assert isinstance(path, str), f"Key '{key}': path is not a string"
        assert (
            "." in path
        ), f"Key '{key}': path '{path}' is not a dotted module.ClassName form"
        module_part, class_name = path.rsplit(".", 1)
        assert module_part, f"Key '{key}': empty module part in '{path}'"
        assert class_name, f"Key '{key}': empty class name in '{path}'"
        assert class_name[
            0
        ].isupper(), f"Key '{key}': class name '{class_name}' should start with an uppercase letter"


# ---------------------------------------------------------------------------
# 5. create_strategy method signature accepts env + **kwargs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_strategy_signature() -> None:
    """create_strategy must accept (env, logging_service, **kwargs)."""
    import inspect

    factory = TrainingStrategyFactory()
    sig = inspect.signature(factory.create_strategy)
    param_names = list(sig.parameters.keys())

    assert "env" in param_names
    # **kwargs must be present (VAR_KEYWORD kind)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    assert has_var_keyword, "create_strategy must accept **kwargs"


@pytest.mark.unit
def test_create_strategy_with_mocked_class() -> None:
    """create_strategy delegates to the resolved class with env and kwargs."""

    class _FakeStrategy:
        def __init__(
            self, env: Any, logging_service: Any = None, **kwargs: Any
        ) -> None:
            self.env = env
            self.logging_service = logging_service
            self.extra = kwargs

    factory = TrainingStrategyFactory()
    mock_env = MagicMock()
    mock_env.config = MagicMock()
    mock_logging = MagicMock()

    with patch.object(factory, "get_strategy_class", return_value=_FakeStrategy):
        strategy = factory.create_strategy(
            env=mock_env,
            logging_service=mock_logging,
            custom_kwarg="value",
        )

    assert strategy.env is mock_env
    assert strategy.logging_service is mock_logging
    assert strategy.extra.get("custom_kwarg") == "value"


# ---------------------------------------------------------------------------
# 6. Dispatch via training_mode legacy path (integration-light)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_training_mode_dispatch_reconstruction() -> None:
    """Legacy training_mode='reconstruction' dispatches to ReconstructionTrainingStrategy."""
    from mriforge.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    factory = TrainingStrategyFactory()
    cfg = _config_with_training_mode("reconstruction")
    cls = factory.get_strategy_class(cfg)
    assert cls is ReconstructionTrainingStrategy


@pytest.mark.unit
def test_training_mode_dispatch_gan() -> None:
    """Legacy training_mode='gan' dispatches to GANTrainingStrategy."""
    from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy

    factory = TrainingStrategyFactory()
    cfg = _config_with_training_mode("gan")
    cls = factory.get_strategy_class(cfg)
    assert cls is GANTrainingStrategy


def test_multi_acquisition_strategy_registered() -> None:
    from mriforge.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS.get("multi_acquisition")
    assert path is not None
    assert path.endswith("ConcreteMultiAcquisitionStrategy")
