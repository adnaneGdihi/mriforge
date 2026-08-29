"""Tests for the StrategyCapabilities seam (cached-cascade WS-X).

The contract rides on the resolved strategy class as a ``ClassVar`` (so the ~25
reconstruction short-name aliases share one contract), retrieved via
``TrainingStrategyFactory.get_strategy_capabilities``.
"""

from __future__ import annotations

from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy
from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory
from mriforge.models.capabilities import StrategyCapabilities


def test_base_has_default_empty_contract() -> None:
    assert isinstance(BaseTrainingStrategy.capabilities, StrategyCapabilities)
    assert BaseTrainingStrategy.capabilities.supported_paradigms == frozenset()
    assert BaseTrainingStrategy.capabilities.required_config_fields == ()


def test_factory_reads_class_level_contract() -> None:
    factory = TrainingStrategyFactory()

    class _Annotated(BaseTrainingStrategy):  # type: ignore[misc]
        capabilities = StrategyCapabilities(
            supported_paradigms=frozenset({"reconstruction"}),
            required_config_fields=("training.diffusion.num_timesteps",),
        )

    caps = factory.get_strategy_capabilities(_Annotated)
    assert "reconstruction" in caps.supported_paradigms
    assert "training.diffusion.num_timesteps" in caps.required_config_fields


def test_factory_defaults_for_unannotated_class() -> None:
    factory = TrainingStrategyFactory()

    class _Plain:  # not a strategy, no capabilities attr
        pass

    assert factory.get_strategy_capabilities(_Plain) == StrategyCapabilities()


def test_factory_resolves_short_name() -> None:
    # 'gan' resolves through STRATEGY_CLASS_PATHS to a real class; it carries the
    # default (empty) contract until WS-7 populates it — but the call must work.
    factory = TrainingStrategyFactory()
    caps = factory.get_strategy_capabilities("gan")
    assert isinstance(caps, StrategyCapabilities)
