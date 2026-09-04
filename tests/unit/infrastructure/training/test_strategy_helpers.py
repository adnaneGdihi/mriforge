"""Unit tests for StrategyInitializationHelper.

Targets ``spectramr.infrastructure.training.strategy_helpers.StrategyInitializationHelper``.

Properties verified:
- ``initialize_loss_computer`` raises ``AttributeError`` when strategy has
  no ``env`` or ``state`` attribute (pitfall #9 — no silent fallback).
- ``initialize_loss_computer`` raises ``AttributeError`` when context
  lacks ``loss_function``.
- ``initialize_loss_computer`` sets ``strategy.loss_computer`` when context
  is valid.
- ``initialize_profiling_service`` sets ``_profiling_enabled`` attribute.
- ``create_profiling_context`` is a context manager that yields without error.
- ``initialize_data_consistency`` sets ``dc_layer = None`` when ``use_dc=False``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLossComputer:
    def __init__(self, context) -> None:
        self.context = context


class _ContextWithLossFn:
    loss_function = {"l1": torch.nn.L1Loss()}
    model_type = "gan"


class _ContextWithoutLossFn:
    pass


class _StrategyWithEnv:
    def __init__(self) -> None:
        self.env = _ContextWithLossFn()


class _StrategyWithState:
    def __init__(self) -> None:
        self.state = _ContextWithLossFn()


class _StrategyMissingContext:
    pass  # no env or state


class _StrategyBadContext:
    def __init__(self) -> None:
        self.env = _ContextWithoutLossFn()


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_import_helper() -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    assert StrategyInitializationHelper is not None


# ---------------------------------------------------------------------------
# initialize_loss_computer: raises when context is absent
# ---------------------------------------------------------------------------


def test_initialize_loss_computer_raises_no_env_or_state() -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    strategy = _StrategyMissingContext()
    with pytest.raises(AttributeError, match="'env' or 'state'"):
        StrategyInitializationHelper.initialize_loss_computer(strategy, _FakeLossComputer)


def test_initialize_loss_computer_raises_no_loss_function() -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    strategy = _StrategyBadContext()
    with pytest.raises(AttributeError, match="loss_function"):
        StrategyInitializationHelper.initialize_loss_computer(strategy, _FakeLossComputer)


# ---------------------------------------------------------------------------
# initialize_loss_computer: success path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_cls", [_StrategyWithEnv, _StrategyWithState])
def test_initialize_loss_computer_sets_attribute(strategy_cls) -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    strategy = strategy_cls()
    StrategyInitializationHelper.initialize_loss_computer(strategy, _FakeLossComputer)
    assert hasattr(strategy, "loss_computer")
    assert isinstance(strategy.loss_computer, _FakeLossComputer)


# ---------------------------------------------------------------------------
# initialize_profiling_service
# ---------------------------------------------------------------------------


def test_initialize_profiling_service_sets_attribute() -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    strategy = _StrategyWithEnv()
    StrategyInitializationHelper.initialize_profiling_service(strategy)
    assert hasattr(strategy, "_profiling_enabled")
    assert isinstance(strategy._profiling_enabled, bool)


# ---------------------------------------------------------------------------
# create_profiling_context
# ---------------------------------------------------------------------------


def test_create_profiling_context_is_context_manager() -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    strategy = _StrategyWithEnv()
    strategy._profiling_enabled = False
    side_effect = []
    with StrategyInitializationHelper.create_profiling_context(strategy, "op"):
        side_effect.append("inside")
    assert side_effect == ["inside"]


# ---------------------------------------------------------------------------
# initialize_data_consistency
# ---------------------------------------------------------------------------


def test_initialize_data_consistency_false_sets_none() -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    strategy = _StrategyWithEnv()
    StrategyInitializationHelper.initialize_data_consistency(strategy, use_dc=False)
    assert strategy.dc_layer is None
    assert strategy.use_dc is False


def test_initialize_data_consistency_no_model_sets_none() -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    strategy = _StrategyWithEnv()
    # Strategy has no generator_model attribute → should gracefully set dc_layer=None
    StrategyInitializationHelper.initialize_data_consistency(strategy, use_dc=True)
    assert strategy.dc_layer is None


# ---------------------------------------------------------------------------
# Edge: initialize_metrics_adapter handles exception gracefully
# ---------------------------------------------------------------------------


def test_initialize_metrics_adapter_handles_error() -> None:
    from spectramr.infrastructure.training.strategy_helpers import (
        StrategyInitializationHelper,
    )

    class _BadAdapter:
        def __init__(self):
            raise RuntimeError("construction failed")

    strategy = _StrategyWithEnv()
    # Should NOT propagate the exception; sets metrics_adapter=None instead.
    StrategyInitializationHelper.initialize_metrics_adapter(strategy, _BadAdapter)
    assert strategy.metrics_adapter is None
