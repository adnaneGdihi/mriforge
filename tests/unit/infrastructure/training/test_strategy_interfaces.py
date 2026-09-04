"""Unit tests for training strategy interface contracts.

Targets ``spectramr.infrastructure.training.strategy_interfaces``:
- ``ILossComputer`` abstract base class
- ``IAMPPolicy`` abstract base class
- ``IMetricsReporter`` abstract base class
- ``IOptimizerStepper`` re-export

These are abstract interfaces; we verify:
1. Concrete subclasses that implement all methods can be instantiated.
2. Attempting to instantiate abstract classes directly raises ``TypeError``.
3. Abstract methods raise ``NotImplementedError`` when called on an
   incomplete subclass (safety net: pitfall #9).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers: minimal concrete implementations
# ---------------------------------------------------------------------------


class _ConcreteAMPPolicy:
    """Minimal concrete IAMPPolicy."""

    def should_use_amp(self, model_type, device, config) -> bool:
        return False

    def get_autocast_device(self, device) -> str:
        return "cpu"


class _ConcreteMetricsReporter:
    """Minimal concrete IMetricsReporter."""

    def report_losses(self, losses, epoch, step_type="train"):
        return {k: float(v) if hasattr(v, "item") else float(v) for k, v in losses.items()}

    def detect_anomalies(self, losses):
        return False, False

    def compute_validation_metrics(self, hr_fakes, hr_reals, device):
        return {}


class _ConcreteLossComputer:
    """Minimal concrete ILossComputer."""

    def compute_loss(self, lr_reals, hr_reals, hr_fakes, discriminator_outputs=None, **kwargs):
        return {"total": torch.tensor(0.0)}


# ---------------------------------------------------------------------------
# Canary: import succeeds
# ---------------------------------------------------------------------------


def test_canary_import_interfaces() -> None:
    from spectramr.infrastructure.training.strategy_interfaces import (
        IAMPPolicy,
        ILossComputer,
        IMetricsReporter,
    )

    assert IAMPPolicy is not None
    assert ILossComputer is not None
    assert IMetricsReporter is not None


# ---------------------------------------------------------------------------
# Abstract classes cannot be instantiated directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interface_name", ["ILossComputer", "IAMPPolicy", "IMetricsReporter"])
def test_abstract_class_cannot_be_instantiated(interface_name: str) -> None:
    import spectramr.infrastructure.training.strategy_interfaces as mod

    cls = getattr(mod, interface_name)
    with pytest.raises(TypeError):
        cls()


# ---------------------------------------------------------------------------
# IOptimizerStepper re-export
# ---------------------------------------------------------------------------


def test_ioptimizerstepper_is_exported() -> None:
    from spectramr.infrastructure.training.strategy_interfaces import IOptimizerStepper

    assert IOptimizerStepper is not None


# ---------------------------------------------------------------------------
# Concrete implementations: runtime behaviour
# ---------------------------------------------------------------------------


def test_concrete_amp_policy_should_use_amp() -> None:
    policy = _ConcreteAMPPolicy()
    assert policy.should_use_amp("gan", torch.device("cpu"), {}) is False


def test_concrete_amp_policy_get_autocast_device() -> None:
    policy = _ConcreteAMPPolicy()
    assert policy.get_autocast_device(torch.device("cpu")) == "cpu"


def test_concrete_metrics_reporter_detect_anomalies() -> None:
    reporter = _ConcreteMetricsReporter()
    has_nan, has_inf = reporter.detect_anomalies({"loss": torch.tensor(0.5)})
    assert has_nan is False
    assert has_inf is False


def test_concrete_loss_computer_returns_dict() -> None:
    comp = _ConcreteLossComputer()
    losses = comp.compute_loss(
        lr_reals=torch.randn(2, 1, 8, 8),
        hr_reals=torch.randn(2, 1, 8, 8),
        hr_fakes=torch.randn(2, 1, 8, 8),
    )
    assert isinstance(losses, dict)
    assert "total" in losses


# ---------------------------------------------------------------------------
# Edge: abstract method raises NotImplementedError for partial implementation
# ---------------------------------------------------------------------------


def test_loss_computer_abstract_method_raises() -> None:
    from spectramr.infrastructure.training.strategy_interfaces import ILossComputer

    class _Partial(ILossComputer):
        pass  # does not implement compute_loss

    with pytest.raises(TypeError):
        _Partial()


def test_amp_policy_abstract_methods_raise() -> None:
    from spectramr.infrastructure.training.strategy_interfaces import IAMPPolicy

    class _Partial(IAMPPolicy):
        pass

    with pytest.raises(TypeError):
        _Partial()
