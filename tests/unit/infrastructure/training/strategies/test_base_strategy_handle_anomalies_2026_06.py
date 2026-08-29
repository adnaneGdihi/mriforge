"""Regression test for SC-001: ``BaseTrainingStrategy._handle_anomalies`` must
return ``dict[str, float]`` — both at the signature level and at runtime.

The method routes through ``_convert_metrics_to_floats`` (a static mixin method
that reduces every tensor to a Python float), so the return-type annotation was
corrected from ``dict[str, torch.Tensor]`` to ``dict[str, float]``. A subclass
reading ``_last_step_metrics`` (set from this return value) and treating entries
as tensors would attempt tensor ops on plain floats — a latent misuse bug the
wrong annotation invited.

The method touches no construction state, so it is exercised on a bare instance
built via ``object.__new__`` with a duck-typed ``env`` (the existing
strategy-unit-test pattern, see ``test_base_strategy_owned_state_2026_06``).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy


class _Strat(BaseTrainingStrategy):
    """Concrete subclass so ``object.__new__`` is not blocked by the abstract
    method; the behaviour under test lives entirely on the base/mixins."""

    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - stub
        return {}


def _bare() -> _Strat:
    s = object.__new__(_Strat)
    s.env = SimpleNamespace(generator=None, discriminator=None, losses={})
    return s


def test_handle_anomalies_return_annotation_is_float() -> None:
    annot = inspect.signature(BaseTrainingStrategy._handle_anomalies).return_annotation
    assert annot == "dict[str, float]"


def test_handle_anomalies_returns_floats() -> None:
    strategy = _bare()
    result = strategy._handle_anomalies(
        {"g_total_loss": torch.tensor(1.5)}, {}, 0
    )
    assert all(isinstance(v, float) for v in result.values())
    assert result["g_total_loss"] == 1.5


def test_handle_anomalies_reduces_non_scalar_tensor_to_float() -> None:
    strategy = _bare()
    result = strategy._handle_anomalies(
        {"d_total_loss": torch.tensor([2.0, 4.0])}, {}, 0
    )
    # Non-scalar tensors are reduced via mean() on-device, then resolved to float.
    assert isinstance(result["d_total_loss"], float)
    assert result["d_total_loss"] == 3.0
