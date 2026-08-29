"""Tests for ``GenerativeTrainingStrategy`` (Phase-3 maximum-likelihood loop).

The strategy is model-agnostic: it computes a likelihood in priority order
(``log_prob`` -> ``training_loss`` -> ``(z, log_det)`` -> reconstruction
fallback). We exercise each path by calling ``_compute_losses_impl`` directly
with a fake model, bypassing the heavy ``BaseTrainingStrategy.__init__`` via
``__new__`` (the method only touches ``kwargs['model']`` here).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.strategies.generative import (  # noqa: E402
    GenerativeTrainingStrategy,
)


def _strategy() -> GenerativeTrainingStrategy:
    # Skip __init__ (needs a full TrainingEnvironment); we only call the
    # pure loss method with an explicit model.
    return GenerativeTrainingStrategy.__new__(GenerativeTrainingStrategy)


class _LogProbModel(torch.nn.Module):
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), -5.0)


class _TrainingLossModel(torch.nn.Module):
    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).mean()


class _FlowModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        z = torch.zeros(x.shape[0], 4)
        log_det = torch.zeros(x.shape[0])
        return z, log_det


class _PlainModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


def test_log_prob_path() -> None:
    s = _strategy()
    x = torch.randn(3, 1, 8, 8)
    out = s._compute_losses_impl(x, x, epoch=0, model=_LogProbModel())
    assert "g_total_loss" in out
    assert torch.isclose(out["g_total_loss"], torch.tensor(5.0), atol=1e-5)


def test_training_loss_path() -> None:
    s = _strategy()
    x = torch.ones(2, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0, model=_TrainingLossModel())
    assert torch.isclose(out["g_total_loss"], torch.tensor(1.0), atol=1e-6)


def test_flow_z_logdet_path() -> None:
    s = _strategy()
    x = torch.randn(2, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0, model=_FlowModel())
    # z=0, log_det=0 -> log p = -0.5*d*log(2pi); NLL = +0.5*d*log(2pi)
    expected = 0.5 * 4 * math.log(2 * math.pi)
    assert torch.isclose(out["g_total_loss"], torch.tensor(expected), atol=1e-4)


def test_reconstruction_fallback_path() -> None:
    s = _strategy()
    x = torch.randn(2, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0, model=_PlainModel())
    # pred == x -> MSE 0
    assert torch.isclose(out["g_total_loss"], torch.tensor(0.0), atol=1e-6)
    assert "loss_recon" in out


def test_explicit_model_does_not_touch_env() -> None:
    """Passing ``model=`` must not evaluate ``self.generator_model`` (→ ``self.env``).

    Regression for the eager ``dict.get`` default: ``kwargs.get('model',
    self.generator_model)`` accessed ``self.env.generator`` even when the caller
    supplied ``model=``, crashing every explicit-model call on a bare strategy.
    The bare ``__new__`` instance here has no ``env`` attribute at all.
    """
    s = _strategy()
    assert not hasattr(s, "env")  # bare __new__ — no environment attached
    out = s._compute_losses_impl(
        torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4), epoch=0, model=_PlainModel()
    )
    assert "g_total_loss" in out


def test_standard_normal_nll_helper() -> None:
    z = torch.zeros(5, 16)
    log_det = torch.zeros(5)
    nll = GenerativeTrainingStrategy._standard_normal_nll(z, log_det)
    expected = 0.5 * 16 * math.log(2 * math.pi)
    assert torch.isclose(nll, torch.tensor(expected), atol=1e-4)
