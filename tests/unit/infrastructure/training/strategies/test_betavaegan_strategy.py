"""Tests for ``BetaVAEGANStrategy`` — the β-TC aux-objective injection.

The strategy reuses the full GANTrainingStrategy generator/discriminator
loop and only adds the model's ``beta_tc_term`` to the generator loss. We
test that injection in isolation by stubbing the parent's
``_compute_losses_impl`` (so the heavy GAN machinery / TrainingEnvironment
isn't needed) and a fake generator that exposes ``beta_tc_term``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies import gan as gan_mod  # noqa: E402
from spectramr.infrastructure.training.strategies.betavaegan_strategy import (  # noqa: E402
    BetaVAEGANStrategy,
)


class _FakeGen(torch.nn.Module):
    def beta_tc_term(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(2.0, requires_grad=True)


class _PlainGen(torch.nn.Module):
    """No beta_tc_term -> strategy must degrade to a plain GAN."""


def _make_strategy(monkeypatch, generator) -> BetaVAEGANStrategy:
    s = BetaVAEGANStrategy.__new__(BetaVAEGANStrategy)
    # Stub the parent generator-loss so we test only our addition.
    monkeypatch.setattr(
        gan_mod.GANTrainingStrategy,
        "_compute_losses_impl",
        lambda self, i, t, e, **k: {"g_total_loss": torch.tensor(5.0), "loss_adv": torch.tensor(1.0)},
    )
    # generator_model is a read-only property returning self.env.generator.
    s.env = SimpleNamespace(generator=generator)
    s.config = SimpleNamespace(training=SimpleNamespace(gan=SimpleNamespace(beta_tc_weight=1.0)))
    return s


def test_beta_tc_added_to_generator_loss(monkeypatch) -> None:
    s = _make_strategy(monkeypatch, _FakeGen())
    x = torch.randn(2, 1, 16, 16)
    out = s._compute_losses_impl(x, x, epoch=0)
    assert "loss_beta_tc" in out
    assert torch.isclose(out["loss_beta_tc"], torch.tensor(2.0))
    # 5.0 (base) + 1.0*2.0 (beta_tc) = 7.0
    assert torch.isclose(out["g_total_loss"], torch.tensor(7.0))


def test_weight_scales_beta_tc(monkeypatch) -> None:
    s = _make_strategy(monkeypatch, _FakeGen())
    s.config.training.gan.beta_tc_weight = 0.5
    out = s._compute_losses_impl(torch.randn(2, 1, 16, 16), None, epoch=0)
    # 5.0 + 0.5*2.0 = 6.0
    assert torch.isclose(out["g_total_loss"], torch.tensor(6.0))


def test_degrades_gracefully_without_beta_tc_term(monkeypatch) -> None:
    s = _make_strategy(monkeypatch, _PlainGen())
    out = s._compute_losses_impl(torch.randn(2, 1, 16, 16), None, epoch=0)
    assert "loss_beta_tc" not in out
    assert torch.isclose(out["g_total_loss"], torch.tensor(5.0))
