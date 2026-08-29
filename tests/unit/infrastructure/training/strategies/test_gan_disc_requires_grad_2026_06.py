"""Regression: the generator step must not deposit gradients into the
discriminator.

The G closure runs ``discriminator(fake)`` to score the generator. With the
discriminator's parameters left ``requires_grad=True``, the G backward computes
and accumulates "make D call fakes real" gradients into ``D.grad``. Under
gradient accumulation > 1 those leaked gradients survive to D's
``optimizer.step()`` and partially train D in the direction G wants — corrupting
the adversarial game (and wasting a full D backward every step otherwise).

The fix toggles ``discriminator.requires_grad_(False)`` for the G step and
``requires_grad_(True)`` for the D step. Gradients still flow to the *generator*
through D's activations (that does not require D's params to track grad).
"""

from __future__ import annotations

import types

import torch
from torch import nn

from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy
from mriforge.models.losses.computers.base import LossOutput


class _FakeLossComputer:
    def compute_generator_loss(
        self, pred, target, discriminator, epoch=0, iteration=0, **kw
    ):
        return LossOutput(total=discriminator(pred).mean(), components={}, metrics={})

    def compute_discriminator_loss(
        self, real, fake, discriminator, epoch=0, iteration=0, **kw
    ):
        total = discriminator(real).mean() + discriminator(fake).mean()
        return LossOutput(total=total, components={}, metrics={})


def _make_strategy(gen, disc):
    s = object.__new__(GANTrainingStrategy)
    s.env = types.SimpleNamespace(generator=gen, step=0)
    s.loss_computer = _FakeLossComputer()
    s.config = types.SimpleNamespace(
        training=types.SimpleNamespace(enforce_output_range=False)
    )
    s._last_step_metrics = {}
    s._compute_training_metrics = lambda **k: {}
    return s


def test_generator_step_does_not_leak_grad_into_discriminator():
    gen = nn.Conv2d(1, 1, 1)
    disc = nn.Conv2d(1, 1, 1)
    s = _make_strategy(gen, disc)
    x = torch.randn(1, 1, 4, 4)

    g_closure = s._train_generator_step(x, x, disc, epoch=0, iteration=0)
    g_loss = g_closure()
    g_loss.backward()

    assert any(p.grad is not None for p in gen.parameters()), (
        "generator must receive gradients on its own step"
    )
    assert all(p.grad is None for p in disc.parameters()), (
        "generator step leaked gradients into the discriminator (D.grad set) — "
        "corrupts D under gradient accumulation"
    )


def test_discriminator_step_keeps_discriminator_trainable():
    gen = nn.Conv2d(1, 1, 1)
    disc = nn.Conv2d(1, 1, 1)
    s = _make_strategy(gen, disc)
    x = torch.randn(1, 1, 4, 4)

    # Simulate a prior G step having frozen D, then run the D step.
    disc.requires_grad_(False)
    d_closure = s._train_discriminator_step(x, x, disc, epoch=0, iteration=0)
    d_loss = d_closure()
    d_loss.backward()

    assert all(p.requires_grad for p in disc.parameters()), (
        "discriminator step must re-enable grad on D's parameters"
    )
    assert any(p.grad is not None for p in disc.parameters()), (
        "discriminator step must produce gradients for D"
    )
