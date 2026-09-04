"""Regression: hinge and RaLSGAN must use their canonical ±1 margins.

* Hinge D loss (Lim & Ye / SAGAN / BigGAN) is
  ``E[relu(1 - D(real))] + E[relu(1 + D(fake))]``. The pre-fix fake term was
  ``relu(D(fake) + label_smoothing)`` → at the default smoothing 0 the ``+1``
  margin vanished, so D's fake loss saturated as soon as ``D(fake) < 0``
  instead of ``< -1``.
* RaLSGAN (Jolicoeur-Martineau) is
  ``E[(D(x) - E[D(x̂)] - 1)²] + E[(D(x̂) - E[D(x)] + 1)²]``. The pre-fix code
  substituted ``label_smoothing`` (0) for the relativistic ``+1``, pushing
  toward the mean instead of the ±1 margin — contradicting the class docstring.
"""

from __future__ import annotations

import torch

from spectramr.models.losses.gan_loss_library import HingeLoss, RALSGANLoss


def test_hinge_discriminator_fake_margin_is_one():
    loss = HingeLoss(label_smoothing=0.0)
    d_real = torch.tensor([2.0])
    d_fake = torch.tensor([0.5])
    real_loss, fake_loss = loss.compute_discriminator_loss(d_real, d_fake)

    # real: relu(1 - 2) = 0 ; fake: relu(1 + 0.5) = 1.5 (NOT relu(0.5) = 0.5)
    assert torch.allclose(real_loss, torch.tensor(0.0))
    assert torch.allclose(fake_loss, torch.tensor(1.5)), fake_loss


def test_ralsgan_discriminator_uses_relativistic_plus_one():
    loss = RALSGANLoss(label_smoothing=0.0)
    d_real = torch.tensor([2.0])
    d_fake = torch.tensor([0.5])
    loss_real, loss_fake = loss.compute_discriminator_loss(d_real, d_fake)

    # mean_real=2, mean_fake=0.5
    # loss_real = (2 - 0.5 - 1)^2 = 0.25
    # loss_fake = (0.5 - 2 + 1)^2 = 0.25  (NOT (0.5 - 2)^2 = 2.25)
    assert torch.allclose(loss_real, torch.tensor(0.25)), loss_real
    assert torch.allclose(loss_fake, torch.tensor(0.25)), loss_fake


def test_ralsgan_generator_uses_relativistic_plus_one():
    loss = RALSGANLoss(label_smoothing=0.0)
    d_real = torch.tensor([2.0])
    d_fake = torch.tensor([0.5])
    g_loss = loss.compute_generator_loss(d_fake, real_outputs_d=d_real)

    # loss_fake = (0.5 - 2 - 1)^2 = 6.25 ; loss_real = (2 - 0.5 + 1)^2 = 6.25
    # G = 0.5 * (6.25 + 6.25) = 6.25
    assert torch.allclose(g_loss, torch.tensor(6.25)), g_loss
