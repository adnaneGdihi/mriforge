"""The GAN computer must apply the adversarial term, and apply it exactly once.

Two defects lived here, both invisible to the smoke suite because they cancelled
into a plausible-looking run:

1. **The generator never received adversarial gradient.** ``adversarial`` is in
   ``LEGACY_WARMUP_LOSSES`` and ``warmup_iterations`` defaults to 1000, but every
   ``_get_loss_weight`` call in this computer passed ``epoch`` and dropped
   ``iteration`` -- so the warm-up gate saw iteration 0 forever and the weight
   resolved to 0.0 on step 1 and on step 100,000 alike. The discriminator trained
   against a generator optimising pure reconstruction: the exact L1 collapse
   ``losses.gan.enable_adversarial`` was added to prevent.

2. **Pre-weighted library terms were re-weighted.** ``gan_loss_library`` returns
   sub-terms already scaled by their lambdas plus its own ``*_total_loss``.
   Stacking that dict asked ``resolve_loss_weight`` for weights for names like
   ``d_loss_real`` / ``g_adv_loss`` that no schema declares -- it refused to
   invent one and raised, killing every GAN run at its first discriminator step.
   Had weights existed, the total would have counted every term twice.

These assert the mechanism directly rather than searching a metrics blob: the
run-level report aggregates only totals, so a component-level facade is not
observable from ``best_metrics``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


@pytest.fixture
def gan_computer():
    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer
    from spectramr.pipelines.fit import _resolve_fit_config

    config = _resolve_fit_config(None, paradigm="gan", epochs=1, max_iterations=2)
    return UnifiedGANLossComputer(config, torch.device("cpu")), config


def _pair(size: int = 64):
    return torch.randn(2, 1, size, size), torch.randn(2, 1, size, size)


def test_adversarial_term_is_absent_during_warmup(gan_computer):
    """Warm-up is correct behaviour and must be preserved, not fixed away."""
    computer, config = gan_computer
    assert config.losses.reconstruction.warmup_iterations > 0
    x, y = _pair()
    out = computer.compute_generator_loss(
        pred=nn.Conv2d(1, 1, 3, padding=1)(x),
        target=y,
        discriminator=nn.Conv2d(1, 1, 3, padding=1),
        epoch=0,
        iteration=0,
    )
    assert not any("adv" in k for k in out.components), (
        f"adversarial term fired inside the warm-up window: {sorted(out.components)}"
    )


def test_adversarial_term_fires_after_warmup(gan_computer):
    """The regression: past warm-up the generator MUST get an adversarial term.

    Before the fix this stayed reconstruction-only forever, because ``iteration``
    never reached the weight resolver.
    """
    computer, config = gan_computer
    past_warmup = config.losses.reconstruction.warmup_iterations + 1
    x, y = _pair()
    out = computer.compute_generator_loss(
        pred=nn.Conv2d(1, 1, 3, padding=1)(x),
        target=y,
        discriminator=nn.Conv2d(1, 1, 3, padding=1),
        epoch=0,
        iteration=past_warmup,
    )
    assert any("adv" in k for k in out.components), (
        "generator loss carries no adversarial component past warm-up — it is "
        f"training on reconstruction alone: {sorted(out.components)}"
    )


def test_discriminator_total_is_not_double_counted(gan_computer):
    """The D total must equal the library's own sum, not the sum of everything.

    ``components`` carries the pre-weighted sub-terms AND ``d_total_loss``; naively
    stacking all of them doubles the discriminator objective. Pinning the exact
    2x relationship makes that regression unmistakable.
    """
    computer, _ = gan_computer
    x, y = _pair()
    out = computer.compute_discriminator_loss(
        real=y,
        fake=nn.Conv2d(1, 1, 3, padding=1)(x),
        discriminator=nn.Conv2d(1, 1, 3, padding=1),
        epoch=0,
        iteration=0,
    )
    assert "d_total_loss" in out.components
    torch.testing.assert_close(out.total, out.components["d_total_loss"])

    naive = sum(out.components.values())
    assert not torch.allclose(out.total, naive), (
        "the total equals the naive sum of every component — the pre-computed "
        "d_total_loss is being added alongside the sub-terms it already contains"
    )


def test_discriminator_loss_does_not_raise_on_undeclared_subterm_weights(gan_computer):
    """``d_loss_real`` has no ``lambda_d_loss_real``; it must never be looked up.

    This is what made every GAN run die at its first discriminator step.
    """
    computer, _ = gan_computer
    x, y = _pair()
    out = computer.compute_discriminator_loss(
        real=y,
        fake=nn.Conv2d(1, 1, 3, padding=1)(x),
        discriminator=nn.Conv2d(1, 1, 3, padding=1),
        epoch=0,
        iteration=0,
    )
    assert torch.is_tensor(out.total)
