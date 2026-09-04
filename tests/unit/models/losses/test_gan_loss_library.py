"""Tests for adversarial loss strategies.

Targets ``spectramr.models.losses.gan_loss_library``. Each adversarial loss
strategy implements ``compute_generator_loss`` + ``compute_discriminator_loss``.

Categories:

- Base class fail-loud (``NotImplementedError`` for both methods)
- Standard GAN: BCE on D outputs (real vs fake targets), label smoothing
- LSGAN: least-squares on logits
- WGAN: linear means, no sigmoid
- Hinge: max-margin formulation
- Registry: each strategy is registered under its canonical name
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.gan_loss_library import (
    AdversarialLossStrategy,
    HingeLoss,
    LSGANLoss,
    StandardGANLoss,
    WGANLoss,
)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


def test_base_compute_generator_loss_raises() -> None:
    """The base class is abstract — direct ``compute_generator_loss`` raises."""
    base = AdversarialLossStrategy()
    with pytest.raises(NotImplementedError, match="compute_generator_loss"):
        base.compute_generator_loss(torch.zeros(2))


def test_base_compute_discriminator_loss_raises() -> None:
    """The base class is abstract — direct ``compute_discriminator_loss`` raises."""
    base = AdversarialLossStrategy()
    with pytest.raises(NotImplementedError, match="compute_discriminator_loss"):
        base.compute_discriminator_loss(torch.zeros(2), torch.zeros(2))


def test_base_forward_returns_zero_in_validation_loop() -> None:
    """Base ``forward`` is a dummy returning a 0 scalar (validation use)."""
    base = AdversarialLossStrategy()
    out = base(torch.zeros(2))
    assert out.dim() == 0
    assert out.item() == 0.0


# ---------------------------------------------------------------------------
# StandardGANLoss (BCE)
# ---------------------------------------------------------------------------


def test_standard_gan_label_smoothing_default_zero() -> None:
    """Default label_smoothing = 0."""
    loss = StandardGANLoss()
    assert loss.label_smoothing == 0.0


def test_standard_gan_d_loss_returns_two_terms() -> None:
    """``compute_discriminator_loss`` returns ``(real_loss, fake_loss)`` tuple."""
    loss = StandardGANLoss()
    d_real = torch.randn(8, 1)
    d_fake = torch.randn(8, 1)
    real_loss, fake_loss = loss.compute_discriminator_loss(d_real, d_fake)
    assert real_loss.dim() == 0
    assert fake_loss.dim() == 0


def test_standard_gan_g_loss_lower_when_d_thinks_fake_is_real() -> None:
    """Generator loss ↓ as ``D(fake)`` becomes more positive (more 'real')."""
    loss = StandardGANLoss()
    d_thinks_fake = torch.full((4, 1), -10.0)  # D says strongly fake
    d_thinks_real = torch.full((4, 1), 10.0)  # D says strongly real

    g_loss_bad = loss.compute_generator_loss(d_thinks_fake).item()
    g_loss_good = loss.compute_generator_loss(d_thinks_real).item()
    assert g_loss_good < g_loss_bad


def test_standard_gan_label_smoothing_preserves_finite_loss() -> None:
    """With label smoothing, both targets are still in (0, 1) → finite loss."""
    loss = StandardGANLoss(label_smoothing=0.1)
    real = torch.randn(2, 1)
    fake = torch.randn(2, 1)
    rl, fl = loss.compute_discriminator_loss(real, fake)
    assert torch.isfinite(rl)
    assert torch.isfinite(fl)


# ---------------------------------------------------------------------------
# LSGAN
# ---------------------------------------------------------------------------


def test_lsgan_g_loss_at_target_is_zero() -> None:
    """``D(fake) = 1`` (the real target) → squared loss = 0."""
    loss = LSGANLoss()
    d_fake = torch.ones(4, 1)
    g_loss = loss.compute_generator_loss(d_fake)
    assert g_loss.item() == 0.0


def test_lsgan_g_loss_quadratic_in_distance() -> None:
    """Doubling the distance from target quadruples the squared loss."""
    loss = LSGANLoss()
    d_close = torch.full((4, 1), 0.5)  # |0.5 - 1| = 0.5
    d_far = torch.full((4, 1), 0.0)  # |0.0 - 1| = 1.0
    l_close = loss.compute_generator_loss(d_close).item()
    l_far = loss.compute_generator_loss(d_far).item()
    assert pytest.approx(l_far / l_close, rel=1e-5) == 4.0


# ---------------------------------------------------------------------------
# WGAN
# ---------------------------------------------------------------------------


def test_wgan_g_loss_is_negated_mean() -> None:
    """``WGAN G loss = -mean(D(fake))``."""
    loss = WGANLoss()
    d_fake = torch.tensor([1.0, 2.0, 3.0, 4.0])
    g_loss = loss.compute_generator_loss(d_fake)
    assert pytest.approx(g_loss.item()) == -2.5


def test_wgan_d_loss_returns_negated_means() -> None:
    """``WGAN D returns (-mean(real), mean(fake))`` tuple."""
    loss = WGANLoss()
    real = torch.tensor([1.0, 1.0, 1.0])
    fake = torch.tensor([2.0, 2.0, 2.0])
    rl, fl = loss.compute_discriminator_loss(real, fake)
    assert rl.item() == -1.0
    assert fl.item() == 2.0


# ---------------------------------------------------------------------------
# HingeLoss
# ---------------------------------------------------------------------------


def test_hinge_g_loss_is_negated_mean() -> None:
    """Hinge generator loss = ``-mean(D(fake))``."""
    loss = HingeLoss()
    d_fake = torch.tensor([3.0, 4.0, 5.0])
    out = loss.compute_generator_loss(d_fake)
    assert pytest.approx(out.item()) == -4.0


def test_hinge_g_loss_lower_for_higher_d_output() -> None:
    """Generator loss is decreasing in D(fake)."""
    loss = HingeLoss()
    out_low = loss.compute_generator_loss(torch.full((4,), 0.0))
    out_high = loss.compute_generator_loss(torch.full((4,), 5.0))
    assert out_high.item() < out_low.item()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_gan_strategies_registered() -> None:
    """All four strategies are in the loss registry."""
    from spectramr.models.losses.registry import list_available

    available = set(list_available())
    expected = {"gan_standard", "gan_lsgan", "gan_wgan", "gan_hinge"}
    assert expected <= available


def test_gradient_penalty_name_no_longer_aliases_r1() -> None:
    """``gradient_penalty`` must not silently resolve to R1 (issue #191).

    R1 penalises ``‖∇D(real)‖`` toward 0; WGAN-GP interpolates real↔fake and
    penalises ``‖∇D(x̂)‖`` toward 1. The name was an alias of
    ``r1_regularization`` and handed R1 to anyone asking for WGAN-GP. It is
    removed so the ambiguous name fails loud instead of returning the wrong
    regulariser.
    """
    from spectramr.models.losses.registry import (
        LossRegistry,
        create_loss,
        is_registered,
    )

    assert not is_registered("gradient_penalty")
    assert "gradient_penalty" not in LossRegistry._aliases
    with pytest.raises(ValueError):
        create_loss("gradient_penalty")
    # The canonical R1 name still resolves.
    assert type(create_loss("r1_regularization")).__name__ == "R1RegularizationLoss"


def test_r1_and_wgan_gp_are_numerically_distinct() -> None:
    """R1 and WGAN-GP give different values on the same batch (issue #191).

    Guards against silently re-adding the alias: if the two ever collapse to
    the same number, the "they are interchangeable" assumption behind the old
    alias would look defensible again.
    """
    import torch.nn as nn

    from spectramr.models.losses.gan_loss_library import gradient_penalty_loss
    from spectramr.models.losses.registry import create_loss

    torch.manual_seed(0)

    class _TinyD(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Conv2d(1, 1, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).flatten(1).mean(1)

    disc = _TinyD()
    real = torch.randn(2, 1, 8, 8)
    fake = torch.randn(2, 1, 8, 8)

    r1_val = create_loss("r1_regularization", weight=1.0)(disc, real.clone())
    gp_val = gradient_penalty_loss(disc, real.clone(), fake.clone())
    assert not torch.isclose(r1_val, gp_val)


# ---------------------------------------------------------------------------
# CompositeGANLoss: advertised lambda > 0 must build the real component or
# RAISE — never silently zero the weight / skip the term (pitfall #9/#16)
# ---------------------------------------------------------------------------


def _composite(**overrides):
    from spectramr.models.losses.gan_loss_library import CompositeGANLoss

    kwargs = {
        "adv_strategy": StandardGANLoss(),
        "perceptual_loss": None,
        "lambda_l1": 1.0,
        "lambda_perceptual": 0.0,
        "lambda_adv": 1.0,
        "lambda_feat_match": 0.0,
        "lambda_gp": 0.0,
    }
    kwargs.update(overrides)
    return CompositeGANLoss(**kwargs)


def test_composite_positive_ssim_lambdas_build_real_sublosses() -> None:
    """lambda_ssim/lambda_ms_ssim > 0 construct real, non-None sub-losses."""
    from spectramr.models.losses.ssim_loss import MSSSIMLoss, SSIMLoss

    loss = _composite(lambda_ssim=0.5, lambda_ms_ssim=0.3)
    assert isinstance(loss.ssim_loss, SSIMLoss)
    assert isinstance(loss.ms_ssim_loss, MSSSIMLoss)
    # weights preserved, not zeroed
    assert loss.lambda_ssim == 0.5
    assert loss.lambda_ms_ssim == 0.3


def test_composite_positive_lpips_lambda_builds_non_none(monkeypatch) -> None:
    """lambda_lpips > 0 wires a non-None LPIPS loss (stubbed: no weights DL)."""
    import spectramr.models.losses.lpips_loss as lpips_mod

    class _StubLPIPS(torch.nn.Module):
        def __init__(self, net: str = "alex", verbose: bool = False) -> None:
            super().__init__()

        def forward(self, pred, target):  # pragma: no cover - not exercised
            return torch.tensor(0.0)

    monkeypatch.setattr(lpips_mod, "LPIPSLoss", _StubLPIPS)
    loss = _composite(lambda_lpips=0.8)
    assert loss.lpips_loss is not None
    assert loss.lambda_lpips == 0.8


def test_composite_zero_lambdas_never_construct_components() -> None:
    """lambda == 0 keeps the component un-constructed (None)."""
    loss = _composite()
    assert loss.ssim_loss is None
    assert loss.ms_ssim_loss is None
    assert loss.lpips_loss is None


def test_composite_ssim_import_failure_raises(monkeypatch) -> None:
    """Simulated ssim_loss ImportError must raise, not zero lambda_ssim."""
    import sys

    monkeypatch.setitem(sys.modules, "spectramr.models.losses.ssim_loss", None)
    with pytest.raises(ImportError, match="lambda_ssim"):
        _composite(lambda_ssim=0.5)


def test_composite_ms_ssim_import_failure_raises(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "spectramr.models.losses.ssim_loss", None)
    with pytest.raises(ImportError, match="lambda_ms_ssim"):
        _composite(lambda_ms_ssim=0.3)


def test_composite_lpips_import_failure_raises_with_install_hint(
    monkeypatch,
) -> None:
    """Simulated lpips ImportError must raise with an install hint."""
    import sys

    monkeypatch.setitem(sys.modules, "spectramr.models.losses.lpips_loss", None)
    with pytest.raises(ImportError, match="pip install lpips"):
        _composite(lambda_lpips=0.8)
