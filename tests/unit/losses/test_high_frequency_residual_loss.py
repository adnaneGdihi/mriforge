"""Tests for ``HighFrequencyResidualLoss``.

Locks ``TODO/backlog_multi_contrast_remaining.md`` Item 2: the
high-frequency residual loss isolates the HF component of the target
so a small residual head on top of a latent-diffusion decoder learns
only the missing detail, not a full copy of the target.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.high_frequency_residual_loss import (
    HighFrequencyResidualLoss,
    _gaussian_lowpass,
)


def test_identical_inputs_yield_zero_loss() -> None:
    """When ``pred == target`` the HF residuals are identical → loss = 0."""
    loss = HighFrequencyResidualLoss()
    x = torch.randn(2, 1, 32, 32)
    assert loss(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_constant_inputs_yield_zero_loss() -> None:
    """Flat images have zero HF content; residual loss is zero for both."""
    loss = HighFrequencyResidualLoss()
    a = torch.full((2, 1, 32, 32), 0.3)
    b = torch.full((2, 1, 32, 32), 0.7)
    # Both flat → HF components are both ~0 → loss approx 0.
    assert loss(a, b).item() == pytest.approx(0.0, abs=1e-5)


def test_lowpass_attenuates_high_frequency_content() -> None:
    """Gaussian lowpass smooths a checkerboard pattern significantly."""
    # High-frequency checkerboard.
    x = torch.zeros(1, 1, 16, 16)
    x[:, :, ::2, ::2] = 1.0
    x[:, :, 1::2, 1::2] = 1.0
    lp = _gaussian_lowpass(x, sigma=2.0)
    # Variance of the smoothed version is much smaller than the original.
    assert lp.var().item() < x.var().item() * 0.5


def test_loss_grows_when_predicted_lacks_high_frequencies() -> None:
    """Predicting a smoothed version of the target hurts more than predicting the target."""
    loss = HighFrequencyResidualLoss(sigma=2.0)
    torch.manual_seed(0)
    target = torch.randn(2, 1, 32, 32)
    pred_perfect = target
    # ``pred_smooth`` is the lowpass of the target — by construction it
    # has *zero* HF content, so the HF residual loss is large.
    pred_smooth = _gaussian_lowpass(target, sigma=2.0)
    l_perfect = loss(pred_perfect, target).item()
    l_smooth = loss(pred_smooth, target).item()
    assert l_smooth > l_perfect
    assert l_perfect == pytest.approx(0.0, abs=1e-7)


def test_loss_rejects_shape_mismatch() -> None:
    """Mismatched ``pred``/``target`` shapes raise ``ValueError``."""
    loss = HighFrequencyResidualLoss()
    a = torch.randn(2, 1, 32, 32)
    b = torch.randn(2, 1, 16, 16)
    with pytest.raises(ValueError, match="shape mismatch"):
        loss(a, b)


def test_constructor_rejects_invalid_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be > 0"):
        HighFrequencyResidualLoss(sigma=0.0)


def test_constructor_rejects_unknown_loss_type() -> None:
    with pytest.raises(ValueError, match="loss_type"):
        HighFrequencyResidualLoss(loss_type="bogus")


@pytest.mark.parametrize("loss_type", ["l1", "l2", "smooth_l1"])
def test_all_loss_types_return_finite_scalar(loss_type: str) -> None:
    loss = HighFrequencyResidualLoss(loss_type=loss_type)
    torch.manual_seed(0)
    a = torch.randn(2, 1, 16, 16)
    b = torch.randn(2, 1, 16, 16)
    val = loss(a, b)
    assert val.shape == ()
    assert torch.isfinite(val).item()


def test_loss_is_registered_in_registry() -> None:
    """The ``@register_loss`` decorator wired the class into the loss registry."""
    # Importing the package triggers the decorator chain. Look up the name.
    from spectramr.models.losses import list_available

    names = list_available()
    assert "high_frequency_residual" in names or "HighFrequencyResidualLoss" in names


def test_gradient_flows_through_loss() -> None:
    """Gradients propagate through the lowpass + L1 chain (training-friendly)."""
    loss = HighFrequencyResidualLoss()
    torch.manual_seed(0)
    pred = torch.randn(1, 1, 16, 16, requires_grad=True)
    target = torch.randn(1, 1, 16, 16)
    out = loss(pred, target)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
