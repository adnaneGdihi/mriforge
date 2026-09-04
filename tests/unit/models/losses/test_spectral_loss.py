"""Tests for ``LogSpectralLoss``.

Targets ``spectramr.models.losses.spectral_loss``. ``L1`` between log-magnitude
spectra; works on both k-space and image inputs (FFT applied unless
``skip_fft=True``).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.spectral_loss import LogSpectralLoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Defaults: eps=1.0, skip_fft=False."""
    loss = LogSpectralLoss()
    assert loss.eps == 1.0
    assert loss.skip_fft is False


def test_explicit_construction() -> None:
    """Custom eps + skip_fft persisted."""
    loss = LogSpectralLoss(eps=0.01, skip_fft=True)
    assert loss.eps == 0.01
    assert loss.skip_fft is True


# ---------------------------------------------------------------------------
# Identity yields zero
# ---------------------------------------------------------------------------


def test_identity_yields_zero_loss() -> None:
    """``loss(x, x) == 0`` (same FFT magnitude → zero L1)."""
    loss = LogSpectralLoss()
    x = torch.randn(1, 1, 16, 16)
    out = loss(x, x)
    assert out.item() == 0.0


def test_complex_identity_yields_zero() -> None:
    """Complex input identity → zero loss."""
    loss = LogSpectralLoss()
    x = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    out = loss(x, x)
    assert out.item() < 1e-6


# ---------------------------------------------------------------------------
# Skip FFT path
# ---------------------------------------------------------------------------


def test_skip_fft_treats_input_as_kspace() -> None:
    """``skip_fft=True`` skips FFT — assumes input is already k-space."""
    loss_skip = LogSpectralLoss(skip_fft=True)
    x = torch.complex(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    out = loss_skip(x, x)
    assert out.item() < 1e-6


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def test_loss_grows_with_residual_magnitude() -> None:
    """Larger residual magnitude → larger log-spectral loss."""
    loss = LogSpectralLoss()
    target = torch.zeros(1, 1, 16, 16)
    pred_small = torch.full_like(target, 0.1)
    pred_large = torch.full_like(target, 1.0)
    assert loss(pred_small, target).item() < loss(pred_large, target).item()


# ---------------------------------------------------------------------------
# Multi-channel real/imag → complex coercion
# ---------------------------------------------------------------------------


def test_two_channel_real_input_handled() -> None:
    """``[B, 2, H, W]`` real-stacked → coerced to complex internally."""
    loss = LogSpectralLoss()
    x = torch.randn(1, 2, 8, 8)
    out = loss(x, x)
    assert out.item() < 1e-6


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8), (2, 1, 16, 16), (1, 1, 16, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_shape_matrix(shape: tuple[int, ...]) -> None:
    """LogSpectralLoss handles parametrised shapes."""
    loss = LogSpectralLoss()
    pred = torch.randn(*shape)
    out = loss(pred, pred.clone())
    assert out.dim() == 0
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows() -> None:
    """Backward through the loss reaches the prediction tensor."""
    loss = LogSpectralLoss()
    pred = torch.randn(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 1, 8, 8)
    loss(pred, target).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_log_spectral_registered() -> None:
    """``log_spectral`` is in the registry."""
    from spectramr.models.losses.registry import list_available

    assert "log_spectral" in list_available()
