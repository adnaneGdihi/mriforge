"""Tests for ``FrequencyLoss``.

Targets ``spectramr.models.losses.frequency_loss``. L1 loss between
magnitude spectra of magnitude images. Domain-agnostic (works on
either k-space or image-domain inputs).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.frequency_loss import FrequencyLoss


# ---------------------------------------------------------------------------
# Identity & residual scaling
# ---------------------------------------------------------------------------


def test_identity_yields_near_zero() -> None:
    """``loss(x, x)`` is near zero (eps from sqrt-stabilisation)."""
    loss_fn = FrequencyLoss()
    x = torch.randn(1, 1, 16, 16)
    out = loss_fn(x, x)
    assert out.item() < 1e-3


def test_loss_grows_with_residual() -> None:
    """Larger residual → larger spectral L1 loss."""
    loss_fn = FrequencyLoss()
    target = torch.zeros(1, 1, 16, 16)
    pred_small = torch.full_like(target, 0.1)
    pred_large = torch.full_like(target, 1.0)
    assert loss_fn(pred_small, target).item() < loss_fn(pred_large, target).item()


def test_complex_input_collapsed_to_magnitude() -> None:
    """Complex input is reduced to magnitude before FFT."""
    loss_fn = FrequencyLoss()
    cplx = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    out = loss_fn(cplx, cplx)
    assert out.item() < 1e-3


def test_two_channel_real_imag_collapsed_to_magnitude() -> None:
    """``[B, 2, H, W]`` real/imag is RSS-combined to magnitude."""
    loss_fn = FrequencyLoss()
    x = torch.randn(1, 2, 8, 8)
    out = loss_fn(x, x)
    assert out.item() < 1e-3


def test_gradient_flows() -> None:
    """Backward through the loss reaches the prediction tensor."""
    loss_fn = FrequencyLoss()
    pred = torch.randn(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 1, 8, 8)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8), (2, 1, 16, 16), (1, 1, 32, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_frequency_loss_shape_matrix(shape: tuple[int, ...]) -> None:
    """FrequencyLoss handles the parametrised shape matrix."""
    loss_fn = FrequencyLoss()
    pred = torch.randn(*shape)
    out = loss_fn(pred, pred.clone())
    assert out.dim() == 0
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_frequency_loss_registered_under_aliases() -> None:
    """``frequency`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "frequency" in list_available()
