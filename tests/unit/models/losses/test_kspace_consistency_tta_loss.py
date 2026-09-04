"""Tests for the k-space data-consistency TTA loss.

Pins the centered-FFT convention: the loss pushes the image-domain prediction to
k-space with the centered SSOT DFT (``fft2c``), NOT a raw uncentered
``torch.fft.fft2`` (CLAUDE.md pitfall #2). The measured k-space ``y`` and the
sampling mask ``M`` are conventionally centered (DC in the middle); an uncentered
forward would mask the wrong k-space lines whenever an external centered
``y``/``M`` is supplied.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.kspace_consistency_tta_loss import KspaceConsistencyTTALoss


def test_loss_uses_centered_fft2c_forward() -> None:
    torch.manual_seed(0)
    pred = torch.randn(1, 1, 8, 8)
    target_k = fft2c(torch.randn(1, 1, 8, 8).to(torch.complex64))  # centered measured k

    out = KspaceConsistencyTTALoss(weight=1.0)(pred, target_k)

    expected = (fft2c(pred.to(torch.complex64)) - target_k).abs().mean()
    assert torch.allclose(out, expected, atol=1e-6)
    # And it must NOT equal the uncentered convention — proving the migration.
    uncentered = (
        torch.fft.fft2(pred.to(torch.complex64), dim=(-2, -1), norm="ortho") - target_k
    ).abs().mean()
    assert not torch.allclose(out, uncentered, atol=1e-4)


def test_loss_zero_when_prediction_matches_measurement() -> None:
    # If the recon's centered k-space equals the measured k-space, DC loss is ~0.
    torch.manual_seed(1)
    img = torch.randn(1, 1, 8, 8)
    target_k = fft2c(img.to(torch.complex64))
    out = KspaceConsistencyTTALoss(weight=1.0)(img, target_k)
    assert out.abs().item() < 1e-5


def test_pred_is_kspace_skips_internal_fft() -> None:
    # With pred_is_kspace=True the prediction is already k-space (no internal FFT).
    torch.manual_seed(2)
    pred_k = torch.randn(1, 1, 6, 6, dtype=torch.complex64)
    target_k = torch.randn(1, 1, 6, 6, dtype=torch.complex64)
    out = KspaceConsistencyTTALoss(weight=1.0, pred_is_kspace=True)(pred_k, target_k)
    expected = (pred_k - target_k).abs().mean()
    assert torch.allclose(out, expected, atol=1e-6)
