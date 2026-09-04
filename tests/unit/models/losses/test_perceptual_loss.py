"""``PerceptualLoss`` (``models/losses/perceptual_loss.py``): non-finite inputs are refused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture
def perceptual(monkeypatch):
    """A PerceptualLoss over a stub VGG (37 identity layers, no download)."""
    import torchvision

    stub = SimpleNamespace(features=nn.Sequential(*[nn.Identity() for _ in range(37)]))
    monkeypatch.setattr(torchvision.models, "vgg19", lambda **_: stub)
    from spectramr.models.losses.perceptual_loss import PerceptualLoss

    return PerceptualLoss()


def test_a_nan_prediction_raises_instead_of_zeroing_the_loss(perceptual) -> None:
    """Planted violation: until 2026-09-03 this returned a zero loss and the run went on."""
    x = torch.rand(1, 3, 8, 8)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite values in the prediction"):
        perceptual(x, torch.rand(1, 3, 8, 8))


def test_an_inf_target_raises(perceptual) -> None:
    y = torch.rand(1, 3, 8, 8)
    y[0, 1, 2, 3] = float("inf")
    with pytest.raises(ValueError, match="non-finite values in the target"):
        perceptual(torch.rand(1, 3, 8, 8), y)


def test_finite_inputs_give_a_finite_loss(perceptual) -> None:
    loss = perceptual(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
    assert torch.isfinite(loss).all()
