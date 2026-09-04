"""gradient_entropy — image-sharpness autofocus loss (Atkinson et al.).

The entropy of the normalised gradient-magnitude distribution: a sharp image
concentrates gradient energy at edges (sparse → low entropy); a blurred image
spreads it (higher entropy). So **lower = sharper** (the minimise convention).
For exp_vf_35 it is a no-grad VALIDATION diagnostic of the recon's sharpness
gain — NOT a θ-training objective (the NUFFT does not pass gradients to the
trajectory, so a through-NUFFT sharpness loss could never train θ; wiring it as
one would be a facade, pitfall #16).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.gradient_entropy_loss import GradientEntropyLoss


def test_returns_scalar_requiring_grad_and_backward():
    loss = GradientEntropyLoss()
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    out = loss(x, None)
    assert out.shape == torch.Size([])
    assert out.requires_grad
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_sharp_image_has_lower_gradient_entropy_than_blurry():
    loss = GradientEntropyLoss()
    # sharp: a single step edge → gradient concentrated on one column (low entropy)
    sharp = torch.zeros(1, 1, 16, 16)
    sharp[..., 8:] = 1.0
    # blurry: a smooth ramp → gradient spread uniformly (high entropy)
    blurry = torch.linspace(0, 1, 16).view(1, 1, 1, 16).expand(1, 1, 16, 16).contiguous()
    assert float(loss(sharp, None)) < float(loss(blurry, None))


def test_accepts_complex_input_via_magnitude():
    loss = GradientEntropyLoss()
    x = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    out = loss(x, None)
    assert out.shape == torch.Size([]) and torch.isfinite(out)


def test_registered_in_image_domain():
    from spectramr.models.losses.registry import create_loss

    inst = create_loss("gradient_entropy")
    assert isinstance(inst, GradientEntropyLoss)
