r"""Tests for the PISCO parallel-imaging self-consistency loss (SOTA plan P3.3).

PISCO (Spieker et al. 2024) enforces — without a calibration region — that the
local linear k-space self-consistency operator (SPIRiT kernel) fit on disjoint
subsets of the acquired multi-coil k-space *agrees*: ``‖G_A − G_B‖²``. Genuine
parallel-imaging data (multi-coil image with smooth sensitivities) is
self-consistent ⇒ low loss; structureless k-space ⇒ high loss.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.fft_ops import fft2c
from mriforge.models.losses.pisco_self_consistency_loss import (
    PiscoSelfConsistencyLoss,
    pisco_self_consistency,
)
from mriforge.models.losses.registry import create_loss


def _pi_consistent_kspace(b=1, c=4, h=16, w=16):
    """Multi-coil k-space from a true image × smooth sensitivities (SPIRiT-consistent)."""
    torch.manual_seed(0)
    img = torch.randn(b, 1, h, w)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    coils = torch.stack(
        [torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2))
         for cx, cy in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)][:c]],
        dim=0,
    ).view(1, c, h, w)
    multicoil = (img * coils).to(torch.complex64)
    return fft2c(multicoil)


def test_loss_lower_for_pi_consistent_than_random():
    consistent = _pi_consistent_kspace()
    torch.manual_seed(1)
    random_k = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    lc = pisco_self_consistency(consistent)
    lr = pisco_self_consistency(random_k)
    assert lc < lr, f"consistent {float(lc):.4g} not < random {float(lr):.4g}"


def test_loss_scalar_nonneg_differentiable():
    k = _pi_consistent_kspace().requires_grad_(True)
    loss = pisco_self_consistency(k)
    assert loss.ndim == 0 and float(loss.detach()) >= 0.0 and loss.requires_grad
    loss.backward()
    assert k.grad is not None and torch.isfinite(k.grad).all()


def test_registered_loss_accepts_complex_and_interleaved():
    fn = create_loss("pisco_self_consistency")
    assert isinstance(fn, PiscoSelfConsistencyLoss)
    cpx = _pi_consistent_kspace()
    out_c = fn(prediction=cpx, target=torch.zeros_like(cpx))
    # real-interleaved [B, 2C, H, W]
    ri = torch.cat([cpx.real, cpx.imag], dim=1)
    out_r = fn(prediction=ri, target=torch.zeros_like(ri))
    assert out_c.ndim == 0 and out_r.ndim == 0
    assert torch.isfinite(out_c) and torch.isfinite(out_r)


def test_rejects_single_coil():
    """PISCO needs ≥ 2 coils (the cross-coil self-consistency is the signal)."""
    with pytest.raises(ValueError, match="coil"):
        pisco_self_consistency(torch.randn(1, 1, 16, 16, dtype=torch.complex64))
