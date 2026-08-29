"""Regression: MAE k-space masking must handle the 4-D [B,C,H,W] batch.

2026-05-31 audit: the ``mask_domain == "kspace"`` path used
``FFTTransformer.fft2`` (which unpacks a 5-D volume) and raised ``ValueError``
on every step. The fix routes through the fft2c/ifft2c SSOT, which handle the
4-D batch. This test confirms the SSOT round-trip is identity-preserving on the
MAE shape and that the strategy module imports cleanly.
"""

from __future__ import annotations

import torch

import mriforge.infrastructure.training.strategies.pretraining  # noqa: F401  (import must resolve)
from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c


def test_kspace_roundtrip_handles_4d_mae_batch():
    x = torch.randn(2, 1, 8, 8)  # the 4-D shape FFTTransformer.fft2 crashed on
    xc = torch.complex(x, torch.zeros_like(x))
    recovered = ifft2c(fft2c(xc)).real  # all-pass round-trip (no mask)
    assert recovered.shape == x.shape
    assert torch.allclose(recovered, x, atol=1e-4)
