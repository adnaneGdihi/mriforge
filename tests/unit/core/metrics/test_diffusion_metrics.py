"""Unit tests for core/metrics/diffusion_metrics.py.

Covers the CCSNR (cc_snr) no-silent-fallback contract (CLAUDE.md #9):
- __init__ raises ValueError on an unknown noise_method.
- forward(noise_method='difference', dwi_repeat=None) raises ValueError
  instead of quietly switching to the background estimator.
- forward(noise_method='background') returns a finite scalar.

Tensors are built with the EXACT dtypes/shapes the forward signature is
jaxtyping/beartype-annotated for: dwi float [B, C, H, W]; cc_mask /
background_mask integer [B, 1, H, W].
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.diffusion_metrics import CCSNR

pytestmark = pytest.mark.unit


def _make_dwi(batch: int = 1, channels: int = 1, h: int = 8, w: int = 8) -> torch.Tensor:
    """Deterministic positive-valued DWI tensor, float dtype, [B, C, H, W]."""
    torch.manual_seed(0)
    return torch.rand(batch, channels, h, w, dtype=torch.float32) + 0.5


def _make_mask(batch: int = 1, h: int = 8, w: int = 8) -> torch.Tensor:
    """Integer binary mask, [B, 1, H, W], with a non-empty foreground region."""
    mask = torch.zeros(batch, 1, h, w, dtype=torch.int64)
    mask[:, :, 2:5, 2:5] = 1
    return mask


# ---------------------------------------------------------------------------
# (1) Unknown noise_method must raise at construction (no silent fallback).
# ---------------------------------------------------------------------------


def test_ccsnr_unknown_noise_method_raises():
    """CCSNR(noise_method='bogus') raises ValueError at __init__."""
    with pytest.raises(ValueError):
        CCSNR(noise_method="bogus")


# ---------------------------------------------------------------------------
# (2) 'difference' with no repeat acquisition must raise (no fallback to bg).
# ---------------------------------------------------------------------------


def test_ccsnr_difference_without_repeat_raises():
    """noise_method='difference' with dwi_repeat=None raises ValueError."""
    metric = CCSNR(noise_method="difference")
    dwi = _make_dwi()
    cc_mask = _make_mask()
    with pytest.raises(ValueError):
        metric(dwi, cc_mask, dwi_repeat=None)


# ---------------------------------------------------------------------------
# (3) 'background' path returns a finite scalar.
# ---------------------------------------------------------------------------


def test_ccsnr_background_returns_finite_scalar():
    """noise_method='background' returns a finite, scalar SNR tensor."""
    metric = CCSNR(noise_method="background")
    dwi = _make_dwi()
    cc_mask = _make_mask()
    result = metric(dwi, cc_mask)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert torch.isfinite(result)
