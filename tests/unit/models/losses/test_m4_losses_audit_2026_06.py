"""Regression test for the 2026-06 audit fix to ``PMAGlobalLoss`` (m4_losses).

The SSIM term sat behind a bare ``except Exception`` that replaced a failed SSIM
with a constant 1.0 (silently dropping the structural term — pitfall #10), and
recomputed ``targ_mag.max().item()`` (a host sync) once PER CHANNEL. The fallback
is removed and the data-range sync is hoisted out of the channel loop.
"""
from __future__ import annotations

import inspect

import torch

from spectramr.models.losses.m4_losses import PMAGlobalLoss


def test_no_silent_ssim_fallback_and_hoisted_data_range() -> None:
    src = inspect.getsource(PMAGlobalLoss.forward)
    # The bare-except SSIM fallback is gone.
    assert "except Exception:" not in src
    assert "torch.tensor(1.0, device=prediction.device)" not in src
    # data_range resolved ONCE (outside the per-channel loop), not per channel.
    assert "ssim_data_range = float(targ_mag.max())" in src
    assert "data_range=targ_mag.max().item()" not in src


def test_forward_runs_and_is_finite_multichannel() -> None:
    loss = PMAGlobalLoss()
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 16, 16)
    out = loss(pred, target)
    assert out.ndim == 0
    assert torch.isfinite(out).all()
