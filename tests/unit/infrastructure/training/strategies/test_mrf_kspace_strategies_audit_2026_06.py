"""Regression test for the 2026-06 audit fix to ``SpatiotemporalMRFReconStrategy``.

The SFC-Beltrami regulariser (``mu_reg``/``nu_reg``) derived ``mu``/``nu`` from
the no-grad ``input_batch``, so it was a constant w.r.t. model parameters
(zero gradient) and ``lambda_mu``/``lambda_nu`` were inert. It now derives from
the model output ``recon`` so the regulariser has a gradient path.
"""
from __future__ import annotations

import inspect
import types

import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.mrf_kspace_strategies import (
    SpatiotemporalMRFReconStrategy,
)


class _Gen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,T,H,W] -> [B,P,H,W]
        return self.conv(x.mean(dim=2))


def test_beltrami_regulariser_carries_gradient_to_model() -> None:
    gen = _Gen()
    inp = torch.randn(1, 4, 2, 8, 8)
    # target == recon so the L1 fidelity term contributes ~0 gradient; any
    # gradient that survives must come from the Beltrami regulariser.
    with torch.no_grad():
        target = gen(inp).clone()

    s = object.__new__(SpatiotemporalMRFReconStrategy)
    s.lambda_mu = 1.0
    s.lambda_nu = 1.0
    s.env = types.SimpleNamespace(generator=gen)

    out = s._compute_losses_impl(inp, target, epoch=0)
    assert out["g_total_loss"].requires_grad

    gen.zero_grad()
    out["g_total_loss"].backward()
    gnorm = sum(
        p.grad.abs().sum().item() for p in gen.parameters() if p.grad is not None
    )
    # With the old input-derived mu/nu this would be ~0 (dead regulariser).
    assert gnorm > 0.0


def test_source_derives_from_recon_not_input() -> None:
    src = inspect.getsource(SpatiotemporalMRFReconStrategy._compute_losses_impl)
    assert "x = recon[:, 0]" in src
    assert "x = input_batch[:, 0]" not in src
