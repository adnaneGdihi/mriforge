"""Regression: the CRLB Fisher-trace surrogate must carry a gradient back to
the pulse (generator output).

2026-05-31 audit: ``_toy_fisher_trace`` took ``∂signal/∂θ`` with
``create_graph=False``, so the resulting Fisher trace (and thus ``-CRLB``) was
autograd-detached from ``pulse`` — the generator learned only the smoothness
penalty, never the CRLB objective. ``create_graph=True`` restores the path.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.training.strategies.mrf_acquisition_strategies import (
    CRLBMRFPulseDesignStrategy,
)


def test_fisher_trace_flows_gradient_to_pulse():
    pulse = torch.randn(2, 4, 2, requires_grad=True)  # [B, T, (alpha, TR)]
    params = (torch.rand(2, 2) + 0.5).requires_grad_(True)  # [B, (T1, T2)]
    fisher = CRLBMRFPulseDesignStrategy._toy_fisher_trace(pulse, params)
    # With create_graph=True the Fisher trace stays attached to `pulse`.
    assert fisher.requires_grad
    grad = torch.autograd.grad(fisher, pulse, allow_unused=True)[0]
    assert grad is not None and torch.any(grad != 0)
