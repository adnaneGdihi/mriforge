"""Shared test scaffolding for normalizing-flow models.

Factors out the invertibility round-trip assertion used by ``glow`` and
``equivariant_flow`` (and reusable by the existing ``normalizing_flow``).
Written but not executed here; they run on the cluster.
"""

from __future__ import annotations

import torch


def assert_roundtrip(model, x: torch.Tensor, *, atol: float = 1e-4) -> None:
    """Assert ``f^{-1}(f(x)) ~= x`` for a flow exposing forward/inverse.

    ``model.forward`` must return ``(z, log_det)`` and ``model.inverse``
    must accept ``(z, x.shape)`` and return the reconstruction.
    """
    z, _ = model(x)
    x_rec = model.inverse(z, x.shape)
    assert x_rec.shape == x.shape
    max_err = (x_rec - x).abs().max().item()
    assert max_err < atol, f"flow round-trip residual {max_err} >= {atol}"


def assert_log_prob_finite(model, x: torch.Tensor) -> None:
    """Assert ``model.log_prob(x)`` is a finite per-sample vector."""
    lp = model.log_prob(x)
    assert lp.shape == (x.shape[0],)
    assert torch.isfinite(lp).all(), "log_prob produced non-finite values"


def assert_gradient_flows(model, x: torch.Tensor) -> None:
    """Assert a scalar loss on ``log_prob`` produces non-zero gradients."""
    x = x.clone().requires_grad_(True)
    loss = -model.log_prob(x).mean()
    loss.backward()
    grad_total = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    assert grad_total > 0, "no gradient flowed through the flow"
