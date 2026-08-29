"""Regression: BlochConstraint was an inert (detached) loss that crashed.

2026-06-11 infrastructure audit. Each term was accumulated via ``.item()`` (so
the "soft regularizer" carried no gradient — pitfall #16), then
``torch.nan_to_num`` was called on the resulting Python float, raising TypeError.
The forward now returns a differentiable tensor.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.optimization.constraints.bloch import BlochConstraint


def test_forward_returns_differentiable_tensor():
    c = BlochConstraint(t1_min=1e-3, t1_max=5000.0)
    t1 = torch.full((4,), -10.0, requires_grad=True)  # violates positivity
    t2 = torch.full((4,), 20.0, requires_grad=True)  # T2 > T1 violates ordering
    loss = c(t1, t2)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() > 0.0  # violations present
    loss.backward()
    assert t1.grad is not None and torch.any(t1.grad != 0)


def test_forward_zero_when_satisfied():
    c = BlochConstraint(t1_min=1e-3, t1_max=5000.0)
    t1 = torch.full((4,), 1000.0)
    t2 = torch.full((4,), 80.0)  # 0 < T2 < T1
    assert c(t1, t2).item() == 0.0
