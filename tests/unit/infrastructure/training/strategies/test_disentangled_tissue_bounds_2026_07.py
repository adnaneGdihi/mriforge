"""Regression: DisentangledTrainingStrategy._compute_tissue_bounds_loss must be
differentiable.

The helper summed its ReLU penalties with a trailing ``.item()``, which detaches
the result into a gradient-free constant. It is currently unused, but if it were
ever wired into the training objective it would contribute exactly zero gradient
(a silent dead-loss trap). The ``.item()`` is removed so the penalty carries grad.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)


def _bare_strategy() -> DisentangledTrainingStrategy:
    strat = DisentangledTrainingStrategy.__new__(DisentangledTrainingStrategy)
    strat.device = torch.device("cpu")
    return strat


def test_tissue_bounds_loss_is_differentiable():
    strat = _bare_strategy()
    # t2 < t1 - 50 violates the first bound, producing a positive, grad-carrying
    # penalty (relu(t2 + 50 - t1) with t1=100, t2=10 -> relu(-40)=0; use a
    # violating pair instead).
    t1 = torch.tensor([100.0], requires_grad=True)
    t2 = torch.tensor([90.0], requires_grad=True)  # t2 + 50 - t1 = 40 > 0 -> active
    tp = {"t1": t1, "t2": t2}

    loss = strat._compute_tissue_bounds_loss(tp, None)
    assert loss.requires_grad, "tissue-bounds loss was detached (dead-loss trap)"
    assert loss.item() > 0
    loss.backward()
    assert t1.grad is not None and t2.grad is not None


def test_tissue_bounds_loss_zero_inputs_returns_zero_tensor():
    strat = _bare_strategy()
    out = strat._compute_tissue_bounds_loss(None, None)
    assert torch.is_tensor(out)
    assert out.item() == 0.0
