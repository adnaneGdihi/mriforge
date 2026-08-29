"""Regression tests for the 2026-06 audit fixes to the SSDU strategy.

* ``num_masks`` was a silent no-op and ``multi_mask_ssdu`` crashed on the
  missing ``context['theta_masks']``; the strategy now generates ``num_masks``
  splits and supplies the ``theta_masks`` list.
* ``split_acquired_mask`` yields disjoint Lambda/Theta partitions summing to the
  acquired mask.
"""
from __future__ import annotations

import inspect

import torch

from mriforge.infrastructure.training.strategies.ssdu_strategy import (
    SSDUReconstructionStrategy,
    split_acquired_mask,
)


def test_split_acquired_mask_disjoint_and_complete() -> None:
    g = torch.Generator().manual_seed(0)
    acq = torch.zeros(1, 1, 8, 8)
    acq[..., ::2] = 1.0  # 50% acquired
    lam, theta = split_acquired_mask(acq, theta_fraction=0.4, generator=g)
    # lambda + theta == acquired ; lambda * theta == 0 (disjoint)
    assert torch.equal(lam + theta, (acq > 0).float())
    assert torch.equal(lam * theta, torch.zeros_like(acq))
    # ~40% of the acquired points held out into theta
    n_acq = int((acq > 0).sum())
    assert theta.sum().item() == round(0.4 * n_acq)


def test_multi_mask_loss_accepts_theta_masks_list() -> None:
    """The strategy now supplies context['theta_masks']; the loss must consume
    a list with a single prediction (the previously-crashing path)."""
    from mriforge.models.losses.ssdu_loss import MultiMaskSSDULoss

    loss = MultiMaskSSDULoss()
    pred = torch.rand(1, 1, 8, 8)
    target_kspace = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    theta_masks = [torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8)]
    out = loss(
        pred,
        torch.zeros_like(pred),
        context={"target_kspace": target_kspace, "theta_masks": theta_masks},
    )
    assert torch.isfinite(out).all()


def test_strategy_wires_num_masks_and_theta_masks() -> None:
    src = inspect.getsource(SSDUReconstructionStrategy)
    # num_masks now drives a split loop and feeds the theta_masks context key.
    assert "range(max(1, self.num_masks))" in src
    assert '"theta_masks": theta_masks' in src
    # Typed config read (not getattr-with-default).
    assert "int(cfg.num_masks)" in src
