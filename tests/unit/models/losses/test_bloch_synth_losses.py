"""Tests for bloch-synthesis losses (MICCAI MRIxFields2026, idea 2.1)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.bloch_synth_losses import (
    BlochSourceConsistencyLoss,
    DispersionPriorLoss,
)


def test_dispersion_prior_zero_inside_envelope() -> None:
    loss = DispersionPriorLoss(lo=0.3, hi=0.4)
    beta = torch.tensor([[0.30], [0.35], [0.40]])
    assert float(loss(beta)) == 0.0


def test_dispersion_prior_positive_outside() -> None:
    loss = DispersionPriorLoss(lo=0.3, hi=0.4)
    assert float(loss(torch.tensor([[0.1]]))) == pytest.approx(0.2, abs=1e-6)
    assert float(loss(torch.tensor([[0.6]]))) == pytest.approx(0.2, abs=1e-6)


def test_dispersion_prior_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError):
        DispersionPriorLoss(lo=0.4, hi=0.3)


def test_source_consistency_zero_when_matched() -> None:
    loss = BlochSourceConsistencyLoss()
    y = torch.rand(2, 1, 8, 8)
    assert float(loss(prediction=y, target=y)) == 0.0


def test_source_consistency_positive_when_not() -> None:
    loss = BlochSourceConsistencyLoss()
    y = torch.rand(2, 1, 8, 8)
    assert float(loss(prediction=y + 0.2, target=y)) > 0.0


def test_reachable_via_create_loss() -> None:
    from spectramr.models.losses import create_loss

    assert isinstance(create_loss("dispersion_prior"), DispersionPriorLoss)
    assert isinstance(
        create_loss("bloch_source_consistency"), BlochSourceConsistencyLoss
    )
