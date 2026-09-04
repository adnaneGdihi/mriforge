"""Tests for MonotoneFieldOrderedGenerator (B-2.6)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.models.generators.monotone_field_generator import (
    MonotoneFieldOrderedGenerator,
)


def test_forward_shape() -> None:
    m = MonotoneFieldOrderedGenerator(n_bases=4)
    out = m(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 7.0]))
    assert out.shape == (2, 1, 16, 16)


def test_render_is_monotone_in_field_by_construction() -> None:
    # HEADLINE (anti-facade): with enforce_monotone=True the render is non-decreasing
    # in the field for EVERY pixel — and this holds at init (the softplus cumulative-
    # residual head guarantees it structurally, no training needed).
    torch.manual_seed(0)
    m = MonotoneFieldOrderedGenerator(enforce_monotone=True, n_bases=4).eval()
    x = torch.rand(3, 1, 16, 16)
    prev = None
    for fld in (0.1, 0.5, 1.5, 3.0, 7.0):
        cur = m(x, field_strength=torch.full((3,), fld))
        if prev is not None:
            assert torch.all(cur >= prev - 1e-5), f"inversion at field {fld}"
        prev = cur


def test_nonmonotone_model_can_invert() -> None:
    # The ablation (enforce_monotone=False) uses raw signed coefficients, so the render
    # is NOT guaranteed monotone — a strongly-negative coefficient inverts the order.
    m = MonotoneFieldOrderedGenerator(enforce_monotone=False, n_bases=4).eval()
    with torch.no_grad():
        nn.init.zeros_(m.coef_head.weight)
        nn.init.constant_(m.coef_head.bias, -2.0)  # negative increment => decreasing
    x = torch.rand(1, 1, 16, 16)
    lo = m(x, field_strength=torch.tensor([0.5]))
    hi = m(x, field_strength=torch.tensor([7.0]))
    assert torch.any(hi < lo - 1e-4)  # higher field renders SMALLER -> not monotone


def test_field_changes_output() -> None:
    # The field is load-bearing (monotone bases vary with tau) even at init.
    torch.manual_seed(0)
    m = MonotoneFieldOrderedGenerator(n_bases=4).eval()
    x = torch.rand(2, 1, 16, 16)
    a = m(x, field_strength=torch.tensor([0.1, 0.1]))
    b = m(x, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(a, b)


def test_rejects_complex() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        MonotoneFieldOrderedGenerator()(
            torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.0])
        )


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        MonotoneFieldOrderedGenerator(in_channels=2)


def test_rejects_inverted_tau_range() -> None:
    # Review #1: tau_max <= tau_min would make the bases DECREASING -> silently invert
    # the monotone-by-construction guarantee. Must raise at construction.
    with pytest.raises(ValueError, match="tau_max"):
        MonotoneFieldOrderedGenerator(tau_min=0.85, tau_max=-1.0)
    with pytest.raises(ValueError, match="tau_max"):
        MonotoneFieldOrderedGenerator(tau_min=0.0, tau_max=0.0)


def test_registered() -> None:
    from spectramr.models.registry import MODEL_REGISTRY

    assert "monotone_field_ordered_generator" in MODEL_REGISTRY
