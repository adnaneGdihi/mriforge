"""Tests for ``HistogramConsistencyLoss``.

Targets ``mriforge.models.losses.histogram_loss``. Differentiable
soft-binned-histogram + L1-CDF distance — an Earth Mover's
Distance-style intensity-distribution alignment loss.

Categories:

- Construction: bin centres span ``[min_val, max_val]`` symmetrically
- Property: identity inputs → zero loss
- Property: shifted intensity → strictly positive
- Property: histogram normalises to a proper probability mass
- Sanity-shape: 4D inputs accepted
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.histogram_loss import HistogramConsistencyLoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default ``bins=100`` and ``[0, 1]`` range."""
    loss = HistogramConsistencyLoss()
    assert loss.bins == 100
    assert loss.min_val == 0.0
    assert loss.max_val == 1.0
    assert loss.bin_centers.shape == (100,)


def test_bin_centers_symmetric() -> None:
    """Bin centres land at half-bin offsets from each edge."""
    loss = HistogramConsistencyLoss(bins=4, min_val=0.0, max_val=1.0)
    expected = torch.tensor([0.125, 0.375, 0.625, 0.875])
    assert torch.allclose(loss.bin_centers, expected)


def test_default_sigma_equals_delta() -> None:
    """Default sigma equals one bin width."""
    loss = HistogramConsistencyLoss(bins=10)
    assert loss.sigma == loss.delta


# ---------------------------------------------------------------------------
# Soft histogram
# ---------------------------------------------------------------------------


def test_soft_histogram_sums_to_one() -> None:
    """Per-batch soft histogram is a probability mass (sums to ≈ 1)."""
    loss = HistogramConsistencyLoss(bins=20, min_val=0.0, max_val=1.0)
    x = torch.rand(2, 1, 16, 16)
    hist = loss._compute_soft_histogram(x)
    sums = hist.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_soft_histogram_shape() -> None:
    """Histogram has shape ``[B, bins]``."""
    loss = HistogramConsistencyLoss(bins=32)
    x = torch.rand(3, 1, 8, 8)
    hist = loss._compute_soft_histogram(x)
    assert hist.shape == (3, 32)


# ---------------------------------------------------------------------------
# Forward — identity & shift
# ---------------------------------------------------------------------------


def test_zero_loss_for_identity() -> None:
    """``loss(x, x) == 0`` (matching CDFs)."""
    torch.manual_seed(0)
    loss = HistogramConsistencyLoss(bins=20)
    x = torch.rand(1, 1, 16, 16)
    out = loss(x, x)
    assert pytest.approx(out.item(), abs=1e-7) == 0.0


def test_loss_grows_with_intensity_shift() -> None:
    """A constant intensity shift moves the CDF, increasing the loss."""
    torch.manual_seed(0)
    loss = HistogramConsistencyLoss(bins=50, min_val=0.0, max_val=1.0)
    x = torch.rand(1, 1, 32, 32) * 0.4 + 0.1  # values in [0.1, 0.5]
    y_close = x + 0.05
    y_far = x + 0.4
    out_close = loss(x, y_close).item()
    out_far = loss(x, y_far).item()
    assert out_far > out_close


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


def test_handles_multichannel_input() -> None:
    """Multi-channel input is flattened per-batch and runs without errors."""
    loss = HistogramConsistencyLoss(bins=16)
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    out = loss(x, y)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_through_soft_binning() -> None:
    """Backward reaches the input via the Gaussian soft-binning."""
    loss = HistogramConsistencyLoss(bins=16)
    x = torch.rand(1, 1, 8, 8, requires_grad=True)
    y = torch.rand(1, 1, 8, 8)
    loss(x, y).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_histogram_consistency_registered() -> None:
    """``histogram_consistency`` is in the loss registry."""
    from mriforge.models.losses.registry import list_available

    assert "histogram_consistency" in list_available()
