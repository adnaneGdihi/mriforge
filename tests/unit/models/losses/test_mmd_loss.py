"""Tests for ``MMDLoss``.

Targets ``mriforge.models.losses.mmd_loss``. Multi-bandwidth Gaussian-kernel
Maximum Mean Discrepancy: a closed-form, gradient-friendly two-sample
distance for unpaired domain alignment.

Categories:

- Construction: invalid sigmas raise; default tuple stored
- Property: ``MMD(P, P) ≈ 0`` (same distribution)
- Property: ``MMD(P, Q) > 0`` for genuinely-different distributions
- Edge: < 2 samples per side raises (unbiased estimator denominator)
- Sanity-shape: 4D input flattens correctly
- Registry: ``mmd`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.mmd_loss import MMDLoss, _multi_kernel_mmd


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_sigmas_persisted() -> None:
    """Default sigma tuple has 5 bandwidths."""
    loss = MMDLoss()
    assert loss.sigmas == (1.0, 2.0, 4.0, 8.0, 16.0)


def test_empty_sigmas_raises() -> None:
    """Empty sigma tuple → ``ValueError``."""
    with pytest.raises(ValueError, match="at least one bandwidth"):
        MMDLoss(sigmas=())


def test_non_positive_sigma_raises() -> None:
    """Any sigma ≤ 0 → ``ValueError``."""
    with pytest.raises(ValueError, match="must be > 0"):
        MMDLoss(sigmas=(1.0, -2.0))


# ---------------------------------------------------------------------------
# Property: MMD properties
# ---------------------------------------------------------------------------


def test_mmd_same_distribution_near_zero() -> None:
    """``MMD(P, P)`` for two large samples from N(0,1) is small."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(1.0, 2.0, 4.0))
    x = torch.randn(64, 8)
    y = torch.randn(64, 8)
    out = loss(x, y)
    # Two identically-distributed Gaussians: MMD is small but not exactly zero.
    assert out.abs().item() < 1.0


def test_mmd_different_distribution_positive() -> None:
    """Mean-shifted samples → MMD is meaningfully positive."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(1.0, 2.0, 4.0))
    x = torch.randn(64, 8)
    y = torch.randn(64, 8) + 5.0  # mean-shifted
    out = loss(x, y)
    assert out.item() > 0.1  # well-separated → strictly positive


def test_mmd_grows_with_separation() -> None:
    """Larger mean shift → larger MMD."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(1.0, 2.0, 4.0))
    x = torch.randn(64, 8)
    y_small = torch.randn(64, 8) + 1.0
    y_big = torch.randn(64, 8) + 5.0
    mmd_small = loss(x, y_small).item()
    mmd_big = loss(x, y_big).item()
    assert mmd_big > mmd_small


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


def test_too_few_samples_raises() -> None:
    """``n < 2`` or ``m < 2`` → ``ValueError``."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        _multi_kernel_mmd(torch.randn(1, 4), torch.randn(4, 4), sigmas=(1.0,))


def test_unequal_batch_sizes_supported() -> None:
    """Different ``n`` and ``m`` are both accepted (unpaired alignment)."""
    torch.manual_seed(0)
    loss = MMDLoss()
    x = torch.randn(16, 4)
    y = torch.randn(32, 4)
    out = loss(x, y)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Spatial flattening
# ---------------------------------------------------------------------------


def test_flatten_spatial_default_handles_4d() -> None:
    """``[B, C, H, W]`` input is flattened to ``[B, C·H·W]`` before MMD."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(2.0,))
    x = torch.randn(4, 1, 8, 8)
    y = torch.randn(4, 1, 8, 8)
    out = loss(x, y)
    assert out.dim() == 0  # scalar


def test_flatten_disabled_requires_2d() -> None:
    """``flatten_spatial=False`` consumes ``[B, D]`` directly."""
    loss = MMDLoss(sigmas=(2.0,), flatten_spatial=False)
    x = torch.randn(4, 16)
    y = torch.randn(4, 16)
    out = loss(x, y)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_input() -> None:
    """Backward through MMD reaches the prediction."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(2.0,))
    x = torch.randn(8, 4, requires_grad=True)
    y = torch.randn(8, 4)
    loss(x, y).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_mmd_registered() -> None:
    """``mmd`` is in the loss registry."""
    from mriforge.models.losses.registry import list_available

    assert "mmd" in list_available()
