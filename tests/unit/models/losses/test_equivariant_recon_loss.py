"""Tests for ``EquivariantSSLReconLoss``.

Targets ``mriforge.models.losses.equivariant_recon_loss``. Measurement-
domain equivariance penalty: ``f(R y) - R f(y)``.

Categories:

- Construction: invalid ``norm`` rejected
- Identity (the two branches are equal) → loss = 0
- ``norm="l1"`` and ``norm="l2"`` produce different (but related)
  values
- Missing ``context`` raises
- Shape mismatch between branches raises
- Gradient flows back to the prediction
- Registry: ``equivariant_ssl_recon`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.equivariant_recon_loss import EquivariantSSLReconLoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_unknown_norm_rejected() -> None:
    """Only ``l1`` / ``l2`` accepted."""
    with pytest.raises(ValueError, match="norm"):
        EquivariantSSLReconLoss(norm="cosine")


def test_default_norm_is_l2() -> None:
    """Default norm is ``l2``."""
    loss = EquivariantSSLReconLoss()
    assert loss.norm == "l2"


# ---------------------------------------------------------------------------
# Identity → zero loss
# ---------------------------------------------------------------------------


def test_zero_loss_when_branches_match() -> None:
    """``f(R y) == R f(y)`` → loss = 0."""
    loss = EquivariantSSLReconLoss(norm="l2")
    pred = torch.randn(1, 1, 4, 4)
    out = loss(pred, target=torch.zeros_like(pred), context={"transformed_recon": pred})
    assert pytest.approx(out.item(), abs=1e-7) == 0.0


# ---------------------------------------------------------------------------
# Norm switch
# ---------------------------------------------------------------------------


def test_l1_and_l2_differ_on_random_inputs() -> None:
    """Same inputs, different norms → different scalar values."""
    pred = torch.randn(1, 1, 4, 4)
    ref = torch.randn(1, 1, 4, 4)
    loss_l1 = EquivariantSSLReconLoss(norm="l1")
    loss_l2 = EquivariantSSLReconLoss(norm="l2")
    val_l1 = loss_l1(pred, target=torch.zeros_like(pred), context={"transformed_recon": ref}).item()
    val_l2 = loss_l2(pred, target=torch.zeros_like(pred), context={"transformed_recon": ref}).item()
    assert val_l1 != val_l2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_context_raises() -> None:
    """``context=None`` rejected."""
    loss = EquivariantSSLReconLoss()
    with pytest.raises(ValueError, match="context"):
        loss(torch.zeros(1, 1, 4, 4), target=torch.zeros(1, 1, 4, 4))


def test_missing_transformed_recon_raises() -> None:
    """Empty context dict rejected."""
    loss = EquivariantSSLReconLoss()
    with pytest.raises(ValueError, match="transformed_recon"):
        loss(torch.zeros(1, 1, 4, 4), target=torch.zeros(1, 1, 4, 4), context={})


def test_shape_mismatch_raises() -> None:
    """Branches must have matching shapes."""
    loss = EquivariantSSLReconLoss()
    pred = torch.zeros(1, 1, 4, 4)
    ref = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="shape"):
        loss(pred, target=torch.zeros_like(pred), context={"transformed_recon": ref})


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_prediction() -> None:
    """Backward through the loss reaches the prediction."""
    loss = EquivariantSSLReconLoss()
    pred = torch.randn(1, 1, 4, 4, requires_grad=True)
    ref = torch.randn(1, 1, 4, 4)
    loss(pred, target=torch.zeros_like(pred), context={"transformed_recon": ref}).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``equivariant_ssl_recon`` resolvable."""
    from mriforge.models.losses.registry import list_available

    assert "equivariant_ssl_recon" in list_available()
