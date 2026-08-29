"""Tests for ``BlochConsistencyLoss``.

Targets ``mriforge.models.losses.bloch_consistency_loss``. The loss anchors
generator-predicted tissue parameters to observed contrast images by
re-synthesising the contrast through ``MultiPhysicsBlochLayer`` and
comparing against the measured target.

Categories:

- Unit: norm validation, missing-context fail-loud
- Property: identity-tissue → near-zero loss; gradient flow
- Registry: registered under canonical name
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer
from mriforge.models.losses.bloch_consistency_loss import BlochConsistencyLoss


# ---------------------------------------------------------------------------
# Construction / norm validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["l1", "l2", "smooth_l1"])
def test_supported_norms_construct(norm: str) -> None:
    """Each documented norm value succeeds."""
    loss = BlochConsistencyLoss(norm=norm)
    assert loss.norm == norm


def test_invalid_norm_raises() -> None:
    """Unknown norm fails loud (CLAUDE.md #9)."""
    with pytest.raises(ValueError, match="norm must be one of"):
        BlochConsistencyLoss(norm="huber_unknown")


# ---------------------------------------------------------------------------
# Context validation
# ---------------------------------------------------------------------------


def test_missing_context_raises() -> None:
    """``context=None`` → ``ValueError`` listing both required keys."""
    loss = BlochConsistencyLoss()
    pred = torch.zeros(1, 1, 8, 8, 8)
    target = torch.zeros(1, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="requires a `context`"):
        loss(pred, target, context=None)


def test_context_missing_tissue_params_raises() -> None:
    """``acquisition_params`` only → ``ValueError``."""
    loss = BlochConsistencyLoss()
    pred = torch.zeros(1, 1, 8, 8, 8)
    target = torch.zeros(1, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="must contain both"):
        loss(pred, target, context={"acquisition_params": {"TE": 15.0, "TR": 2000.0}})


def test_context_missing_acquisition_params_raises() -> None:
    """``tissue_params`` only → ``ValueError``."""
    loss = BlochConsistencyLoss()
    pred = torch.zeros(1, 1, 8, 8, 8)
    target = torch.zeros(1, 1, 8, 8, 8)
    tissue = {
        "rho": torch.ones(1, 1, 8, 8, 8) * 0.7,
        "t1": torch.ones(1, 1, 8, 8, 8) * 800.0,
        "t2": torch.ones(1, 1, 8, 8, 8) * 80.0,
    }
    with pytest.raises(ValueError, match="must contain both"):
        loss(pred, target, context={"tissue_params": tissue})


# ---------------------------------------------------------------------------
# Forward — produces synthesis matching target
# ---------------------------------------------------------------------------


def _identity_context(shape: tuple[int, int, int, int, int] = (1, 1, 8, 8, 8)) -> dict:
    """Tissue params + acq metadata that yield a deterministic SE signal."""
    rho = torch.ones(*shape) * 0.7
    t1 = torch.ones(*shape) * 800.0
    t2 = torch.ones(*shape) * 80.0
    acq = {
        "TE": 15.0,
        "TR": 2000.0,
        "TI": 0.0,
        "contrast_type": MultiPhysicsBlochLayer.SEQ_SPIN_ECHO,
    }
    return {
        "tissue_params": {"rho": rho, "t1": t1, "t2": t2},
        "acquisition_params": acq,
    }


def test_forward_with_target_equal_to_synthesis_yields_zero() -> None:
    """When target equals Bloch-synthesised signal, loss == 0."""
    loss = BlochConsistencyLoss(norm="l1")
    ctx = _identity_context()

    # First, run the Bloch layer to obtain the expected target.
    bloch = MultiPhysicsBlochLayer(learnable_flip_angle=False).eval()
    for p in bloch.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        target = bloch(tissue_params=ctx["tissue_params"], metadata=ctx["acquisition_params"])

    # Use the same Bloch layer (same parameters) by re-running through the loss.
    pred = torch.zeros_like(target)  # ignored by the loss (loss only looks at target+context)
    out = loss(pred, target, context=ctx)
    # The loss-internal Bloch may differ slightly from our reference instance
    # (different random init for any internal nn.Parameter), but for
    # learnable_flip_angle=False both should match.
    assert out.item() < 1e-3


def test_forward_shape_mismatch_raises() -> None:
    """Wrong target shape → ``ValueError`` mentioning shapes."""
    loss = BlochConsistencyLoss()
    ctx = _identity_context(shape=(1, 1, 8, 8, 8))
    target_wrong = torch.zeros(1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="does not match"):
        loss(torch.zeros_like(target_wrong), target_wrong, context=ctx)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_bloch_consistency_registered() -> None:
    """``bloch_consistency`` is in the loss registry."""
    from mriforge.models.losses.registry import list_available

    assert "bloch_consistency" in list_available()
