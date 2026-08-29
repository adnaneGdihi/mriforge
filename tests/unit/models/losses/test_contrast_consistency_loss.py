r"""Tests for ``ContrastConsistencyLoss``.

Targets ``mriforge.models.losses.contrast_consistency_loss``.

Plan acceptance criteria (idea 2 §2.8):

- *"Loss is zero when T1 matches Bottomley prediction; positive
  otherwise"* — covered by the two property tests below.

Other contract tests:

- Construction validates ``weight`` / ``reduction``
- Shape mismatches raise
- Tissue-channel count must match ``tissue_order``
- Tissue-probability mass = 0 → loss = 0 (no anatomy to penalise)
- Gradient flows back to ``predicted_t1_ms``
- Registry: ``contrast_consistency`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.relaxation_priors import (
    TissueClass,
    bottomley_t1,
)
from mriforge.models.losses.contrast_consistency_loss import (
    DEFAULT_TISSUE_ORDER,
    ContrastConsistencyLoss,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_negative_weight_rejected() -> None:
    """``weight < 0`` raises."""
    with pytest.raises(ValueError, match="weight"):
        ContrastConsistencyLoss(weight=-1.0)


def test_invalid_reduction_rejected() -> None:
    """Only ``mean`` or ``sum`` accepted."""
    with pytest.raises(ValueError, match="reduction"):
        ContrastConsistencyLoss(reduction="median")


def test_default_tissue_order() -> None:
    """Default tissue ordering is ``(WM, GM, CSF)``."""
    loss = ContrastConsistencyLoss()
    assert loss.tissue_order == DEFAULT_TISSUE_ORDER


# ---------------------------------------------------------------------------
# Closed-form: zero loss when prediction matches Bottomley target
# ---------------------------------------------------------------------------


def _wm_only_probs(B: int, H: int, W: int) -> torch.Tensor:
    """Tissue probability that is all-WM, zero elsewhere."""
    K = 3  # WM, GM, CSF
    probs = torch.zeros(B, K, H, W)
    probs[:, 0] = 1.0  # WM
    return probs


def test_zero_loss_when_prediction_matches_bottomley_for_wm() -> None:
    """Pred = Bottomley(WM, B0) everywhere → loss = 0 with WM-only probs."""
    b0 = 1.5
    target = bottomley_t1(b0, TissueClass.WHITE_MATTER)
    pred = torch.full((1, 1, 4, 4), target)
    probs = _wm_only_probs(1, 4, 4)
    loss = ContrastConsistencyLoss(weight=1.0)
    val = loss(pred, probs, field_strength_T=b0).item()
    assert pytest.approx(val, abs=1e-6) == 0.0


def test_loss_grows_with_deviation_from_bottomley() -> None:
    """Larger gap between prediction and Bottomley target → larger loss."""
    b0 = 3.0
    target = bottomley_t1(b0, TissueClass.WHITE_MATTER)
    pred_close = torch.full((1, 1, 4, 4), target + 50.0)
    pred_far = torch.full((1, 1, 4, 4), target + 500.0)
    probs = _wm_only_probs(1, 4, 4)
    loss = ContrastConsistencyLoss(weight=1.0)
    val_close = loss(pred_close, probs, field_strength_T=b0).item()
    val_far = loss(pred_far, probs, field_strength_T=b0).item()
    assert val_far > val_close > 0.0


# ---------------------------------------------------------------------------
# Empty tissue mass
# ---------------------------------------------------------------------------


def test_zero_loss_when_no_tissue_mass() -> None:
    """All-zero probability map → zero loss (nothing to penalise)."""
    pred = torch.full((1, 1, 4, 4), 1234.0)
    probs = torch.zeros(1, 3, 4, 4)
    loss = ContrastConsistencyLoss(weight=1.0)
    val = loss(pred, probs, field_strength_T=1.5).item()
    assert val == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_predicted_t1_wrong_channel_count_rejected() -> None:
    """``predicted_t1_ms.shape[1] != 1`` raises."""
    pred = torch.zeros(1, 2, 4, 4)
    probs = torch.zeros(1, 3, 4, 4)
    loss = ContrastConsistencyLoss()
    with pytest.raises(ValueError, match="predicted_t1_ms"):
        loss(pred, probs, field_strength_T=1.5)


def test_tissue_probabilities_3d_rejected() -> None:
    """``tissue_probabilities`` must be 4-D."""
    pred = torch.zeros(1, 1, 4, 4)
    probs = torch.zeros(3, 4, 4)
    loss = ContrastConsistencyLoss()
    with pytest.raises(ValueError, match="tissue_probabilities"):
        loss(pred, probs, field_strength_T=1.5)


def test_tissue_channel_count_mismatch_rejected() -> None:
    """Channel count of probs must equal ``len(tissue_order)``."""
    pred = torch.zeros(1, 1, 4, 4)
    probs = torch.zeros(1, 5, 4, 4)  # too many
    loss = ContrastConsistencyLoss()
    with pytest.raises(ValueError, match="tissue_probabilities has"):
        loss(pred, probs, field_strength_T=1.5)


def test_zero_field_strength_rejected() -> None:
    """``B0 <= 0`` rejected."""
    pred = torch.zeros(1, 1, 4, 4)
    probs = torch.zeros(1, 3, 4, 4)
    loss = ContrastConsistencyLoss()
    with pytest.raises(ValueError, match="field_strength_T"):
        loss(pred, probs, field_strength_T=0.0)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_gradient_flows_to_predicted_t1() -> None:
    """Backward through the loss reaches ``predicted_t1_ms``."""
    pred = torch.full((1, 1, 4, 4), 500.0, requires_grad=True)
    probs = _wm_only_probs(1, 4, 4)
    loss = ContrastConsistencyLoss()
    loss(pred, probs, field_strength_T=3.0).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``contrast_consistency`` resolvable in the loss registry."""
    from mriforge.models.losses.registry import list_available

    assert "contrast_consistency" in list_available()
