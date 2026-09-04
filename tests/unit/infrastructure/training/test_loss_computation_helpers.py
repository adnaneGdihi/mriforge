"""Unit tests for infrastructure/training/loss_computation_helpers.py.

Covers: GANLossHelper.extract_generator_losses,
        ReconstructionLossHelper.compute_deep_supervision_loss,
        and the r1-disabled branch of apply_r1_regularization.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.loss_computation_helpers import (
    GANLossHelper,
    ReconstructionLossHelper,
)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gan_loss_helper_canary():
    """R1 disabled → apply_r1_regularization returns zero tensor."""
    helper = GANLossHelper(enable_r1_regularization=False)
    device = torch.device("cpu")
    # Dummy real images — not used when R1 is disabled
    real = torch.randn(2, 1, 8, 8)
    import torch.nn as nn

    disc = nn.Identity()  # minimal discriminator stand-in
    result = helper.apply_r1_regularization(disc, real)
    assert result.item() == pytest.approx(0.0), "Disabled R1 should return 0.0"


# ---------------------------------------------------------------------------
# Parametrized: extract_generator_losses
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "losses",
    [
        {"g_adv": torch.tensor(1.0), "g_l1": torch.tensor(0.5)},
        {"loss_total": torch.tensor(2.0)},
        {},
    ],
)
def test_extract_generator_losses_returns_copy(losses):
    """extract_generator_losses returns a copy of all input losses."""
    helper = GANLossHelper()
    result = helper.extract_generator_losses(losses)
    assert result == losses
    # Confirm it is a shallow copy, not the same object
    if losses:
        assert result is not losses


# ---------------------------------------------------------------------------
# Parametrized: compute_deep_supervision_loss
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("n_outputs", [0, 1, 3])
def test_compute_deep_supervision_loss(n_outputs):
    """Deep supervision loss returns scalar tensor for any number of outputs."""
    helper = ReconstructionLossHelper()
    target = torch.zeros(2, 1, 16, 16)
    intermediates = [torch.zeros(2, 1, 16, 16) for _ in range(n_outputs)]
    result = helper.compute_deep_supervision_loss(intermediates, target)
    assert result.ndim == 0 or result.ndim <= 1  # scalar or reducible
    assert torch.isfinite(result)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_deep_supervision_empty_outputs():
    """Empty intermediate outputs return zero loss (not crash)."""
    helper = ReconstructionLossHelper()
    target = torch.ones(2, 1, 16, 16)
    result = helper.compute_deep_supervision_loss([], target)
    assert result.item() == pytest.approx(0.0)


@pytest.mark.unit
def test_reconstruction_helper_default_state():
    """Default reconstruction helper has no FFT transformer or DC layer."""
    helper = ReconstructionLossHelper()
    assert helper.fft_transformer is None
    assert helper.dc_layer is None


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gan_loss_helper_extract_preserves_tensor_values():
    """Values in extracted dict are identical to inputs (not zeroed / modified)."""
    helper = GANLossHelper()
    val = torch.tensor(3.14)
    losses = {"my_loss": val}
    result = helper.extract_generator_losses(losses)
    assert result["my_loss"].item() == pytest.approx(3.14, abs=1e-5)
