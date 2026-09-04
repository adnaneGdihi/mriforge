"""Unit tests for models/stability/gan_balance_manager.py.

Covers: GANBalanceManager — configure_for_model_type, update_losses,
        should_skip_discriminator_update, get_discriminator_lr_factor.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.stability.gan_balance_manager import GANBalanceManager

_CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_manager_canary():
    """Manager initialises with expected attributes."""
    mgr = GANBalanceManager("standard", _CPU)
    assert mgr.total_steps == 0
    assert mgr.emergency_mode is False
    assert mgr.history_size == 0


# ---------------------------------------------------------------------------
# Parametrized: model-type-specific config
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type,expected_clip",
    [
        ("kan_unet", 1.0),
        ("vit_encoder", 1.0),
        ("unet_base", 1.0),
        ("diffusion_unet", 0.1),
    ],
)
def test_configure_for_model_type(model_type, expected_clip):
    mgr = GANBalanceManager(model_type, _CPU)
    assert mgr.gradient_clip_norm == pytest.approx(expected_clip, abs=1e-6), (
        f"{model_type}: expected clip {expected_clip}, got {mgr.gradient_clip_norm}"
    )


# ---------------------------------------------------------------------------
# update_losses behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_losses_increments_steps():
    mgr = GANBalanceManager("standard", _CPU)
    for i in range(5):
        mgr.update_losses(d_loss=0.5, g_loss=0.5)
    assert mgr.total_steps == 5
    assert mgr.history_size == 5


@pytest.mark.unit
def test_update_losses_ring_buffer_wraps():
    """Ring buffer wraps correctly when more than window_size steps recorded."""
    mgr = GANBalanceManager("standard", _CPU, window_size=5)
    for i in range(12):
        mgr.update_losses(d_loss=float(i), g_loss=float(i))
    assert mgr.total_steps == 12
    assert mgr.history_size == 5  # capped at window_size


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_skip_discriminator_returns_false_with_insufficient_history():
    """skip returns False when fewer than 5 history entries exist."""
    mgr = GANBalanceManager("standard", _CPU)
    for _ in range(3):
        mgr.update_losses(0.001, 1.0)
    assert not mgr.should_skip_discriminator_update()


@pytest.mark.unit
def test_lr_factor_returns_one_with_insufficient_history():
    mgr = GANBalanceManager("standard", _CPU)
    assert mgr.get_discriminator_lr_factor() == pytest.approx(1.0)


@pytest.mark.unit
def test_emergency_mode_activates_on_very_low_d_loss():
    """Repeated near-zero D-losses should activate emergency mode."""
    mgr = GANBalanceManager("standard", _CPU, emergency_threshold=0.05)
    for _ in range(15):
        mgr.update_losses(d_loss=0.001, g_loss=2.0)
    assert mgr.emergency_mode is True


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_window_size_zero_raises_on_update():
    """window_size=0 must raise IndexError on the first update (no silent wrong data)."""
    mgr = GANBalanceManager("standard", _CPU, window_size=0)
    with pytest.raises((IndexError, ZeroDivisionError, RuntimeError)):
        mgr.update_losses(d_loss=0.5, g_loss=0.5)
