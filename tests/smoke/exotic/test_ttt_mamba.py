"""
Unit tests for TTT-Mamba.
"""

import pytest
import torch

from spectramr.models.edge.ttt_mamba import TTTMambaBlock


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_ttt_adaptation(device):
    """Verify that TTT step actually updates weights/output."""
    B, L, D = 1, 16, 8
    model = TTTMambaBlock(dim=D, ttt_lr=0.1, ttt_steps=1).to(device)
    model.eval()  # Enable TTT mode

    x = torch.randn(B, L, D, device=device)

    # 1. Run Forward (Includes TTT)
    out_adapted = model(x)

    # 2. Check if weights inside adapter changed?
    # Actually TTT implementation Clones weights, so self.adapter.weight is UNCHANGED.
    # We need to verify that the output is DIFFERENT from a standard forward using original weights.

    # Run standard forward manually (no TTT)
    with torch.no_grad():
        out_standard = model._forward_impl(x, model.adapter.weight)

    # Check divergence
    diff = (out_adapted - out_standard).abs().sum()
    assert diff > 0, "TTT should change the output vs standard forward"


def test_ttt_training_mode(device):
    """Verify standard training mode bypasses TTT."""
    model = TTTMambaBlock(dim=8).to(device)
    model.train()

    x = torch.randn(2, 16, 8, device=device)
    out = model(x)

    assert out.shape == (2, 16, 8)
    # Check requires_grad in out
    assert out.requires_grad
