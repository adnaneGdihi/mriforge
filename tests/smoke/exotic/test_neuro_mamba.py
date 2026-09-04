"""Tests for Neuromorphic Mamba (SOTA 2.0).

Verifies:
- Spiking behavior (binary outputs).
- Surrogate gradient flow.
- Shape consistency.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectramr.models.mamba.neuro_mamba import NeuroMamba, surrogate_spike

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_surrogate_gradient():
    """Verify backprop works through the spike function."""
    x = torch.tensor([0.5, 1.5], requires_grad=True)
    thresh = 1.0

    # Forward
    y = surrogate_spike(x, thresh)
    assert torch.allclose(y, torch.tensor([0.0, 1.0]))

    # Backward
    y.sum().backward()
    assert x.grad is not None
    assert torch.all(
        x.grad != 0
    )  # Should have non-zero gradients even for 0.5 (sigmoid tail)


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=10, max_value=30),
    d_model=st.integers(min_value=4, max_value=16),
)
def test_neuro_mamba_forward(batch_size, seq_len, d_model):
    """Test full forward pass."""
    model = NeuroMamba(in_channels=d_model, base_dim=d_model, num_layers=2).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    out = model(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    # Check if there's spiking activity?
    # Hard to check internal state from outside without hooks, but gradients check implicit connectivity.
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()


if __name__ == "__main__":
    pytest.main([__file__])
