"""Tests for Gauss-MRF (SOTA 2.0).

Verifies:
- Continuous Dictionary shape and differentiability.
- Tissue Property extraction.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectramr.models.volumetric.gauss_mrf import GaussMRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    num_points=st.integers(min_value=50, max_value=100),
)
def test_continuous_dictionary_grads(batch_size, num_points):
    """Test gradient flow from signal to tissue params."""
    model = GaussMRF().to(device)

    # Tissue Params: [B, N, 3] (T1, T2, PD)
    tissue_params = torch.rand(
        batch_size, num_points, 3, device=device, requires_grad=True
    )
    seq_params = torch.rand(batch_size, 3, device=device)

    # Predict signal directly via dictionary for this test
    # (Bypassing TissueGaussians parsing to test dictionary core)
    signal = model.dictionary(tissue_params, seq_params)

    assert signal.shape == (batch_size, num_points, 2)

    # Scalar loss
    loss = signal.sum()
    loss.backward()

    # Check gradients
    assert tissue_params.grad is not None
    assert torch.abs(tissue_params.grad).sum() > 0


def test_gauss_mrf_forward():
    """Test full Gauss MRF forward pass with tensor input."""
    model = GaussMRF().to(device)
    B, C, D, H, W = 1, 13, 4, 4, 4

    # Input tensor with enough channels for T1/T2
    tissue_tensor = torch.randn(B, C, D, H, W, device=device)
    seq_params = torch.rand(B, 3, device=device)

    out = model(tissue_tensor, seq_params)

    assert "signal" in out
    assert "tissue_props" in out

    N = D * H * W
    assert out["signal"].shape == (B, N, 2)
    assert out["tissue_props"].shape == (B, N, 3)


if __name__ == "__main__":
    pytest.main([__file__])
