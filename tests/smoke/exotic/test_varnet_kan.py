"""Tests for VarNet-KAN (SOTA 2.0).

Verifies:
- Conv2DKAN shape correctness.
- Spline basis function.
- Cascade integrity.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mriforge.models.spectral.varnet_kan import Conv2DKAN, SplineBasis, VarNetKAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_spline_basis():
    """Check Spline Basis output shape and range."""
    grid_size = 5
    basis_fn = SplineBasis(grid_size=grid_size).to(device)

    x = torch.linspace(-1.5, 1.5, 100, device=device)  # Test out of bounds too
    out = basis_fn(x)

    # Shape: [100, Grid+1]
    assert out.shape == (100, grid_size + 1)
    # Output should be non-negative (RBF/Spline property)
    assert (out >= 0).all()


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    in_channels=st.integers(min_value=1, max_value=2),  # Reduced range
    img_size=st.integers(min_value=16, max_value=32),
)
def test_conv2d_kan(batch_size, in_channels, img_size):
    """Test KAN Convolution."""
    model = Conv2DKAN(in_channels, in_channels, grid_size=3).to(device)
    x = torch.randn(batch_size, in_channels, img_size, img_size, device=device)

    out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    hidden_channels=st.integers(min_value=4, max_value=8),
)
def test_varnet_kan_cascade(batch_size, hidden_channels):
    """Test full VarNetKAN with Data Consistency Mock."""
    model = VarNetKAN(
        in_channels=2, num_cascades=1, hidden_channels=hidden_channels
    ).to(device)

    H, W = 16, 16
    x = torch.randn(batch_size, 2, H, W, device=device)

    # Mock K-space and Mask
    k0 = torch.randn(batch_size, H, W, dtype=torch.complex64, device=device)
    mask = torch.ones(batch_size, H, W, device=device)  # Full sampling

    # Forward with DC
    out = model(x, k0, mask)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__])
