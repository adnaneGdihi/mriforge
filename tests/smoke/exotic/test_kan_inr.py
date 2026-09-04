"""Tests for KAN-INR (SOTA 2.0).

Verifies:
- Coordinate query processing.
- Batch handling.
- Shape consistency.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectramr.models.spectral.kan_inr import KANINR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    num_points=st.integers(min_value=100, max_value=200),
    dim=st.integers(min_value=2, max_value=3),
)
def test_kan_inr_forward(batch_size, num_points, dim):
    """Verify KAN-INR output shapes."""
    model = KANINR(in_features=dim, hidden_features=32, hidden_layers=2).to(device)

    # Coords in [-1, 1]
    coords = torch.rand(batch_size, num_points, dim, device=device) * 2 - 1

    out = model(coords)

    assert out.shape == (batch_size, num_points, 2)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__])
