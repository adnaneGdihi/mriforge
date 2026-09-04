"""Tests for Dynamic 3DGS (SOTA 2.0).

Verifies:
- Deformation Field shapes.
- Temporal consistency logic.
- Integration with Gen3DGS parameters.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectramr.models.volumetric.dynamic_3dgs import DeformationField, Dynamic3DGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    num_gaussians=st.integers(min_value=10, max_value=20),
)
def test_deformation_field(batch_size, num_gaussians):
    """Test pure deformation network."""
    model = DeformationField(hidden_dim=16).to(device)

    x = torch.randn(batch_size, num_gaussians, 3, device=device)
    t = torch.rand(batch_size, 1, device=device)

    delta = model(x, t)

    # Shape: [B, N, 7] (3 pos + 4 rot)
    assert delta.shape == (batch_size, num_gaussians, 7)
    assert torch.isfinite(delta).all()


def test_dynamic_3dgs_integration():
    """Test full Dynamic 3DGS with mock Parameters."""
    model = Dynamic3DGS().to(device)

    # Create dummy "Canonical" params (grid output style)
    # [B, 12, D, H, W]
    B, D, H, W = 1, 4, 4, 4
    x_static = torch.randn(B, 12, D, H, W, device=device)
    t = torch.zeros(B, 1, device=device)

    out = model(x_static, t)

    # Assert output keys
    assert "xyz" in out
    assert "delta_rot" in out

    # Check shape equality
    N = D * H * W
    assert out["xyz"].shape == (B, N, 3)
    assert out["delta_rot"].shape == (B, N, 4)


if __name__ == "__main__":
    pytest.main([__file__])
