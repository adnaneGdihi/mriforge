"""Tests for Gen-3DGS (SOTA 2.0).

Verifies:
- 3D VAE Encoder/Decoder shapes.
- Gaussian Parameter constraints (Scale > 0, Opacity in [0, 1]).
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectramr.models.volumetric.gen_3dgs import Gen3DGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    spatial_dim=st.integers(min_value=16, max_value=32).map(
        lambda x: x if x % 8 == 0 else 16
    ),  # Divisible by 8
)
def test_gen_3dgs_forward(batch_size, spatial_dim):
    """Test full VAE forward pass and parameter validity."""
    model = Gen3DGS(in_channels=1, latent_dim=32).to(device)

    # 3D Volume: [B, 1, D, H, W]
    D, H, W = spatial_dim, spatial_dim, spatial_dim
    x = torch.randn(batch_size, 1, D, H, W, device=device)

    z, params = model(x)

    # Check shape
    # Latent should be reduced by 8
    expected_latent_dim = (batch_size, 32, D // 8, H // 8, W // 8)
    assert z.shape == expected_latent_dim

    # Output grid should match input spatial dims (if strict U-Net VAE)
    # Our decoder upsamples 3 times (*2, *2, *2) from /8 -> Orignal resolution
    assert params.raw.shape == (batch_size, 12, D, H, W)

    # Check constraints
    scale = params.scale
    assert (scale > 0).all()

    opacity = params.opacity
    assert (opacity >= 0).all() and (opacity <= 1).all()

    rotation = params.rotation
    # Quaternions should be unit norm
    assert torch.allclose(
        torch.norm(rotation, dim=1), torch.tensor(1.0, device=device), atol=1e-5
    )


if __name__ == "__main__":
    pytest.main([__file__])
