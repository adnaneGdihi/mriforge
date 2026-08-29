"""Tests for Metal Artifact Cold Diffusion (SOTA 2.0).

Verifies:
- Metal Degradation logic (Identity at t=0, Signal loss at t>0).
"""

import pytest
import torch

from mriforge.models.diffusion.cold_diffusion import MetalDegradation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_metal_degradation_identity():
    """At t=0, degradation should be identity."""
    H, W = 32, 32
    deg = MetalDegradation((H, W), total_steps=10)

    x = torch.randn(1, 2, H, W, device=device)

    out = deg(x, 0)  # t=0

    assert torch.allclose(x, out, atol=1e-5)


def test_metal_degradation_effect():
    """At t=10, should see signal loss in center (void)."""
    H, W = 32, 32
    deg = MetalDegradation((H, W), total_steps=10)

    # Constant image
    x = torch.ones(1, 2, H, W, device=device)

    out = deg(x, 10)  # Max degradation
    print(f"Output shape: {out.shape}")

    # Center should be 0 due to void
    cy, cx = H // 2, W // 2
    # Check center pixel
    center_val = out[0, :, cy, cx]
    assert torch.allclose(center_val, torch.zeros_like(center_val), atol=1e-5)

    # Edges should be relatively preserved (or at least non-zero)
    edge_val = out[0, :, 0, 0]
    assert not torch.allclose(edge_val, torch.zeros_like(edge_val), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
