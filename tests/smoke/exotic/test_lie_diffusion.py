"""Tests for Lie Group Diffusion (SOTA 2.0).

Verifies:
- SO(3) Exp/Log map consistency.
- Rotation Matrix properties (Determinant, Orthogonality).
"""

import pytest
import torch

from spectramr.models.generative.lie_diffusion import SO3, LieDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_so3_exp_properties():
    """Test Exp Map creates valid rotation matrices."""
    B = 10
    v = torch.randn(B, 3, device=device)
    R = SO3.exp_map(v)

    # Check orthogonality: R @ R.T = I
    I = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
    R_RT = torch.bmm(R, R.transpose(1, 2))
    assert torch.allclose(R_RT, I, atol=1e-5)

    # Check determinant = 1
    det = torch.det(R)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_so3_log_consistency():
    """Test Log(Exp(v)) approx v for small v."""
    B = 10
    v = torch.randn(B, 3, device=device) * 0.1  # Small angles for bijection

    R = SO3.exp_map(v)
    v_rec = SO3.log_map(R)

    assert torch.allclose(v, v_rec, atol=1e-5)


def test_lie_diffusion_forward():
    """Test Lie Diffusion forward pass."""
    model = LieDiffusion(hidden_dim=16).to(device)
    B = 5
    R0 = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)

    loss = model(R0, t=1)

    assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__])
