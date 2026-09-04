"""Unit tests for PhysicalUnwarp.

Pin the composition properties:

- Identity: zero field maps + no GNL ⇒ output equals input.
- Order independence: composing (B0, Maxwell) gives the same
  result as a single field map equal to their sum (linearity in
  the small-perturbation regime).
- Differentiability: gradients flow through the warp.
- Shape contracts and explicit error messages on bad input.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.unwarp import PhysicalUnwarp  # noqa: E402


def _vol(B: int = 1, C: int = 1, D: int = 4, H: int = 16, W: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, C, D, H, W, generator=g)


def test_identity_no_field_no_gnl() -> None:
    vol = _vol()
    out = PhysicalUnwarp(echo_spacing_s=5.5e-4)(vol)
    # Identity grid + bilinear sampling = identity (up to fp noise).
    assert torch.allclose(out, vol, atol=1e-5)


def test_b0_and_maxwell_are_additive() -> None:
    """Composing B0 + Maxwell == single field equal to their sum."""
    vol = _vol()
    D, H, W = vol.shape[-3:]
    g = torch.Generator().manual_seed(1)
    b0 = torch.randn(D, H, W, generator=g) * 5.0  # Hz
    mx = torch.randn(D, H, W, generator=g) * 5.0  # Hz

    op = PhysicalUnwarp(echo_spacing_s=5.5e-4)
    a = op(vol, b0_hz=b0, maxwell_hz=mx)
    b = op(vol, b0_hz=(b0 + mx))
    assert torch.allclose(a, b, atol=1e-5)


def test_grad_flows_through_warp() -> None:
    """Backprop through grid_sample must reach `volume`."""
    vol = _vol().requires_grad_(True)
    D, H, W = vol.shape[-3:]
    b0 = torch.randn(D, H, W) * 5.0
    out = PhysicalUnwarp(echo_spacing_s=5.5e-4)(vol, b0_hz=b0)
    out.sum().backward()
    assert vol.grad is not None
    assert torch.any(vol.grad != 0)


def test_rejects_wrong_dimensionality() -> None:
    op = PhysicalUnwarp(echo_spacing_s=5.5e-4)
    with pytest.raises(ValueError, match="(B, C, D, H, W)"):
        op(torch.zeros(2, 1, 16, 16))  # 4D, not 5D


def test_rejects_bad_field_shape() -> None:
    op = PhysicalUnwarp(echo_spacing_s=5.5e-4)
    vol = _vol(B=2)
    bad = torch.zeros(7, 8, 8, 8)  # batch != 2 and != 1
    with pytest.raises(ValueError, match="\\(D,H,W\\) or \\(B,D,H,W\\)"):
        op(vol, b0_hz=bad)


def test_rejects_bad_gnl_shape() -> None:
    op = PhysicalUnwarp(echo_spacing_s=5.5e-4)
    vol = _vol(B=2)
    bad = torch.zeros(2, 2, 4, 16, 16)  # 2 channels, not 3
    with pytest.raises(ValueError, match="\\(B, 3, D, H, W\\)"):
        op(vol, gnl_displacement=bad)


def test_pe_axis_choice() -> None:
    """Switching pe_axis between H and W must give different outputs
    for an asymmetric field, because they shift along different axes."""
    vol = _vol()
    D, H, W = vol.shape[-3:]
    # Row-only ramp (varies along H) — symmetric along W.
    b0 = torch.zeros(D, H, W)
    b0 += torch.linspace(-10.0, 10.0, H).view(1, H, 1)

    out_h = PhysicalUnwarp(echo_spacing_s=5.5e-4, pe_axis="H")(vol, b0_hz=b0)
    out_w = PhysicalUnwarp(echo_spacing_s=5.5e-4, pe_axis="W")(vol, b0_hz=b0)
    assert not torch.allclose(out_h, out_w, atol=1e-4)
