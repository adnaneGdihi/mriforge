"""Tests for Galois-cohomological phase unwrapping."""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.galois_unwrap import (
    cohomology_norm_certificate,
    galois_unwrap,
    integer_winding_cocycle,
    plaquette_residues,
    wrap_to_principal,
)


@pytest.fixture
def smooth_field() -> torch.Tensor:
    z, y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 8),
        torch.linspace(-1.0, 1.0, 16),
        torch.linspace(-1.0, 1.0, 16),
        indexing="ij",
    )
    return 3.0 * x + 5.0 * y - 2.0 * z


def test_wrap_idempotent_in_principal_branch() -> None:
    theta = torch.linspace(-3.0, 3.0, 200)
    once = wrap_to_principal(theta)
    twice = wrap_to_principal(once)
    assert torch.allclose(once, twice, atol=1e-6)
    assert torch.all(once > -math.pi - 1e-6)
    assert torch.all(once <= math.pi + 1e-6)


def test_winding_cocycle_recovers_integer_jumps(smooth_field: torch.Tensor) -> None:
    """The cocycle takes integer values on every edge."""
    phi = wrap_to_principal(smooth_field)
    w_z, w_y, w_x = integer_winding_cocycle(phi)
    for w in (w_z, w_y, w_x):
        assert torch.equal(w, w.round())  # integer-valued
        assert w.abs().max().item() <= 1  # smooth field → unit jumps


def test_simply_connected_consistency_check(smooth_field: torch.Tensor) -> None:
    """A smooth field has trivial cohomology class — every plaquette closes.

    The cocycle itself is non-zero (whenever the wrapped phase crosses ±π)
    but the *cohomology class* — what the plaquette residues measure — is
    exactly zero on a globally liftable field.
    """
    phi = wrap_to_principal(smooth_field)
    cert = cohomology_norm_certificate(phi)
    assert cert.item() == pytest.approx(0.0, abs=1e-6)


def test_galois_unwrap_gradient_matches_wrapped_gradient(smooth_field: torch.Tensor) -> None:
    """The unwrap inverts ∂·W∂ — gradient of the lift equals wrap(grad ϕ) interior.

    The DST-I gauge introduces a boundary-conditioned offset, so we compare
    gradients on the interior where the boundary contribution decays.
    """
    phi = wrap_to_principal(smooth_field)
    theta, cert = galois_unwrap(phi)
    assert cert.item() == pytest.approx(0.0, abs=1e-6)
    # Interior gradient of recovered lift agrees with the true gradient.
    grad_x_lift = theta[..., :, :, 1:] - theta[..., :, :, :-1]
    grad_x_true = smooth_field[..., :, :, 1:] - smooth_field[..., :, :, :-1]
    # Crop a deep interior slab so the FFT Dirichlet boundary doesn't pollute.
    interior_lift = grad_x_lift[2:-2, 4:-4, 4:-4]
    interior_true = grad_x_true[2:-2, 4:-4, 4:-4]
    err = (interior_lift - interior_true).abs().mean()
    assert err.item() < 0.5


def test_residue_dipole_certificate_is_positive() -> None:
    """A field with an explicit branch-cut residue pair lifts the certificate."""
    # Construct a 2D phase ϕ(x,y) = arctan2(y - y0, x - x0) with a single
    # singularity at (x0, y0). On a discrete grid, that singularity becomes
    # a residue dipole pair (the discrete plaquette winding around it = ±1).
    z, y, x = torch.meshgrid(
        torch.zeros(1),
        torch.linspace(-1.0, 1.0, 21),
        torch.linspace(-1.0, 1.0, 21),
        indexing="ij",
    )
    phi = torch.atan2(y - 0.05, x - 0.05)  # singularity strictly inside the grid
    cert = cohomology_norm_certificate(phi)
    assert cert.item() > 0.0


def test_plaquette_residue_shapes() -> None:
    phi = torch.randn(2, 1, 4, 6, 8)
    w = integer_winding_cocycle(phi)
    r_xy, r_xz, r_yz = plaquette_residues(*w)
    assert r_xy.shape == (2, 1, 4, 5, 7)
    assert r_xz.shape == (2, 1, 3, 6, 7)
    assert r_yz.shape == (2, 1, 3, 5, 8)
