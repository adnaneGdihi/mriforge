"""Tests for the Helmholtz-Hodge decomposition primitive."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.helmholtz_hodge import (
    curl,
    divergence,
    helmholtz_hodge_decompose,
    hodge_orthogonality_residual,
)


def test_divergence_of_constant_field_is_zero() -> None:
    """∇·c = 0 for constant c."""
    v = torch.ones(3, 8, 8, 8)
    d = divergence(v.unsqueeze(0)).squeeze(0)
    # Interior divergence is exactly zero (boundaries are zero-padded).
    assert torch.allclose(d[1:-1, 1:-1, 1:-1], torch.zeros(6, 6, 6), atol=1e-6)


def test_curl_of_constant_field_is_zero() -> None:
    """∇×c = 0 for constant c."""
    v = torch.ones(3, 8, 8, 8)
    c = curl(v.unsqueeze(0)).squeeze(0)
    assert torch.allclose(c[..., 1:-1, 1:-1, 1:-1], torch.zeros(3, 6, 6, 6), atol=1e-6)


def test_curl_of_gradient_is_zero() -> None:
    """∇×∇φ = 0 — vector calculus identity."""
    # Build a smooth scalar field φ.
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, 8),
        torch.linspace(-1, 1, 8),
        torch.linspace(-1, 1, 8),
        indexing="ij",
    )
    phi = (x ** 2 + y ** 2 + z ** 2)
    # ∇φ via central differences.
    grad_z = torch.zeros_like(phi)
    grad_y = torch.zeros_like(phi)
    grad_x = torch.zeros_like(phi)
    grad_z[1:-1] = 0.5 * (phi[2:] - phi[:-2])
    grad_y[:, 1:-1] = 0.5 * (phi[:, 2:] - phi[:, :-2])
    grad_x[:, :, 1:-1] = 0.5 * (phi[:, :, 2:] - phi[:, :, :-2])
    v = torch.stack([grad_z, grad_y, grad_x], dim=0)
    c = curl(v.unsqueeze(0)).squeeze(0)
    # Deep interior should have very small curl.
    assert c[..., 2:-2, 2:-2, 2:-2].abs().max().item() < 1e-3


def test_helmholtz_decomposition_shape() -> None:
    v = torch.randn(3, 8, 8, 8)
    v_irrot, v_sol = helmholtz_hodge_decompose(v)
    assert v_irrot.shape == v.shape
    assert v_sol.shape == v.shape


def test_helmholtz_decomposition_recovers_input() -> None:
    """v_irrot + v_sol = v (by construction)."""
    torch.manual_seed(0)
    v = torch.randn(3, 8, 8, 8)
    v_irrot, v_sol = helmholtz_hodge_decompose(v)
    assert torch.allclose(v_irrot + v_sol, v, atol=1e-5)


def test_hodge_orthogonality_residual_small_in_interior() -> None:
    """<v_irrot, v_sol> ≈ 0 up to boundary / discretisation error."""
    torch.manual_seed(0)
    v = torch.randn(3, 16, 16, 16) * 0.1
    v_irrot, v_sol = helmholtz_hodge_decompose(v)
    residual = hodge_orthogonality_residual(v_irrot, v_sol).item()
    # Order-of-magnitude check.
    energy = (v * v).sum().item()
    assert residual / max(energy, 1e-12) < 0.5


def test_solenoidal_part_is_divergence_free() -> None:
    """v_sol must be divergence-free to round-off.

    Regression: the Poisson solver previously inverted the narrow 3-point
    Laplacian (``-4·sin²(πk)``) while div/grad were zero-boundary differences,
    so the recovered ``v_sol`` had ``max|div(v_sol)| ~ O(1)``. With periodic
    operators + the matched ``-sin²(2πk)`` symbol the solenoidal part is
    genuinely divergence-free.
    """
    torch.manual_seed(0)
    v = torch.randn(3, 8, 8, 8)
    _, v_sol = helmholtz_hodge_decompose(v)
    div_vsol = divergence(v_sol.unsqueeze(0)).squeeze(0)
    assert div_vsol.abs().max().item() < 1e-4


def test_orthogonality_is_round_off_small() -> None:
    """The Hodge inner product is round-off small, not just < 0.5·energy."""
    torch.manual_seed(1)
    v = torch.randn(3, 8, 8, 8)
    v_irrot, v_sol = helmholtz_hodge_decompose(v)
    residual = hodge_orthogonality_residual(v_irrot, v_sol).item()
    energy = (v * v).sum().item()
    assert residual / max(energy, 1e-12) < 1e-3


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        divergence(torch.zeros(2, 2))
    with pytest.raises(ValueError):
        curl(torch.zeros(2, 2))
    with pytest.raises(ValueError):
        helmholtz_hodge_decompose(torch.zeros(4, 8, 8, 8))  # wrong leading dim
    with pytest.raises(ValueError):
        hodge_orthogonality_residual(torch.zeros(3, 4, 4, 4), torch.zeros(3, 4, 4, 5))
