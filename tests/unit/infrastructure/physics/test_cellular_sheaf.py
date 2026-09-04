"""Tests for cellular-sheaf cohomology primitives."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.cellular_sheaf import (
    global_sections_rank,
    mayer_vietoris_obstruction_norm,
    sheaf_coboundary,
    sheaf_laplacian_quadratic_form,
)


def test_constant_sheaf_with_identity_restrictions() -> None:
    """For identity restrictions, the coboundary is x_v - x_u."""
    x = torch.tensor([[1.0], [2.0], [3.0]])  # 3 vertices, stalk dim 1
    edges = torch.tensor([[0, 1], [1, 2]])
    eye = torch.eye(1).expand(2, 1, 1).clone()
    out = sheaf_coboundary(x, edges, eye, eye)
    assert torch.allclose(out, torch.tensor([[1.0], [1.0]]))


def test_laplacian_quadratic_form_zero_on_constant_section() -> None:
    """Constant section with identity restrictions has zero coboundary."""
    x = torch.ones(4, 2)  # 4 vertices, stalk dim 2
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    eye = torch.eye(2).expand(3, 2, 2).clone()
    q = sheaf_laplacian_quadratic_form(x, edges, eye, eye)
    assert q.item() == pytest.approx(0.0, abs=1e-6)


def test_laplacian_quadratic_form_positive_on_non_section() -> None:
    """Non-constant signal across identity-restriction edges has positive Q."""
    x = torch.tensor([[1.0], [2.0], [4.0]])
    edges = torch.tensor([[0, 1], [1, 2]])
    eye = torch.eye(1).expand(2, 1, 1).clone()
    q = sheaf_laplacian_quadratic_form(x, edges, eye, eye)
    # Diffs are 1 and 2 → q = 1² + 2² = 5.
    assert q.item() == pytest.approx(5.0, abs=1e-5)


def test_global_sections_dim_for_connected_graph() -> None:
    """For a connected graph with identity restrictions and stalk dim n,
    the global sections space is n-dimensional (constant vectors)."""
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    eye = torch.eye(2).expand(3, 2, 2).clone()
    rank = global_sections_rank(edges, eye, eye, num_vertices=4)
    assert rank == 2


def test_mayer_vietoris_zero_obstruction_on_agreeing_overlap() -> None:
    """[w] = 0 iff the two patches agree on the overlap."""
    overlap = torch.randn(8)
    res = mayer_vietoris_obstruction_norm(
        patch_a=torch.zeros(16),
        patch_b=torch.zeros(16),
        overlap_a=overlap,
        overlap_b=overlap.clone(),
    )
    assert res.item() == pytest.approx(0.0, abs=1e-6)


def test_mayer_vietoris_positive_on_disagreeing_overlap() -> None:
    """Non-zero overlap discrepancy → positive obstruction."""
    res = mayer_vietoris_obstruction_norm(
        patch_a=torch.zeros(16),
        patch_b=torch.zeros(16),
        overlap_a=torch.ones(8),
        overlap_b=torch.zeros(8),
    )
    # ||1 - 0||₂ = √8.
    assert res.item() == pytest.approx(8 ** 0.5, abs=1e-5)


def test_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError):
        sheaf_coboundary(torch.zeros(4), torch.tensor([[0, 1]]), torch.eye(1).unsqueeze(0), torch.eye(1).unsqueeze(0))
    with pytest.raises(ValueError):
        mayer_vietoris_obstruction_norm(
            torch.zeros(4), torch.zeros(4), torch.zeros(2), torch.zeros(3)
        )


def test_coboundary_gradient_flow() -> None:
    """The coboundary is differentiable in both signals and restriction maps."""
    x = torch.randn(3, 2, requires_grad=True)
    edges = torch.tensor([[0, 1], [1, 2]])
    r_l = torch.randn(2, 2, 2, requires_grad=True)
    r_r = torch.randn(2, 2, 2, requires_grad=True)
    out = sheaf_coboundary(x, edges, r_l, r_r)
    out.sum().backward()
    assert x.grad is not None
    assert r_l.grad is not None
    assert r_r.grad is not None
