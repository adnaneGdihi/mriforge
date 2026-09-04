"""Unit tests for the BlochRelaxationManifold — Phase 0 of audit_plan_novel."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold


@pytest.fixture
def manifold() -> BlochRelaxationManifold:
    return BlochRelaxationManifold(n_mc_samples=4, sigma_signal=0.05)


def test_metric_tensor_is_psd(manifold: BlochRelaxationManifold) -> None:
    """Fisher metric must be symmetric positive semidefinite."""
    theta = torch.tensor([[1.0, 1200.0, 80.0]])  # (M0, T1, T2)
    G = manifold.metric_tensor(theta)
    assert G.shape == (1, 3, 3)
    # symmetry
    assert torch.allclose(G, G.transpose(-1, -2), atol=1e-5)
    # PSD: smallest eigenvalue ≥ 0 (allow small numerical slack from regulariser)
    eigvals = torch.linalg.eigvalsh(G)
    assert eigvals.min().item() >= -1e-4


def test_project_tangent_interior_is_identity(manifold: BlochRelaxationManifold) -> None:
    theta = torch.tensor([[1.0, 1200.0, 80.0]])  # T1 - T2 = 1120, far from boundary
    v = torch.tensor([[0.1, 5.0, -2.0]])
    v_proj = manifold.project_tangent(theta, v)
    assert torch.allclose(v_proj, v, atol=1e-6)


def test_project_tangent_boundary_removes_normal(manifold: BlochRelaxationManifold) -> None:
    """At T1 ≈ T2 the tangent direction (0, -1, +1) must be removed."""
    theta = torch.tensor([[1.0, 100.0, 99.9995]])  # T1 - T2 ≈ 5e-4 < eps
    v_outward = torch.tensor([[0.0, -1.0, 1.0]])  # purely violating direction
    v_proj = manifold.project_tangent(theta, v_outward)
    assert v_proj.abs().sum().item() < 1e-5  # normal component fully removed


def test_exp_map_never_leaves_manifold(manifold: BlochRelaxationManifold) -> None:
    """Random geodesics from interior must remain in M for any step count."""
    torch.manual_seed(0)
    theta0 = torch.tensor([[1.0, 1200.0, 80.0]])
    for _ in range(8):
        v = torch.randn(1, 3) * 10.0  # generous velocity
        theta_next = manifold.exp_map(theta0, v, n_steps=8)
        m0, t1, t2 = theta_next[0]
        assert m0.item() > 0
        assert t1.item() > 0
        assert t2.item() > 0
        assert t2.item() <= t1.item() + 1e-3


def test_geodesic_distance_is_nonnegative(manifold: BlochRelaxationManifold) -> None:
    a = torch.tensor([[1.0, 1200.0, 80.0]])
    b = torch.tensor([[1.0, 800.0, 60.0]])
    d = manifold.geodesic_distance(a, b)
    assert d.item() >= 0


def test_geodesic_distance_zero_at_identity(manifold: BlochRelaxationManifold) -> None:
    a = torch.tensor([[1.0, 1200.0, 80.0]])
    d = manifold.geodesic_distance(a, a)
    assert d.item() == pytest.approx(0.0, abs=1e-6)
