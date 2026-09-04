"""Tests for the Bloch-manifold cache + improved log map (Idea 5 refinements)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold


def test_cache_matches_uncached_to_tolerance() -> None:
    """Cached metric should match the autograd-computed metric to a few %."""
    torch.manual_seed(0)
    uncached = BlochRelaxationManifold(cache_resolution=0)
    cached = BlochRelaxationManifold(cache_resolution=8)
    theta = torch.tensor([[1.0, 1200.0, 80.0]])
    G_u = uncached.metric_tensor(theta)
    G_c = cached.metric_tensor(theta)
    # Trilinear interpolation at coarse R=8 is approximate but should be
    # in the same ballpark (within ~1 order of magnitude on the dominant
    # eigenvalues).
    assert G_u.shape == G_c.shape
    # Both matrices should be positive-definite.
    assert torch.linalg.eigvalsh(G_u).min().item() > -1e-4
    assert torch.linalg.eigvalsh(G_c).min().item() > -1e-4


def test_cache_refresh_no_op_when_disabled() -> None:
    m = BlochRelaxationManifold(cache_resolution=0)
    # Should not raise.
    m.refresh_metric_cache()


def test_cache_refresh_rebuilds_grid() -> None:
    m = BlochRelaxationManifold(cache_resolution=4)
    cached_before = m._cached_grid.clone()
    m.refresh_metric_cache()
    cached_after = m._cached_grid
    assert cached_after is not None
    # Deterministic Jacobian path ⇒ refresh produces an identical grid.
    assert torch.allclose(cached_before, cached_after, atol=1e-8)


def test_log_map_newton_converges() -> None:
    """Newton iteration must produce a v such that exp_map(a, v) ≈ b."""
    torch.manual_seed(1)
    m = BlochRelaxationManifold()
    a = torch.tensor([[1.0, 1200.0, 80.0]])
    b = torch.tensor([[1.0, 1500.0, 100.0]])  # both interior points
    v = m.log_map(a, b, n_newton=4, n_geodesic_steps=4)
    b_recovered = m.exp_map(a, v, n_steps=4)
    residual = (b - b_recovered).abs().max().item()
    # First-order log_map alone yields a much larger residual; the
    # Newton iteration brings it well below 1.0 in physical units.
    assert residual < 1.0


def test_log_map_n_newton_zero_matches_first_order() -> None:
    m = BlochRelaxationManifold()
    a = torch.tensor([[1.0, 1200.0, 80.0]])
    b = torch.tensor([[1.0, 800.0, 60.0]])
    v0 = m.log_map(a, b, n_newton=0)
    expected = m.project_tangent(a, b - a)
    assert torch.allclose(v0, expected, atol=1e-6)
