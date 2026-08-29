"""Tests for the primary-decomposition vanishing-ideal primitive."""

from __future__ import annotations

import torch

from mriforge.infrastructure.physics.primary_ideal import (
    quadratic_component,
    vanishing_ideal_residual,
)


def test_quadratic_component_zero_on_shell():
    center = torch.tensor([0.0, 0.0, 0.0])
    # a point at distance exactly radius from the centre lies on g_k = 0.
    on_shell = torch.tensor([[1.0, 0.0, 0.0]])
    g = quadratic_component(on_shell, center, radius=1.0)
    assert torch.allclose(g, torch.zeros(1), atol=1e-6)


def test_vanishing_ideal_residual_zero_on_union():
    centers = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    radii = torch.tensor([1.0, 1.0])
    on_first_shell = torch.tensor([[1.0, 0.0, 0.0]])  # on V(g_0)
    res = vanishing_ideal_residual(on_first_shell, centers, radii)
    assert float(res.item()) < 1e-3


def test_vanishing_ideal_residual_positive_off_union():
    centers = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    radii = torch.tensor([1.0, 1.0])
    off = torch.tensor([[2.5, 2.5, 2.5]])  # off both shells
    res = vanishing_ideal_residual(off, centers, radii)
    assert float(res.item()) > 1.0


def test_residual_count_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        vanishing_ideal_residual(
            torch.zeros(1, 3), torch.zeros(2, 3), torch.zeros(3)
        )
