"""Tests for the per-pixel Bernoulli Fisher-Rao geometry primitive (B-3.4)."""

from __future__ import annotations

import math

import torch

from spectramr.models.blocks.fisher_rao_sphere import (
    angle_to_intensity,
    fisher_rao_geodesic_distance,
    fisher_rao_geodesic_step,
    intensity_to_angle,
)


def test_embed_unembed_inverse() -> None:
    x = torch.rand(2, 1, 8, 8)
    assert torch.allclose(angle_to_intensity(intensity_to_angle(x)), x, atol=1e-5)


def test_geodesic_step_identity_at_zero() -> None:
    x = torch.rand(2, 1, 8, 8)
    assert torch.allclose(fisher_rao_geodesic_step(x, torch.zeros_like(x)), x, atol=1e-5)


def test_geodesic_step_bounded_by_construction() -> None:
    # THE distinctive Fisher-Rao property: cos^2 keeps the output in [0,1] for ANY angular
    # velocity (norm-preserving), unlike a Euclidean additive update.
    x = torch.rand(4, 1, 8, 8)
    out = fisher_rao_geodesic_step(x, torch.full_like(x, 100.0))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_geodesic_distance_properties() -> None:
    x = torch.rand(3, 1, 8, 8)
    assert torch.allclose(fisher_rao_geodesic_distance(x, x), torch.zeros_like(x), atol=1e-4)
    # antipodal Bernoullis (0 vs 1) are pi/2 apart (max great-circle distance)
    d = float(fisher_rao_geodesic_distance(torch.tensor(0.0), torch.tensor(1.0)))
    assert abs(d - math.pi / 2) < 1e-2


def test_geodesic_shooting_reaches_target_exactly() -> None:
    # Shooting from x_s by the full angular gap reaches x_t (the geodesic IS the slerp).
    x_s, x_t = torch.tensor(0.2), torch.tensor(0.8)
    delta = intensity_to_angle(x_t) - intensity_to_angle(x_s)
    assert torch.allclose(fisher_rao_geodesic_step(x_s, delta), x_t, atol=1e-5)


def test_velocity_gradient_vanishes_at_boundaries() -> None:
    # DISCLOSED CAVEAT (review): d(step)/d(delta) = -2*sqrt(x(1-x)) vanishes at the intensity
    # boundaries (x->0 background, x->1 bright tissue) -> ~weak learning signal there; full at
    # midtone. Documents the trainability/ablation-conditioning asymmetry vs the Euclidean arm.
    for xv in (0.0, 1.0):
        x = torch.tensor([[[[xv]]]], dtype=torch.float64)
        d = torch.zeros_like(x, requires_grad=True)
        (g,) = torch.autograd.grad(fisher_rao_geodesic_step(x, d).sum(), d)
        assert abs(float(g)) < 1e-2  # boundary: vanishes (eps-clamped escape ~2e-3)
    x = torch.tensor([[[[0.5]]]], dtype=torch.float64)
    d = torch.zeros_like(x, requires_grad=True)
    (g,) = torch.autograd.grad(fisher_rao_geodesic_step(x, d).sum(), d)
    assert abs(float(g)) > 0.9  # midtone: full signal
