"""Tests for persistent-homology MRF features."""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.quantitative.persistent_homology_mrf import (
    MRFParameterStabilityMetric,
    PersistenceDiameterMetric,
    TopologicalMaskCertificate,
    kspace_sampling_pattern_h0_certificate,
    persistence_diameter,
    persistence_lifetimes,
    topological_completeness_certificate,
    vietoris_rips_h0_diagram,
)


def test_h0_diagram_recovers_mst_edges() -> None:
    """For points on a line at integer positions, MST edge weights are 1."""
    pts = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    diag = vietoris_rips_h0_diagram(pts)
    assert diag.shape == (4, 2)
    assert torch.allclose(diag[:, 1], torch.ones(4))


def test_rotation_invariance() -> None:
    """SO(3) rotations preserve pairwise distances → diagrams are identical."""
    torch.manual_seed(0)
    pts = torch.randn(20, 3)
    # Random rotation matrix via QR of a Gaussian.
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    rotated = pts @ Q.T
    diag_a = vietoris_rips_h0_diagram(pts)[:, 1]
    diag_b = vietoris_rips_h0_diagram(rotated)[:, 1]
    assert torch.allclose(diag_a, diag_b, atol=1e-5)


def test_persistence_diameter_lipschitz_in_m0() -> None:
    """diam(α · Γ) = α · diam(Γ) → linear in M0 scale."""
    torch.manual_seed(1)
    base = torch.randn(50, 3)
    base = base / base.norm(dim=-1, keepdim=True)  # on the sphere
    for alpha in (0.5, 1.0, 2.0, 5.0):
        d = persistence_diameter(alpha * base)
        d_ref = persistence_diameter(base)
        assert d.item() == pytest.approx(alpha * d_ref.item(), rel=1e-4)


def test_diameter_metric_handles_batched_trajectory() -> None:
    torch.manual_seed(2)
    batched = torch.randn(4, 30, 3)
    metric = PersistenceDiameterMetric()
    value = metric(batched)
    assert isinstance(value, float)
    assert value > 0.0


def test_stability_metric_zero_on_equal_diagrams() -> None:
    metric = MRFParameterStabilityMetric()
    pts = torch.randn(15, 3)
    assert metric(pts, pts) == pytest.approx(0.0, abs=1e-6)


def test_stability_metric_bottleneck_bound() -> None:
    """Stability theorem: bottleneck ≤ L^∞ perturbation."""
    torch.manual_seed(3)
    pts = torch.randn(20, 3)
    perturbation = 0.01 * torch.randn_like(pts)
    metric = MRFParameterStabilityMetric()
    bd = metric(pts + perturbation, pts)
    assert bd <= 2.0 * perturbation.norm(dim=-1).max().item() + 1e-4


def test_lifetimes_match_diagram_deaths() -> None:
    torch.manual_seed(4)
    pts = torch.randn(10, 2)
    diag = vietoris_rips_h0_diagram(pts)
    lifetimes = persistence_lifetimes(pts)
    assert torch.allclose(lifetimes, diag[:, 1])


def test_metric_invalid_shape_raises() -> None:
    metric = PersistenceDiameterMetric()
    with pytest.raises(ValueError):
        metric(torch.zeros(5))


# ---------- Topological mask certification (Batch-2 #6) ----------------------


def test_sampling_pattern_certificate_zero_for_empty_mask() -> None:
    """Empty (or single-point) masks return 0 by convention."""
    mask = torch.zeros(16, 16, dtype=torch.bool)
    out = kspace_sampling_pattern_h0_certificate(mask)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_sampling_pattern_certificate_for_uniform_mask() -> None:
    """A uniform regular Cartesian mask has small MST edge weight (≈1)."""
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[::1, ::4] = True  # every 4th column
    cert = kspace_sampling_pattern_h0_certificate(mask).item()
    assert cert > 0.0
    # All adjacent samples are at distance 1 (along columns), so max MST
    # edge weight ≈ 4 (the column gap).
    assert cert <= 4.5


def test_tcc_zero_on_fully_sampled_mask() -> None:
    """A fully-sampled mask is topologically complete (TCC = 0)."""
    mask = torch.ones(16, 16, dtype=torch.bool)
    cert = topological_completeness_certificate(mask, fov_pixels=16).item()
    assert cert == 0.0


def test_tcc_positive_on_extremely_sparse_mask() -> None:
    """Two faraway points produce a positive TCC at small Nyquist scale."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[5, 5] = True
    mask[55, 55] = True
    cert = topological_completeness_certificate(mask, fov_pixels=64).item()
    # MST edge weight ≈ √(50² + 50²) > 16 = fov/4, so 1 class survives.
    assert cert >= 1.0


def test_topological_mask_metric_handles_2d_and_3d_input() -> None:
    metric = TopologicalMaskCertificate(fov_pixels=16)
    mask_2d = torch.ones(16, 16, dtype=torch.bool)
    mask_3d = torch.ones(3, 16, 16, dtype=torch.bool)
    assert metric(mask_2d) == pytest.approx(0.0, abs=1e-6)
    assert metric(mask_3d) == pytest.approx(0.0, abs=1e-6)


def test_topological_mask_metric_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        TopologicalMaskCertificate(fov_pixels=0)
    with pytest.raises(ValueError):
        TopologicalMaskCertificate(fov_pixels=16)(torch.zeros(5))
