"""Manifold diagnostics (C2/C3 estimators + the certified error bound).

The reach estimator is pinned to the one geometry with a closed form — on a
circle of radius R every chord satisfies ``||y-p||^2 / (2 ||(y-p)_norm||) = R``
exactly, so the plug-in min recovers R up to tangent-estimation error. The
bound is checked for its exact value and monotonicity in all four arguments.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mriforge.core.metrics.manifold_diagnostics import (
    certified_error_bound,
    estimate_reach,
    manifold_departure,
    step_budget_ratio,
    tangential_defect,
)


def _circle(radius: float, n: int = 200) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


class TestEstimateReach:
    def test_circle_reach_is_its_radius(self):
        for radius in (1.0, 2.5):
            tau = estimate_reach(_circle(radius), intrinsic_dim=1)
            assert tau == pytest.approx(radius, rel=0.05)

    def test_flat_line_has_infinite_reach(self):
        """An affine cloud has no normal excursion: reach must be inf, not an error."""
        points = np.stack([np.linspace(0.0, 1.0, 50), np.zeros(50)], axis=1)
        assert estimate_reach(points, intrinsic_dim=1) == math.inf

    def test_smaller_circle_has_smaller_reach(self):
        assert estimate_reach(_circle(0.5)) < estimate_reach(_circle(2.0))

    def test_validation_raises(self):
        pts = _circle(1.0, n=30)
        with pytest.raises(ValueError, match="intrinsic_dim"):
            estimate_reach(pts, intrinsic_dim=2)  # == ambient dim
        with pytest.raises(ValueError, match="k_neighbors"):
            estimate_reach(pts, intrinsic_dim=1, k_neighbors=1)
        with pytest.raises(ValueError, match="2-D"):
            estimate_reach(np.zeros(5), intrinsic_dim=1)


class TestStepBudgetRatio:
    def test_c2_compliant_level(self):
        verdict = step_budget_ratio(1.0, 2.5)
        assert verdict["kappa"] == pytest.approx(0.4)
        assert verdict["well_posed"] is True
        assert verdict["satisfies_c2"] is True
        assert verdict["amplification"] == pytest.approx(1.0 / 0.6)

    def test_c2_boundary_is_inclusive(self):
        """kappa = 1/2 exactly is C2's cap (Lambda = 2)."""
        verdict = step_budget_ratio(1.0, 2.0)
        assert verdict["satisfies_c2"] is True
        assert verdict["amplification"] == pytest.approx(2.0)

    def test_over_cap_but_well_posed(self):
        verdict = step_budget_ratio(0.8, 1.0)
        assert verdict["satisfies_c2"] is False
        assert verdict["well_posed"] is True

    def test_ill_posed_level(self):
        verdict = step_budget_ratio(3.0, 2.5)
        assert verdict["well_posed"] is False
        assert verdict["amplification"] == math.inf

    def test_validation_raises(self):
        with pytest.raises(ValueError, match="tau_hat"):
            step_budget_ratio(1.0, 0.0)
        with pytest.raises(ValueError, match="delta_t"):
            step_budget_ratio(-1.0, 2.0)


class TestTangentialDefect:
    @staticmethod
    def _line_cloud(n: int = 40) -> np.ndarray:
        return np.stack([np.linspace(0.0, 1.0, n), np.zeros(n)], axis=1)

    def test_normal_displacements_have_zero_defect(self):
        pts = self._line_cloud()
        disp = np.stack([np.zeros(len(pts)), np.full(len(pts), 0.3)], axis=1)
        assert tangential_defect(pts, disp) == pytest.approx(0.0, abs=1e-9)

    def test_tangential_displacements_have_defect_one(self):
        pts = self._line_cloud()
        disp = np.stack([np.full(len(pts), 0.3), np.zeros(len(pts))], axis=1)
        assert tangential_defect(pts, disp) == pytest.approx(1.0)

    def test_diagonal_displacement_splits_evenly(self):
        pts = self._line_cloud()
        disp = np.full((len(pts), 2), 0.3)  # 45 degrees to the tangent
        assert tangential_defect(pts, disp) == pytest.approx(math.sqrt(0.5), rel=1e-6)

    def test_supremum_not_mean(self):
        """One tangential displacement among normals must set theta_hat alone."""
        pts = self._line_cloud()
        disp = np.stack([np.zeros(len(pts)), np.full(len(pts), 0.3)], axis=1)
        disp[7] = [0.3, 0.0]
        assert tangential_defect(pts, disp) == pytest.approx(1.0)

    def test_all_zero_displacements_are_defect_free(self):
        pts = self._line_cloud()
        assert tangential_defect(pts, np.zeros_like(pts)) == 0.0

    def test_validation_raises(self):
        pts = self._line_cloud()
        with pytest.raises(ValueError, match="align"):
            tangential_defect(pts, np.zeros((3, 2)))


class TestManifoldDeparture:
    def test_member_has_zero_departure(self):
        refs = _circle(1.0, n=16)
        assert manifold_departure(refs[3], refs) == pytest.approx(0.0)

    def test_planted_distance(self):
        refs = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert manifold_departure(np.array([1.0, 0.5]), refs) == pytest.approx(0.5)

    def test_validation_raises(self):
        with pytest.raises(ValueError, match="ambient"):
            manifold_departure(np.zeros(3), np.zeros((4, 2)))
        with pytest.raises(ValueError, match="at least one"):
            manifold_departure(np.zeros(2), np.zeros((0, 2)))


class TestCertifiedErrorBound:
    def test_exact_value(self):
        """mu + omega (kappa + L mu) = 0.1 + 2 (0.05 + 3 * 0.1) = 0.8"""
        assert certified_error_bound(0.1, 0.05, 2.0, L_T=3.0) == pytest.approx(0.8)

    def test_monotone_in_every_argument(self):
        base = certified_error_bound(0.1, 0.05, 2.0, L_T=1.0)
        assert certified_error_bound(0.2, 0.05, 2.0, L_T=1.0) > base
        assert certified_error_bound(0.1, 0.10, 2.0, L_T=1.0) > base
        assert certified_error_bound(0.1, 0.05, 3.0, L_T=1.0) > base
        assert certified_error_bound(0.1, 0.05, 2.0, L_T=2.0) > base

    def test_perfect_reconstruction_bound_is_zero(self):
        assert certified_error_bound(0.0, 0.0, 5.0) == 0.0

    def test_validation_raises(self):
        with pytest.raises(ValueError, match="mu_hat"):
            certified_error_bound(-0.1, 0.0, 1.0)
        with pytest.raises(ValueError, match="omega_T"):
            certified_error_bound(0.1, 0.0, -1.0)
