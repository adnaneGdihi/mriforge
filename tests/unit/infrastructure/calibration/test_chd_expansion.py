"""Unit tests for ``mriforge.infrastructure.calibration.chd``.

CHD in this module stands for **Calibrated Hallucination Detector** —
it is *not* a distance function. The detector is a threshold-based
classifier whose threshold is the empirical ``(1 - β)``-quantile of
detector scores on non-hallucinated calibration pixels, with a
Dvoretzky–Kiefer–Wolfowitz slack term for the FPR upper bound.

Covers:

* :func:`dkw_slack` — monotonicity in ``n``, validation, range,
* :class:`CalibratedHallucinationDetector` — construction
  validation, ``fit`` / ``predict`` round-trip, the ``(1-β)``
  quantile property (≤ β fraction of calibration scores end up
  flagged), threshold / FPR-bound accessor errors before ``fit``,
  ``bound_is_tight`` audit gate, empirical FPR on a fresh split.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.calibration.chd import (
    CalibratedHallucinationDetector,
    dkw_slack,
)


# ---------------------------------------------------------------------------
# dkw_slack
# ---------------------------------------------------------------------------


class TestDkwSlack:
    """``slack = sqrt(ln(2/δ) / (2 n))``."""

    def test_known_value_matches_closed_form(self) -> None:
        n, delta = 1000, 0.05
        expected = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
        assert dkw_slack(n, delta) == pytest.approx(expected)

    def test_monotone_decreasing_in_n(self) -> None:
        values = [dkw_slack(n, 0.05) for n in (100, 1_000, 10_000, 100_000)]
        for prev, curr in zip(values, values[1:]):
            assert curr < prev, f"slack must shrink as n grows; got {values}"

    def test_monotone_decreasing_in_delta(self) -> None:
        """Smaller δ ⇒ tighter confidence ⇒ wider slack."""
        loose = dkw_slack(1000, 0.5)
        medium = dkw_slack(1000, 0.05)
        tight = dkw_slack(1000, 0.001)
        assert loose < medium < tight

    def test_positive(self) -> None:
        assert dkw_slack(100, 0.05) > 0

    @pytest.mark.parametrize("bad_n", [-1, 0])
    def test_non_positive_n_raises(self, bad_n: int) -> None:
        with pytest.raises(ValueError, match="n must"):
            dkw_slack(bad_n, 0.05)

    @pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.1, 1.5])
    def test_delta_outside_open_interval_raises(self, bad_delta: float) -> None:
        with pytest.raises(ValueError, match="delta"):
            dkw_slack(100, bad_delta)


# ---------------------------------------------------------------------------
# CalibratedHallucinationDetector — construction
# ---------------------------------------------------------------------------


class TestCalibratedHallucinationDetectorConstruction:
    @pytest.mark.parametrize("bad_beta", [0.0, 1.0, -0.1, 1.5])
    def test_beta_outside_open_interval_raises(self, bad_beta: float) -> None:
        with pytest.raises(ValueError, match="beta"):
            CalibratedHallucinationDetector(beta=bad_beta)

    @pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.1, 1.5])
    def test_delta_outside_open_interval_raises(self, bad_delta: float) -> None:
        with pytest.raises(ValueError, match="delta"):
            CalibratedHallucinationDetector(beta=0.05, delta=bad_delta)

    def test_initial_state_is_unfitted(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05, delta=1e-3)
        assert chd.is_fitted is False
        assert chd.beta == pytest.approx(0.05)
        assert chd.delta == pytest.approx(1e-3)
        assert chd.n_calibration == 0

    def test_threshold_before_fit_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = chd.threshold

    def test_fpr_upper_bound_before_fit_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = chd.fpr_upper_bound

    def test_predict_before_fit_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        with pytest.raises(RuntimeError, match="not been fitted"):
            chd.predict(torch.tensor([0.0, 1.0]))


# ---------------------------------------------------------------------------
# CalibratedHallucinationDetector — fit
# ---------------------------------------------------------------------------


class TestCalibratedHallucinationDetectorFit:
    def test_fit_returns_self_for_fluent_chain(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        scores = torch.linspace(0.0, 1.0, steps=1000)
        result = chd.fit(scores)
        assert result is chd

    def test_fit_records_calibration_size(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        scores = torch.linspace(0.0, 1.0, steps=500)
        chd.fit(scores)
        assert chd.n_calibration == 500
        assert chd.is_fitted is True

    def test_fit_flattens_multi_dim_input(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        scores = torch.linspace(0.0, 1.0, steps=400).reshape(20, 20)
        chd.fit(scores)
        assert chd.n_calibration == 400

    def test_fit_threshold_matches_quantile(self) -> None:
        """Threshold equals the ``(1-β)``-quantile of the calibration scores."""
        chd = CalibratedHallucinationDetector(beta=0.10)
        scores = torch.linspace(0.0, 1.0, steps=1001, dtype=torch.float64)
        chd.fit(scores)
        expected = float(torch.quantile(scores.double(), 0.90).item())
        assert chd.threshold == pytest.approx(expected)

    def test_zero_dim_scores_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        with pytest.raises(ValueError, match="at least 1-D"):
            chd.fit(torch.tensor(0.5))

    def test_empty_calibration_set_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        with pytest.raises(ValueError, match="empty"):
            chd.fit(torch.empty(0))


# ---------------------------------------------------------------------------
# CalibratedHallucinationDetector — predict / FPR bound
# ---------------------------------------------------------------------------


class TestCalibratedHallucinationDetectorPredict:
    def test_predict_returns_bool_mask(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        chd.fit(torch.linspace(0.0, 1.0, steps=1000))
        mask = chd.predict(torch.tensor([0.0, 0.5, 0.95, 1.0]))
        assert mask.dtype == torch.bool
        assert mask.shape == (4,)

    def test_predict_preserves_input_shape(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        chd.fit(torch.linspace(0.0, 1.0, steps=1000))
        mask = chd.predict(torch.zeros(3, 4, 5))
        assert mask.shape == (3, 4, 5)

    def test_predict_flags_only_scores_strictly_above_threshold(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        chd.fit(torch.linspace(0.0, 1.0, steps=1000))
        tau = chd.threshold
        # By definition: ``predict = scores > τ``.
        scores = torch.tensor([tau - 0.1, tau, tau + 1e-3])
        mask = chd.predict(scores)
        assert bool(mask[0].item()) is False
        assert bool(mask[1].item()) is False  # strict inequality
        assert bool(mask[2].item()) is True

    def test_at_most_beta_fraction_of_calibration_flagged(self) -> None:
        """The calibration scores themselves should have ≤ β fraction above τ."""
        torch.manual_seed(42)
        beta = 0.05
        chd = CalibratedHallucinationDetector(beta=beta)
        scores = torch.randn(10_000)
        chd.fit(scores)
        flagged_fraction = float(chd.predict(scores).float().mean().item())
        # The empirical (1 - β)-quantile guarantees at most β of points
        # are strictly above (allow a tiny float tolerance).
        assert flagged_fraction <= beta + 1e-3

    def test_fpr_upper_bound_equals_beta_plus_dkw(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05, delta=1e-3)
        n = 5_000
        chd.fit(torch.randn(n))
        expected = 0.05 + dkw_slack(n, 1e-3)
        assert chd.fpr_upper_bound == pytest.approx(expected)

    def test_fpr_upper_bound_tightens_with_calibration_size(self) -> None:
        bounds = []
        for n in (200, 2_000, 20_000):
            chd = CalibratedHallucinationDetector(beta=0.05, delta=1e-3)
            chd.fit(torch.randn(n))
            bounds.append(chd.fpr_upper_bound)
        for prev, curr in zip(bounds, bounds[1:]):
            assert curr < prev, f"FPR bound must tighten with n; got {bounds}"

    def test_bound_is_tight_audit_gate(self) -> None:
        """The gate is ``DKW slack ≤ β``."""
        # Large n ⇒ slack ≪ β ⇒ tight.
        tight = CalibratedHallucinationDetector(beta=0.1, delta=1e-3)
        tight.fit(torch.randn(100_000))
        assert tight.bound_is_tight is True

        # Tiny n + tight δ ⇒ slack ≫ β ⇒ NOT tight.
        loose = CalibratedHallucinationDetector(beta=0.05, delta=1e-3)
        loose.fit(torch.randn(10))
        assert loose.bound_is_tight is False

    def test_empirical_fpr_on_fresh_split(self) -> None:
        """Fitting on one draw and evaluating on a fresh draw ⇒ empirical FPR ≈ β."""
        torch.manual_seed(123)
        beta = 0.05
        chd = CalibratedHallucinationDetector(beta=beta, delta=1e-3)
        chd.fit(torch.randn(20_000))
        fresh = torch.randn(20_000)
        emp_fpr = chd.empirical_fpr(fresh)
        # Concentration: empirical FPR within DKW slack of β.
        slack = dkw_slack(20_000, 1e-3)
        assert abs(emp_fpr - beta) <= slack + 1e-3

    def test_empirical_fpr_returns_python_float(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        chd.fit(torch.linspace(0.0, 1.0, steps=1000))
        out = chd.empirical_fpr(torch.linspace(0.0, 1.0, steps=1000))
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0

    def test_threshold_is_python_float(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.05)
        chd.fit(torch.linspace(0.0, 1.0, steps=100))
        assert isinstance(chd.threshold, float)

    def test_quantile_at_extreme_beta_handled(self) -> None:
        """Very small β ⇒ (1-β) close to 1 ⇒ threshold near calibration max."""
        chd = CalibratedHallucinationDetector(beta=0.001)
        scores = torch.linspace(0.0, 1.0, steps=10_001)
        chd.fit(scores)
        assert chd.threshold > 0.99


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_calibration_set_yields_same_threshold(
        self, deterministic_seed: None
    ) -> None:
        scores = torch.randn(5_000)
        a = CalibratedHallucinationDetector(beta=0.05).fit(scores.clone())
        b = CalibratedHallucinationDetector(beta=0.05).fit(scores.clone())
        assert a.threshold == b.threshold
        assert a.n_calibration == b.n_calibration
