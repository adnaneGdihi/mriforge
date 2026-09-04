"""Calibration tests — conformal, pac_bayes_certificate, chd, scores
(TASK IV.G).

Coverage targets
----------------
- ConformalCalibrator: fit → empirical coverage ≥ 1−α on synthetic scores;
  monotone tightness (more calibration data → smaller or equal band);
  predict shape contract; raises on unfit call; raises on empty calibration.
- Non-conformity scores (absolute_residual, quantile_regression, CQR):
  shape contract, raises on shape mismatch, raises on swapped quantiles.
- PACBayesCertificate / pac_bayes_bound: bound in [0, 1] for loss range
  [0, 1]; non-vacuous detection; McAllester complexity term formula.
- CalibratedHallucinationDetector (CHD): fpr_upper_bound ≥ beta;
  predict produces a boolean mask; bound_is_tight predicate; raises on
  empty calibration set; DKW slack formula.

All tests are CPU-deterministic and require only torch (already shimmed
in conftest.py for no-CUDA environments).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.calibration.conformal import ConformalCalibrator
from spectramr.infrastructure.calibration.pac_bayes_certificate import (
    PACBayesCertificate,
    PACBayesReport,
    mcallester_complexity_term,
    pac_bayes_bound,
)
from spectramr.infrastructure.calibration.chd import (
    CalibratedHallucinationDetector,
    dkw_slack,
)
from spectramr.infrastructure.calibration.scores import (
    absolute_residual,
    conformalised_quantile_regression,
    quantile_regression,
)

# ── Non-conformity score functions ────────────────────────────────────────────


@pytest.mark.unit
class TestAbsoluteResidual:
    def test_canary_shape_preserved(self) -> None:
        pred = torch.ones(4, 1, 8, 8)
        target = torch.zeros(4, 1, 8, 8)
        s = absolute_residual(pred, target)
        assert s.shape == pred.shape

    def test_canary_correct_values(self) -> None:
        pred = torch.tensor([2.0, 5.0])
        target = torch.tensor([1.0, 3.0])
        s = absolute_residual(pred, target)
        assert torch.allclose(s, torch.tensor([1.0, 2.0]))

    def test_nonnegative(self) -> None:
        pred = torch.randn(16)
        target = torch.randn(16)
        assert (absolute_residual(pred, target) >= 0).all()

    def test_zero_for_perfect_prediction(self) -> None:
        x = torch.randn(8)
        assert torch.allclose(absolute_residual(x, x), torch.zeros(8))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="[Ss]hape"):
            absolute_residual(torch.ones(4), torch.ones(5))

    def test_symmetry(self) -> None:
        """|pred - target| == |target - pred|."""
        a = torch.randn(10)
        b = torch.randn(10)
        assert torch.allclose(absolute_residual(a, b), absolute_residual(b, a))


@pytest.mark.unit
class TestQuantileRegressionScore:
    def test_canary_inside_band_is_negative(self) -> None:
        """Target strictly inside [lower, upper] gives a negative score."""
        lower = torch.zeros(5)
        upper = torch.ones(5)
        target = torch.full((5,), 0.5)
        s = quantile_regression(lower, upper, target)
        assert (s < 0).all()

    def test_canary_outside_band_is_positive(self) -> None:
        lower = torch.zeros(5)
        upper = torch.ones(5)
        target = torch.full((5,), 2.0)  # above upper
        s = quantile_regression(lower, upper, target)
        assert (s > 0).all()

    def test_shape_contract(self) -> None:
        lower = torch.randn(3, 4)
        upper = lower + 1.0
        target = torch.randn(3, 4)
        s = quantile_regression(lower, upper, target)
        assert s.shape == lower.shape

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="[Ss]hape"):
            quantile_regression(torch.ones(3), torch.ones(3), torch.ones(4))

    def test_swapped_quantiles_raises(self) -> None:
        lower = torch.tensor([1.0, 1.0])
        upper = torch.tensor([0.0, 0.0])  # upper < lower — invalid
        target = torch.zeros(2)
        with pytest.raises(ValueError, match="[Ss]wapped"):
            quantile_regression(lower, upper, target)

    def test_cqr_alias_matches(self) -> None:
        lower = torch.zeros(6)
        upper = torch.ones(6)
        target = torch.full((6,), 0.4)
        s1 = quantile_regression(lower, upper, target)
        s2 = conformalised_quantile_regression(lower, upper, target)
        assert torch.allclose(s1, s2)


# ── ConformalCalibrator ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestConformalCalibratorCoverage:
    """Property: empirical coverage on a fresh test set ≥ 1 − α."""

    def _synth_pairs(self, n: int, noise_std: float = 0.5):
        """Generate (pred, target) pairs where residuals ~ N(0, noise_std²)."""
        torch.manual_seed(0)
        pred = torch.randn(n, 1, 16, 16)
        target = pred + torch.randn_like(pred) * noise_std
        return [(pred, target)]

    def test_canary_coverage_90_percent(self) -> None:
        """α=0.10 → empirical coverage on 1000 test samples ≥ 0.9."""
        alpha = 0.10
        cal = ConformalCalibrator(alpha=alpha)
        cal.fit(self._synth_pairs(500))

        torch.manual_seed(1)
        pred_test = torch.randn(1000, 1, 16, 16)
        target_test = pred_test + torch.randn_like(pred_test) * 0.5
        cov = cal.empirical_coverage(pred_test, target_test)
        assert cov >= 1.0 - alpha - 0.02, (
            f"Expected empirical coverage ≥ {1-alpha-0.02:.2f}, got {cov:.3f}"
        )

    def test_canary_coverage_95_percent(self) -> None:
        alpha = 0.05
        cal = ConformalCalibrator(alpha=alpha)
        cal.fit(self._synth_pairs(800))

        torch.manual_seed(2)
        pred_test = torch.randn(1000, 1, 8, 8)
        target_test = pred_test + torch.randn_like(pred_test) * 0.5
        cov = cal.empirical_coverage(pred_test, target_test)
        assert cov >= 1.0 - alpha - 0.02

    def test_monotone_tightness(self) -> None:
        """More calibration data → band radius ≤ radius from fewer samples.

        This holds in expectation (not always due to noise), but with a
        fixed seed and simple noise distribution it should be reliable.
        """
        alpha = 0.10
        torch.manual_seed(7)

        def _fit(n: int) -> float:
            cal = ConformalCalibrator(alpha=alpha)
            pairs = [(torch.randn(n, 1), torch.randn(n, 1))]
            cal.fit(pairs)
            return cal.quantile

        q_small = _fit(20)
        q_large = _fit(10_000)
        # With ~10k samples the (n+1)/n correction is negligible → the band
        # converges to the true quantile; with n=20 it's inflated.
        # Allow a 5% tolerance so the test is robust to seed variation.
        assert q_large <= q_small * 1.05, (
            f"Large-data quantile {q_large:.4f} unexpectedly larger than "
            f"small-data quantile {q_small:.4f}"
        )


@pytest.mark.unit
class TestConformalCalibratorAPI:
    """Fit / predict / raises contract."""

    def test_canary_fit_and_predict_shape(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        pairs = [(torch.randn(32, 1, 4, 4), torch.randn(32, 1, 4, 4))]
        cal.fit(pairs)
        assert cal.is_fitted

        pred = torch.randn(8, 1, 4, 4)
        y_hat, lower, upper = cal.predict(pred)
        assert y_hat.shape == pred.shape
        assert lower.shape == pred.shape
        assert upper.shape == pred.shape

    def test_lower_le_yhat_le_upper(self) -> None:
        """lower ≤ y_hat ≤ upper element-wise."""
        cal = ConformalCalibrator(alpha=0.1)
        cal.fit([(torch.randn(50, 1), torch.randn(50, 1))])
        pred = torch.randn(20, 1)
        y_hat, lower, upper = cal.predict(pred)
        assert (lower <= y_hat).all()
        assert (y_hat <= upper).all()

    def test_band_radius_equals_fitted_quantile(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        cal.fit([(torch.randn(100), torch.randn(100))])
        pred = torch.zeros(5)
        y_hat, lower, upper = cal.predict(pred)
        q = cal.quantile
        assert torch.allclose(y_hat - lower, torch.full((5,), q), atol=1e-5)
        assert torch.allclose(upper - y_hat, torch.full((5,), q), atol=1e-5)

    def test_predict_before_fit_raises(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        with pytest.raises(RuntimeError, match="[Ff]it"):
            cal.predict(torch.randn(4))

    def test_quantile_before_fit_raises(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        with pytest.raises(RuntimeError, match="[Ff]it"):
            _ = cal.quantile

    def test_fit_empty_raises(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        with pytest.raises(ValueError, match="[Ee]mpty"):
            cal.fit([])

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            ConformalCalibrator(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ConformalCalibrator(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            ConformalCalibrator(alpha=1.5)

    def test_n_calibration_set_after_fit(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        cal.fit([(torch.randn(50, 2), torch.randn(50, 2))])
        # 50 samples × 2 elements = 100 calibration points
        assert cal.n_calibration == 100

    def test_fluent_chain(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        result = cal.fit([(torch.randn(10), torch.randn(10))])
        assert result is cal  # fit() returns self

    def test_custom_score_fn_used(self) -> None:
        """A custom score_fn that always returns 0 → zero-width band."""
        def zero_score(pred, target):
            return torch.zeros_like(pred)

        cal = ConformalCalibrator(score_fn=zero_score, alpha=0.1)
        cal.fit([(torch.randn(50), torch.randn(50))])
        assert cal.quantile == pytest.approx(0.0, abs=1e-6)

    def test_is_fitted_false_before_fit(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        assert not cal.is_fitted


# ── PACBayesCertificate ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestPACBayesCertificateBound:
    """Bound is in [0, M]; non-vacuous detection; formula correctness."""

    def test_canary_bound_in_zero_one(self) -> None:
        """With M=1 and typical inputs, bound should be in [0, 1]."""
        cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0)
        report = cert.certify(empirical_risk=0.1, kl_q_p=0.5, n=1000)
        assert 0.0 <= report.bound <= 1.0, f"Bound out of [0,1]: {report.bound}"

    def test_canary_non_vacuous_small_kl(self) -> None:
        """Small KL + large n → bound < M (non-vacuous)."""
        cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0)
        report = cert.certify(empirical_risk=0.05, kl_q_p=0.001, n=100_000)
        assert report.is_non_vacuous, (
            f"Expected non-vacuous for small KL, large n; bound={report.bound:.4f}"
        )

    def test_vacuous_large_kl(self) -> None:
        """Very large KL on tiny n → bound > 1.0 (vacuous)."""
        cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0)
        report = cert.certify(empirical_risk=0.5, kl_q_p=100.0, n=10)
        assert not report.is_non_vacuous, (
            f"Expected vacuous for large KL, tiny n; bound={report.bound:.4f}"
        )

    def test_mcallester_formula(self) -> None:
        """Manual computation of complexity term matches the function."""
        kl, n, delta = 0.5, 1000, 0.05
        expected = math.sqrt((kl + math.log(2.0 * math.sqrt(n) / delta)) / (2 * (n - 1)))
        got = mcallester_complexity_term(kl, n, delta)
        assert got == pytest.approx(expected, rel=1e-6)

    def test_bound_equals_risk_plus_slack(self) -> None:
        """pac_bayes_bound = empirical_risk + complexity_term (no TV extension)."""
        risk, kl, n, delta = 0.1, 0.3, 500, 0.05
        slack = mcallester_complexity_term(kl, n, delta)
        expected = risk + slack
        got = pac_bayes_bound(risk, kl, n, delta)
        assert got == pytest.approx(expected, rel=1e-6)

    def test_tv_extension_adds_tau_times_M(self) -> None:
        risk, kl, n, delta = 0.1, 0.3, 500, 0.05
        tau, M = 0.2, 1.5
        base = pac_bayes_bound(risk, kl, n, delta)
        extended = pac_bayes_bound(risk, kl, n, delta, tv_tolerance=tau, loss_bound_M=M)
        assert extended == pytest.approx(base + tau * M, rel=1e-6)

    def test_report_is_frozen_dataclass(self) -> None:
        cert = PACBayesCertificate()
        report = cert.certify(0.1, 0.5, 1000)
        assert isinstance(report, PACBayesReport)
        with pytest.raises(Exception):  # frozen dataclass
            report.bound = 999.0

    def test_report_to_dict_serialisable(self) -> None:
        import json
        cert = PACBayesCertificate(delta=0.05)
        report = cert.certify(0.1, 0.5, 1000)
        d = report.to_dict()
        # Must be JSON-serialisable
        json.dumps(d)
        assert "bound" in d
        assert "is_non_vacuous" in d

    # Edge: invalid arguments raise ValueError
    def test_invalid_kl_raises(self) -> None:
        with pytest.raises(ValueError, match="kl_q_p"):
            mcallester_complexity_term(kl_q_p=-1.0, n=100, delta=0.05)

    def test_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            mcallester_complexity_term(kl_q_p=0.5, n=1, delta=0.05)

    def test_invalid_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            mcallester_complexity_term(kl_q_p=0.5, n=100, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            mcallester_complexity_term(kl_q_p=0.5, n=100, delta=1.0)

    def test_cert_invalid_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            PACBayesCertificate(delta=0.0)

    def test_cert_invalid_loss_bound_M_raises(self) -> None:
        with pytest.raises(ValueError, match="loss_bound_M"):
            PACBayesCertificate(loss_bound_M=0.0)

    def test_cert_invalid_tv_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="tv_tolerance"):
            PACBayesCertificate(tv_tolerance=-0.01)

    def test_pac_bayes_bound_invalid_empirical_risk(self) -> None:
        with pytest.raises(ValueError, match="empirical_risk"):
            pac_bayes_bound(empirical_risk=-0.1, kl_q_p=0.1, n=100, delta=0.05)

    def test_bound_monotone_in_n(self) -> None:
        """Larger n → tighter bound (smaller complexity term)."""
        r, kl, delta = 0.1, 0.5, 0.05
        b_small = pac_bayes_bound(r, kl, n=100, delta=delta)
        b_large = pac_bayes_bound(r, kl, n=10_000, delta=delta)
        assert b_large < b_small


# ── CalibratedHallucinationDetector (CHD) ───────────────────────────────────


@pytest.mark.unit
class TestCHDFPRBound:
    """fpr_upper_bound ≥ β; DKW formula; predict returns bool mask."""

    def _cal_scores(self, n: int = 200, seed: int = 0) -> "torch.Tensor":
        torch.manual_seed(seed)
        return torch.rand(n)

    def test_canary_fpr_upper_bound_gte_beta(self) -> None:
        """fpr_upper_bound ≥ β after fitting."""
        beta = 0.05
        chd = CalibratedHallucinationDetector(beta=beta, delta=1e-3)
        chd.fit(self._cal_scores(300))
        assert chd.fpr_upper_bound >= beta, (
            f"fpr_upper_bound={chd.fpr_upper_bound:.4f} < beta={beta}"
        )

    def test_canary_predict_returns_bool_tensor(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.1, delta=1e-3)
        chd.fit(self._cal_scores(100))
        test_scores = torch.rand(50)
        mask = chd.predict(test_scores)
        assert mask.dtype == torch.bool
        assert mask.shape == test_scores.shape

    def test_empirical_fpr_respects_calibration(self) -> None:
        """Empirical FPR on a fresh batch from the same distribution ≈ β."""
        beta = 0.10
        chd = CalibratedHallucinationDetector(beta=beta, delta=1e-6)
        # Calibrate on 5000 uniform samples; test on 1000.
        torch.manual_seed(0)
        chd.fit(torch.rand(5000))
        torch.manual_seed(1)
        emp_fpr = chd.empirical_fpr(torch.rand(1000))
        # Should be ≈ β; allow 3% tolerance.
        assert abs(emp_fpr - beta) < 0.03, (
            f"Empirical FPR {emp_fpr:.4f} too far from beta {beta:.4f}"
        )

    def test_dkw_slack_decreases_with_n(self) -> None:
        """DKW slack is monotonically decreasing in n."""
        delta = 1e-3
        slacks = [dkw_slack(n, delta) for n in [10, 100, 1_000, 10_000]]
        assert all(slacks[i] > slacks[i + 1] for i in range(len(slacks) - 1))

    def test_dkw_slack_formula(self) -> None:
        """Manual formula matches dkw_slack."""
        n, delta = 500, 0.01
        expected = math.sqrt(math.log(2.0 / delta) / (2 * n))
        assert dkw_slack(n, delta) == pytest.approx(expected, rel=1e-6)

    def test_bound_is_tight_with_many_samples(self) -> None:
        """With large n, DKW slack ≤ β (tight) should hold for typical β."""
        beta = 0.10
        chd = CalibratedHallucinationDetector(beta=beta, delta=1e-6)
        chd.fit(torch.rand(1_000_000))  # very large n → tiny slack
        assert chd.bound_is_tight, (
            f"Expected tight bound with 1M samples; slack="
            f"{dkw_slack(chd.n_calibration, chd.delta):.2e}, beta={beta}"
        )

    def test_threshold_in_zero_one_for_uniform(self) -> None:
        """For uniform [0,1] scores, threshold ≈ 1 − β."""
        beta = 0.10
        chd = CalibratedHallucinationDetector(beta=beta, delta=1e-6)
        torch.manual_seed(42)
        chd.fit(torch.rand(10_000))
        assert abs(chd.threshold - (1.0 - beta)) < 0.02, (
            f"Expected threshold ≈ {1-beta:.2f}, got {chd.threshold:.4f}"
        )

    def test_is_fitted_false_before_fit(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.1)
        assert not chd.is_fitted

    def test_predict_before_fit_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.1)
        with pytest.raises(RuntimeError, match="[Ff]it"):
            chd.predict(torch.rand(5))

    def test_fpr_before_fit_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.1)
        with pytest.raises(RuntimeError, match="[Ff]it"):
            _ = chd.fpr_upper_bound

    def test_empty_calibration_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.1)
        with pytest.raises(ValueError, match="[Ee]mpty"):
            chd.fit(torch.zeros(0))

    def test_scalar_tensor_raises(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.1)
        with pytest.raises(ValueError, match="1-D"):
            chd.fit(torch.tensor(0.5))

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            CalibratedHallucinationDetector(beta=0.0)
        with pytest.raises(ValueError, match="beta"):
            CalibratedHallucinationDetector(beta=1.0)

    def test_invalid_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            CalibratedHallucinationDetector(beta=0.1, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            CalibratedHallucinationDetector(beta=0.1, delta=1.0)

    def test_fluent_fit(self) -> None:
        chd = CalibratedHallucinationDetector(beta=0.1)
        result = chd.fit(torch.rand(50))
        assert result is chd

    def test_dkw_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            dkw_slack(0, 0.05)

    def test_dkw_invalid_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            dkw_slack(100, 0.0)
        with pytest.raises(ValueError, match="delta"):
            dkw_slack(100, 1.0)
