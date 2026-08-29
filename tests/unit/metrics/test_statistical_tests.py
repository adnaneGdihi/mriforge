"""Unit tests for StatisticalTests suite."""

import numpy as np
import pytest


class TestStatisticalTests:
    """Tests for the research-grade statistical testing battery."""

    @pytest.fixture
    def st(self):
        from mriforge.core.metrics.statistical_tests import StatisticalTests
        return StatisticalTests

    # ── Legacy API ───────────────────────────────────────────────────

    def test_compute_p_values_backward_compat(self, st):
        """Legacy compute_p_values should still work."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = st.compute_p_values(a, b)
        assert "t_test_p_value" in result
        assert 0.0 <= result["t_test_p_value"] <= 1.0

    # ── Paired t-test ────────────────────────────────────────────────

    def test_paired_ttest_identical(self, st):
        """Identical arrays should give p ≈ 1.0 (no difference)."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = st.paired_ttest(a, a)
        assert result["p_value"] >= 0.99, f"Expected p≈1.0, got {result['p_value']}"
        assert abs(result["mean_diff"]) < 1e-10

    def test_paired_ttest_known_shift(self, st):
        """Constant shift should give small p-value."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=50)
        b = a + 2.0  # Large constant shift

        result = st.paired_ttest(a, b)
        assert result["p_value"] < 0.001, f"Expected p<<0.05, got {result['p_value']}"
        assert abs(result["mean_diff"] - (-2.0)) < 0.1

    def test_paired_ttest_shape_mismatch(self, st):
        """Different-length arrays should raise ValueError."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="equal-length"):
            st.paired_ttest(a, b)

    # ── Wilcoxon ─────────────────────────────────────────────────────

    def test_wilcoxon_identical(self, st):
        """Identical arrays should give p = 1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = st.wilcoxon_signed_rank(a, a)
        assert result["p_value"] == 1.0

    def test_wilcoxon_known_shift(self, st):
        """Large shift should be detected."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=30)
        b = a + 3.0

        result = st.wilcoxon_signed_rank(a, b)
        assert result["p_value"] < 0.01, f"Expected p<0.01, got {result['p_value']}"

    # ── Cohen's d ────────────────────────────────────────────────────

    def test_cohens_d_zero(self, st):
        """Identical arrays → d ≈ 0 → negligible."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = st.cohens_d(a, a, paired=True)
        assert abs(result["cohens_d"]) < 0.01
        assert result["interpretation"] == "negligible"

    def test_cohens_d_large_effect(self, st):
        """Large shift → |d| ≥ 0.8 → large."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=100)
        b = a + 2.0  # Shift of 2 std devs

        result = st.cohens_d(a, b, paired=True)
        assert abs(result["cohens_d"]) > 0.8
        assert result["interpretation"] == "large"

    def test_cohens_d_unpaired(self, st):
        """Unpaired effect size should also work."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(5, 1, size=50)

        result = st.cohens_d(a, b, paired=False)
        assert abs(result["cohens_d"]) > 2.0  # Huge effect

    # ── Bootstrap CI ─────────────────────────────────────────────────

    def test_bootstrap_ci_contains_true_mean(self, st):
        """95% CI should contain the true mean difference."""
        rng = np.random.default_rng(42)
        true_diff = 1.0
        a = rng.normal(true_diff, 0.5, size=50)
        b = np.zeros(50)

        result = st.bootstrap_ci(a, b, confidence=0.95, seed=42)
        assert result["ci_lower"] < true_diff < result["ci_upper"], (
            f"True diff {true_diff} not in [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
        )

    def test_bootstrap_ci_narrower_with_more_samples(self, st):
        """CI should narrow with more data."""
        rng = np.random.default_rng(42)
        b = np.zeros(200)

        a_small = rng.normal(1.0, 0.5, size=10)
        ci_small = st.bootstrap_ci(a_small, b[:10], seed=42)
        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]

        a_large = rng.normal(1.0, 0.5, size=200)
        ci_large = st.bootstrap_ci(a_large, b, seed=42)
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]

        assert width_large < width_small, (
            f"CI should narrow: {width_large:.4f} >= {width_small:.4f}"
        )

    # ── Multiple Comparison Corrections ──────────────────────────────

    def test_bonferroni_correction(self, st):
        """Bonferroni should multiply p-values by m."""
        p_values = [0.01, 0.04, 0.1]
        result = st.bonferroni_correction(p_values, alpha=0.05)

        assert len(result["corrected_p_values"]) == 3
        assert result["corrected_p_values"][0] == pytest.approx(0.03, abs=1e-6)
        assert result["corrected_p_values"][1] == pytest.approx(0.12, abs=1e-6)
        assert result["significant"][0] is True   # 0.03 < 0.05
        assert result["significant"][1] is False  # 0.12 > 0.05

    def test_bonferroni_caps_at_1(self, st):
        """Corrected p-values should not exceed 1.0."""
        result = st.bonferroni_correction([0.5, 0.8], alpha=0.05)
        assert all(p <= 1.0 for p in result["corrected_p_values"])

    def test_fdr_correction(self, st):
        """FDR correction should be less conservative than Bonferroni."""
        p_values = [0.001, 0.01, 0.04, 0.1, 0.5]

        bonf = st.bonferroni_correction(p_values, alpha=0.05)
        fdr = st.fdr_correction(p_values, alpha=0.05)

        n_bonf_sig = sum(bonf["significant"])
        n_fdr_sig = sum(fdr["significant"])
        assert n_fdr_sig >= n_bonf_sig, "FDR should reject at least as many as Bonferroni"

    # ── Full Battery ─────────────────────────────────────────────────

    def test_perform_tests_returns_report(self, st):
        """perform_tests should return a StatisticalReport."""
        import torch

        from mriforge.core.metrics.statistical_tests import StatisticalReport

        pred = torch.randn(10, 1, 16, 16)
        target = torch.randn(10, 1, 16, 16)

        report = st.perform_tests(pred, target, metric_fn="mse")
        assert isinstance(report, StatisticalReport)
        assert "p_value" in report.paired_ttest
        assert "p_value" in report.wilcoxon
        assert "cohens_d" in report.effect_size
        assert "ci_lower" in report.bootstrap_ci
        assert len(report.summary) > 0

    def test_perform_tests_to_dict(self, st):
        """Report should serialize to dict."""
        import torch

        pred = torch.randn(10, 1, 16, 16)
        target = torch.randn(10, 1, 16, 16)

        report = st.perform_tests(pred, target, metric_fn="mae")
        d = report.to_dict()
        assert "paired_ttest" in d
        assert "summary" in d

    def test_perform_tests_insufficient_samples(self, st):
        """N=1 should return graceful message."""
        import torch

        pred = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)

        report = st.perform_tests(pred, target)
        assert "Insufficient" in report.summary


# ── Extended coverage (four-archetype pattern) ───────────────────────────────

class TestStatisticalTestsExtended:
    """Additional archetypes: parametrize, edge, raises, sanity_shape."""

    @pytest.fixture
    def st(self):
        from mriforge.core.metrics.statistical_tests import StatisticalTests
        return StatisticalTests

    # ── parametrize: bootstrap over multiple confidence levels ───────────

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_bootstrap_ci_confidence_levels(self, st, confidence):
        rng = np.random.default_rng(7)
        a = rng.normal(1.0, 0.5, size=50)
        b = np.zeros(50)
        result = st.bootstrap_ci(a, b, confidence=confidence, seed=7)
        assert result["ci_lower"] < result["ci_upper"]

    # ── parametrize: cohens_d paired/unpaired ────────────────────────────

    @pytest.mark.parametrize("paired", [True, False])
    def test_cohens_d_paired_flag(self, st, paired):
        rng = np.random.default_rng(8)
        a = rng.normal(0, 1, size=30)
        b = a + 1.0
        result = st.cohens_d(a, b, paired=paired)
        assert "cohens_d" in result
        assert np.isfinite(result["cohens_d"])

    # ── edge: identical arrays → Wilcoxon p=1, t-test p≈1 ───────────────

    def test_edge_wilcoxon_identical_p_one(self, st):
        a = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = st.wilcoxon_signed_rank(a, a)
        assert result["p_value"] == 1.0

    def test_edge_constant_array_cohens_d(self, st):
        """Constant array has std=0 → d should be 0 or handled gracefully."""
        a = np.full(10, 3.14)
        b = np.full(10, 3.14)
        result = st.cohens_d(a, b, paired=True)
        assert np.isfinite(result["cohens_d"]) or np.isnan(result["cohens_d"])

    def test_edge_length1_bootstrap(self, st):
        """Degenerate n=1 bootstrap — must not raise."""
        a = np.array([1.0])
        b = np.array([2.0])
        result = st.bootstrap_ci(a, b, confidence=0.95, seed=0)
        assert isinstance(result, dict)

    def test_edge_fdr_all_significant(self, st):
        """All p-values near zero → all should remain significant after FDR."""
        p_values = [1e-10, 1e-10, 1e-10]
        result = st.fdr_correction(p_values, alpha=0.05)
        assert all(result["significant"])

    # ── raises: paired_ttest length mismatch ────────────────────────────

    def test_raises_paired_ttest_length_mismatch(self, st):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            st.paired_ttest(a, b)

    # ── sanity_shape: perform_tests over SHAPE_MATRIX_2D ────────────────

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", [
        (2, 1, 8, 8),
        (4, 1, 16, 16),
        (8, 1, 8, 8),
    ], ids=["2x1x8x8", "4x1x16x16", "8x1x8x8"])
    def test_shape_matrix_perform_tests(self, st, shape):
        import torch
        b, c, h, w = shape
        pred = torch.randn(b, c, h, w)
        target = torch.randn(b, c, h, w)
        report = st.perform_tests(pred, target, metric_fn="mse")
        assert hasattr(report, "paired_ttest")
        assert isinstance(report.summary, str)
