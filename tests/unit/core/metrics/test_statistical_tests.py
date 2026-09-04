"""Trust-layer cohort statistics (cold-diffusion T5/A9).

Clopper-Pearson is pinned to published table values; the cluster bootstrap is
checked against the iid bootstrap on a planted strongly-clustered cohort
(subject-level resampling MUST widen the interval); ``dkw_required_n`` is
round-tripped against the ``chd.dkw_slack`` SSOT it inverts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectramr.core.metrics.statistical_tests import StatisticalTests
from spectramr.infrastructure.calibration.chd import dkw_slack


class TestClopperPearsonInterval:
    def test_matches_published_table_at_half(self):
        """k=5, n=10, 95%: the classic (0.1871, 0.8129) interval."""
        ci = StatisticalTests.clopper_pearson_interval(5, 10, alpha=0.05)
        assert ci["proportion"] == pytest.approx(0.5)
        assert ci["ci_lower"] == pytest.approx(0.1871, abs=1e-4)
        assert ci["ci_upper"] == pytest.approx(0.8129, abs=1e-4)

    def test_zero_successes_boundary(self):
        """k=0: lower is exactly 0; upper = 1 − (α/2)^(1/n)."""
        ci = StatisticalTests.clopper_pearson_interval(0, 10, alpha=0.05)
        assert ci["ci_lower"] == 0.0
        assert ci["ci_upper"] == pytest.approx(1.0 - 0.025 ** (1.0 / 10.0), rel=1e-9)

    def test_all_successes_boundary(self):
        """k=n mirrors k=0 by symmetry."""
        ci = StatisticalTests.clopper_pearson_interval(10, 10, alpha=0.05)
        assert ci["ci_upper"] == 1.0
        assert ci["ci_lower"] == pytest.approx(0.025 ** (1.0 / 10.0), rel=1e-9)

    def test_interval_brackets_the_point_estimate(self):
        ci = StatisticalTests.clopper_pearson_interval(3, 50, alpha=0.05)
        assert 0.0 <= ci["ci_lower"] <= ci["proportion"] <= ci["ci_upper"] <= 1.0

    def test_validation_raises(self):
        with pytest.raises(ValueError, match="n >= 1"):
            StatisticalTests.clopper_pearson_interval(0, 0)
        with pytest.raises(ValueError, match="0 <= k <= n"):
            StatisticalTests.clopper_pearson_interval(11, 10)
        with pytest.raises(ValueError, match="alpha"):
            StatisticalTests.clopper_pearson_interval(5, 10, alpha=1.5)


class TestClusterBootstrapCI:
    @staticmethod
    def _clustered_cohort() -> tuple[np.ndarray, np.ndarray]:
        """8 subjects × 25 slices; nearly all variance is BETWEEN subjects."""
        rng = np.random.default_rng(11)
        subject_means = rng.normal(0.0, 1.0, size=8)
        values = np.concatenate(
            [m + rng.normal(0.0, 0.01, size=25) for m in subject_means]
        )
        ids = np.repeat(np.arange(8), 25)
        return values, ids

    def test_wider_than_iid_bootstrap_under_clustering(self):
        """Effective n is 8 subjects, not 200 slices: the subject-level
        interval must be far wider than the slice-iid one."""
        values, ids = self._clustered_cohort()
        cluster = StatisticalTests.cluster_bootstrap_ci(values, ids, seed=0)
        iid = StatisticalTests.bootstrap_ci(values, np.zeros_like(values))
        cluster_width = cluster["ci_upper"] - cluster["ci_lower"]
        iid_width = iid["ci_upper"] - iid["ci_lower"]
        assert cluster_width > 2.0 * iid_width

    def test_mean_and_reproducibility(self):
        values, ids = self._clustered_cohort()
        a = StatisticalTests.cluster_bootstrap_ci(values, ids, seed=7)
        b = StatisticalTests.cluster_bootstrap_ci(values, ids, seed=7)
        assert a == b
        assert a["mean"] == pytest.approx(float(np.mean(values)))
        assert a["n_subjects"] == 8
        assert a["ci_lower"] <= a["mean"] <= a["ci_upper"]

    def test_validation_raises(self):
        with pytest.raises(ValueError, match="align"):
            StatisticalTests.cluster_bootstrap_ci(np.ones(4), np.zeros(3))
        with pytest.raises(ValueError, match="at least one"):
            StatisticalTests.cluster_bootstrap_ci(np.empty(0), np.empty(0))
        with pytest.raises(ValueError, match=">= 2 subjects"):
            StatisticalTests.cluster_bootstrap_ci(np.ones(5), np.zeros(5))


class TestDesignEffect:
    def test_kish_formula(self):
        assert StatisticalTests.design_effect(10.0, 0.5) == pytest.approx(5.5)

    def test_no_correlation_means_no_inflation(self):
        assert StatisticalTests.design_effect(25.0, 0.0) == pytest.approx(1.0)

    def test_singleton_clusters_mean_no_inflation(self):
        assert StatisticalTests.design_effect(1.0, 0.9) == pytest.approx(1.0)

    def test_validation_raises(self):
        with pytest.raises(ValueError, match="mean_cluster_size"):
            StatisticalTests.design_effect(0.5, 0.1)
        with pytest.raises(ValueError, match="icc"):
            StatisticalTests.design_effect(10.0, 1.5)


class TestDkwRequiredN:
    def test_known_value(self):
        """α=0.1, ε=0.05: ⌈ln(20)/0.005⌉ = 600."""
        assert StatisticalTests.dkw_required_n(0.1, 0.05) == 600

    def test_round_trip_against_dkw_slack_ssot(self):
        """The returned n is the MINIMAL n whose slack fits inside eps."""
        for alpha, eps in [(0.1, 0.05), (0.05, 0.02), (0.01, 0.1)]:
            n = StatisticalTests.dkw_required_n(alpha, eps)
            assert dkw_slack(n, alpha) <= eps
            assert dkw_slack(n - 1, alpha) > eps

    def test_matches_closed_form(self):
        assert StatisticalTests.dkw_required_n(0.05, 0.01) == math.ceil(
            math.log(2 / 0.05) / (2 * 0.01**2)
        )

    def test_validation_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            StatisticalTests.dkw_required_n(0.0, 0.05)
        with pytest.raises(ValueError, match="eps"):
            StatisticalTests.dkw_required_n(0.1, 0.0)
