"""Statistical Tests Module.

Provides research-grade statistical analysis utilities for comparing MRI
reconstruction model performance, including paired hypothesis testing,
non-parametric alternatives, effect size computation, bootstrap confidence
intervals, and multiple-comparison corrections.

All methods are stateless (``@staticmethod``) and operate on NumPy arrays
to integrate cleanly with the existing metrics pipeline.

Design:
    - ``compute_p_values()`` — legacy independent t-test (preserved for backward compat)
    - ``paired_ttest()`` — paired t-test for matched subject comparisons
    - ``wilcoxon_signed_rank()`` — non-parametric alternative when normality is violated
    - ``bootstrap_ci()`` — percentile bootstrap confidence intervals
    - ``cohens_d()`` — effect size measurement (paired and unpaired)
    - ``bonferroni_correction()`` — family-wise error rate control
    - ``fdr_correction()`` — Benjamini-Hochberg false discovery rate control
    - ``perform_tests()`` — full battery returning a structured ``StatisticalReport``
    - ``clopper_pearson_interval()`` — exact binomial CI (hallucination-rate reporting)
    - ``cluster_bootstrap_ci()`` — subject-level bootstrap for clustered slice metrics
    - ``design_effect()`` — Kish variance-inflation factor for clustered samples
    - ``dkw_required_n()`` — calibration-set sizing (inverse of ``core.metrics.dkw.dkw_slack``)

References:
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    - Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
    - Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate.
    - Clopper, C. & Pearson, E. (1934). The Use of Confidence or Fiducial Limits.
    - Kish, L. (1965). Survey Sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class StatisticalReport:
    """Structured output from ``StatisticalTests.perform_tests()``.

    Attributes:
        paired_ttest: Dict with keys 't_statistic', 'p_value'.
        wilcoxon: Dict with keys 'statistic', 'p_value'.
        effect_size: Dict with keys 'cohens_d', 'interpretation'.
        bootstrap_ci: Dict with keys 'mean', 'ci_lower', 'ci_upper', 'confidence'.
        summary: Human-readable summary string.
    """

    paired_ttest: dict[str, float] = field(default_factory=dict)
    wilcoxon: dict[str, float] = field(default_factory=dict)
    effect_size: dict[str, float | str] = field(default_factory=dict)
    bootstrap_ci: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, dict | str]:
        """Serialize to nested dictionary for JSON export."""
        return {
            "paired_ttest": self.paired_ttest,
            "wilcoxon": self.wilcoxon,
            "effect_size": self.effect_size,
            "bootstrap_ci": self.bootstrap_ci,
            "summary": self.summary,
        }


class StatisticalTests:
    """Statistical tests for evaluating MRI reconstruction quality.

    All methods are static and operate on NumPy arrays or 1D tensors.
    """

    # ── Legacy API (backward compatibility) ──────────────────────────

    @staticmethod
    def compute_p_values(
        metric_values_a: np.ndarray, metric_values_b: np.ndarray
    ) -> dict[str, float]:
        """Compute p-values for comparing two sets of metric values.

        Uses independent (unpaired) two-sample t-test. Preserved for
        backward compatibility; prefer ``paired_ttest()`` for matched
        subject comparisons.

        Args:
            metric_values_a: Metric values for model A.
            metric_values_b: Metric values for model B.

        Returns:
            Dict with key 't_test_p_value'.
        """
        from scipy import stats

        t_stat, p_val = stats.ttest_ind(metric_values_a, metric_values_b)
        return {"t_test_p_value": float(p_val)}

    # ── Paired Hypothesis Tests ──────────────────────────────────────

    @staticmethod
    def paired_ttest(
        values_a: np.ndarray,
        values_b: np.ndarray,
        alternative: str = "two-sided",
    ) -> dict[str, float]:
        """Paired t-test for matched subject comparisons.

        Appropriate when the same subjects are evaluated by two models
        (e.g., PSNR of model A vs. model B on the same test slices).

        Args:
            values_a: Per-subject metric values for model A [N].
            values_b: Per-subject metric values for model B [N].
            alternative: 'two-sided', 'less', or 'greater'.

        Returns:
            Dict with 't_statistic', 'p_value', 'mean_diff'.

        Raises:
            ValueError: If arrays have different lengths or < 2 samples.
        """
        a = np.asarray(values_a, dtype=np.float64)
        b = np.asarray(values_b, dtype=np.float64)

        if a.shape != b.shape:
            raise ValueError(f"Paired t-test requires equal-length arrays: {a.shape} vs {b.shape}")
        if len(a) < 2:
            raise ValueError("Paired t-test requires at least 2 paired observations")

        # Guard: if all differences are zero, t-test is undefined (0/0 → NaN)
        diff = a - b
        if np.allclose(diff, 0.0):
            return {"t_statistic": 0.0, "p_value": 1.0, "mean_diff": 0.0}

        import warnings

        from scipy import stats

        # SciPy emits a RuntimeWarning ("Precision loss occurred in
        # moment calculation due to catastrophic cancellation") whenever
        # the paired differences have zero or near-zero variance — for
        # example, a perfect constant shift between the two samples.
        # That case is a *valid* and intentional input for this routine
        # (a controlled offset between methods), so swallow the warning
        # here rather than letting the strict-mode test policy promote
        # it to a hard failure.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Precision loss occurred in moment calculation.*",
                category=RuntimeWarning,
            )
            result = stats.ttest_rel(a, b, alternative=alternative)
        return {
            "t_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "mean_diff": float(np.mean(diff)),
        }

    @staticmethod
    def wilcoxon_signed_rank(
        values_a: np.ndarray,
        values_b: np.ndarray,
        alternative: str = "two-sided",
    ) -> dict[str, float]:
        """Wilcoxon signed-rank test (non-parametric paired test).

        Use when normality of paired differences cannot be assumed
        (e.g., small sample sizes typical in MRI studies, N < 30).

        Args:
            values_a: Per-subject metric values for model A [N].
            values_b: Per-subject metric values for model B [N].
            alternative: 'two-sided', 'less', or 'greater'.

        Returns:
            Dict with 'statistic', 'p_value'.

        Raises:
            ValueError: If arrays have different lengths.
        """
        a = np.asarray(values_a, dtype=np.float64)
        b = np.asarray(values_b, dtype=np.float64)

        if a.shape != b.shape:
            raise ValueError(f"Wilcoxon test requires equal-length arrays: {a.shape} vs {b.shape}")

        from scipy import stats

        diff = a - b
        # Remove zero differences (Wilcoxon is undefined for ties at zero)
        nonzero_mask = diff != 0
        if nonzero_mask.sum() < 1:
            return {"statistic": 0.0, "p_value": 1.0}

        result = stats.wilcoxon(
            diff[nonzero_mask],
            alternative=alternative,
        )
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        }

    # ── Effect Size ──────────────────────────────────────────────────

    @staticmethod
    def cohens_d(
        values_a: np.ndarray,
        values_b: np.ndarray,
        paired: bool = True,
    ) -> dict[str, float | str]:
        """Compute Cohen's d effect size.

        For paired data, uses the standard deviation of differences.
        For unpaired data, uses pooled standard deviation.

        Interpretation thresholds (Cohen 1988):
            |d| < 0.2  → negligible
            0.2 ≤ |d| < 0.5 → small
            0.5 ≤ |d| < 0.8 → medium
            |d| ≥ 0.8 → large

        Args:
            values_a: Metric values for model A [N].
            values_b: Metric values for model B [N].
            paired: If True, compute paired effect size.

        Returns:
            Dict with 'cohens_d' (float) and 'interpretation' (str).
        """
        a = np.asarray(values_a, dtype=np.float64)
        b = np.asarray(values_b, dtype=np.float64)

        if paired:
            diff = a - b
            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        else:
            n_a, n_b = len(a), len(b)
            s_a, s_b = np.std(a, ddof=1), np.std(b, ddof=1)
            # Pooled standard deviation
            s_pooled = np.sqrt(((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2))
            d = (np.mean(a) - np.mean(b)) / (s_pooled + 1e-12)

        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {"cohens_d": float(d), "interpretation": interpretation}

    # ── Confidence Intervals ─────────────────────────────────────────

    @staticmethod
    def bootstrap_ci(
        values_a: np.ndarray,
        values_b: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        statistic: str = "mean_diff",
        seed: int | None = 42,
    ) -> dict[str, float]:
        """Bootstrap confidence intervals for a statistic of paired differences.

        Uses the percentile method (Efron & Tibshirani 1993).

        Args:
            values_a: Metric values for model A [N].
            values_b: Metric values for model B [N].
            n_bootstrap: Number of bootstrap resamples.
            confidence: Confidence level (e.g., 0.95 for 95% CI).
            statistic: Which statistic to bootstrap. Options:
                'mean_diff' — mean of paired differences.
                'median_diff' — median of paired differences.
            seed: Random seed for reproducibility.

        Returns:
            Dict with 'observed', 'ci_lower', 'ci_upper', 'confidence'.
        """
        a = np.asarray(values_a, dtype=np.float64)
        b = np.asarray(values_b, dtype=np.float64)
        diff = a - b
        n = len(diff)

        rng = np.random.default_rng(seed)

        stat_fn = np.mean if statistic == "mean_diff" else np.median
        observed = float(stat_fn(diff))

        # Bootstrap
        boot_stats = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_stats[i] = stat_fn(diff[idx])

        alpha = 1.0 - confidence
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        return {
            "observed": observed,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence": confidence,
        }

    # ── Multiple Comparison Corrections ──────────────────────────────

    @staticmethod
    def bonferroni_correction(
        p_values: list[float] | np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, list]:
        """Bonferroni correction for family-wise error rate.

        Conservative correction: reject if p * m < α.

        Args:
            p_values: List of p-values from multiple tests.
            alpha: Desired significance level.

        Returns:
            Dict with 'corrected_p_values', 'significant' (bool list),
            'alpha_corrected'.
        """
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        # Guard the empty case (matches holm_bonferroni_correction): the
        # alpha / m term below divides by zero for m == 0.
        if m == 0:
            return {
                "corrected_p_values": [],
                "significant": [],
                "alpha_corrected": float("nan"),
            }
        corrected = np.minimum(p * m, 1.0)
        significant = corrected < alpha

        return {
            "corrected_p_values": corrected.tolist(),
            "significant": significant.tolist(),
            "alpha_corrected": alpha / m,
        }

    @staticmethod
    def fdr_correction(
        p_values: list[float] | np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, list]:
        """Benjamini-Hochberg FDR correction.

        Less conservative than Bonferroni; controls the expected proportion
        of false discoveries among rejected hypotheses.

        Args:
            p_values: List of p-values from multiple tests.
            alpha: Desired FDR level.

        Returns:
            Dict with 'corrected_p_values', 'significant' (bool list).
        """
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        # Guard the empty case (matches holm_bonferroni_correction): the
        # ``corrected[sorted_idx[-1]]`` access below indexes into an empty array.
        if m == 0:
            return {"corrected_p_values": [], "significant": []}
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]

        # BH adjusted p-values
        corrected = np.empty(m)
        corrected[sorted_idx[-1]] = sorted_p[-1]

        for i in range(m - 2, -1, -1):
            corrected[sorted_idx[i]] = min(corrected[sorted_idx[i + 1]], sorted_p[i] * m / (i + 1))

        corrected = np.minimum(corrected, 1.0)
        significant = corrected < alpha

        return {
            "corrected_p_values": corrected.tolist(),
            "significant": significant.tolist(),
        }

    @staticmethod
    def holm_bonferroni_correction(
        p_values: list[float] | np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, list]:
        """Holm-Bonferroni step-down family-wise-error-rate correction.

        Sort p-values ascending; compare the i-th sorted p (1-indexed)
        against alpha / (m - i + 1). Strictly less conservative than
        Bonferroni; preserves FWER control.

        Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-6 (H4).

        Args:
            p_values: list of p-values from multiple tests.
            alpha: family-wise alpha.

        Returns:
            Dict with ``corrected_p_values``, ``significant`` (bool list).
        """
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        if m == 0:
            return {"corrected_p_values": [], "significant": []}
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]

        corrected_sorted = np.empty(m)
        running_max = 0.0
        for i, pv in enumerate(sorted_p):
            adj = pv * (m - i)
            running_max = max(running_max, adj)
            corrected_sorted[i] = min(running_max, 1.0)

        corrected = np.empty(m)
        corrected[sorted_idx] = corrected_sorted
        significant = corrected < alpha
        return {
            "corrected_p_values": corrected.tolist(),
            "significant": significant.tolist(),
        }

    @staticmethod
    def correct_p_values(
        p_values: list[float] | np.ndarray,
        method: str = "fdr_bh",
        alpha: float = 0.05,
    ) -> dict[str, list]:
        """Dispatch table for multiple-comparison correction.

        Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-6.

        Args:
            p_values: raw p-values.
            method: ``bonferroni`` | ``holm_bonferroni`` | ``fdr_bh`` (default).
            alpha: family-wise / FDR alpha.

        Returns:
            Same dict shape as ``bonferroni_correction``.
        """
        method = method.lower().strip()
        if method == "bonferroni":
            return StatisticalTests.bonferroni_correction(p_values, alpha=alpha)
        if method in {"holm", "holm_bonferroni"}:
            return StatisticalTests.holm_bonferroni_correction(p_values, alpha=alpha)
        if method in {"fdr", "fdr_bh", "benjamini_hochberg"}:
            return StatisticalTests.fdr_correction(p_values, alpha=alpha)
        raise ValueError(
            f"Unknown correction method {method!r}. Use bonferroni | holm_bonferroni | fdr_bh."
        )

    # ── Trust-layer cohort statistics (cold-diffusion T5/A9) ─────────

    @staticmethod
    def clopper_pearson_interval(
        k: int,
        n: int,
        alpha: float = 0.05,
    ) -> dict[str, float]:
        """Exact (Clopper-Pearson) two-sided CI for a binomial proportion.

        Used to report the hallucination rate HR = k/n (number of images whose
        trajectory excursion exceeds the admissible radius) with a
        finite-sample-exact interval: the normal approximation is unusable at
        the small k typical of a certified model.

        Args:
            k: Number of successes (e.g., flagged images), 0 ≤ k ≤ n.
            n: Number of trials (cohort size), ≥ 1.
            alpha: Two-sided miscoverage (0.05 → 95% CI).

        Returns:
            Dict with 'proportion', 'ci_lower', 'ci_upper', 'alpha'.

        Raises:
            ValueError: If n < 1, k outside [0, n], or alpha outside (0, 1).
        """
        if n < 1:
            raise ValueError(f"Clopper-Pearson requires n >= 1, got n={n}")
        if not 0 <= k <= n:
            raise ValueError(f"Clopper-Pearson requires 0 <= k <= n, got k={k}, n={n}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        from scipy import stats

        # The exact interval inverts two binomial tests via Beta quantiles;
        # the Beta parameters are degenerate at the boundaries, where the
        # one-sided limits are exactly 0 and 1.
        lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
        upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
        return {
            "proportion": k / n,
            "ci_lower": lower,
            "ci_upper": upper,
            "alpha": alpha,
        }

    @staticmethod
    def cluster_bootstrap_ci(
        values: np.ndarray,
        subject_ids: np.ndarray,
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> dict[str, float]:
        """Subject-level (cluster) bootstrap CI for the mean of a slice metric.

        Slices from the same subject are correlated; resampling SLICES treats
        them as independent and yields anti-conservative (too narrow)
        intervals. This resamples SUBJECTS with replacement and keeps every
        drawn subject's slices intact, so the interval reflects the effective
        (subject-level) sample size.

        Args:
            values: Per-slice metric values [N].
            subject_ids: Subject label per slice [N] (any hashable dtype).
            n_boot: Number of bootstrap resamples.
            alpha: Two-sided miscoverage (0.05 → 95% CI).
            seed: RNG seed for reproducibility.

        Returns:
            Dict with 'mean', 'ci_lower', 'ci_upper', 'alpha', 'n_subjects'.

        Raises:
            ValueError: If shapes mismatch, values is empty, or < 2 subjects.
        """
        v = np.asarray(values, dtype=np.float64)
        ids = np.asarray(subject_ids)
        if v.shape != ids.shape:
            raise ValueError(f"values and subject_ids must align: {v.shape} vs {ids.shape}")
        if v.size == 0:
            raise ValueError("cluster_bootstrap_ci requires at least one value")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        unique = np.unique(ids)
        n_subjects = len(unique)
        if n_subjects < 2:
            raise ValueError(f"cluster bootstrap requires >= 2 subjects, got {n_subjects}")
        groups = [v[ids == u] for u in unique]

        rng = np.random.default_rng(seed)
        boot_stats = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_subjects, size=n_subjects)
            boot_stats[i] = np.mean(np.concatenate([groups[j] for j in idx]))

        return {
            "mean": float(np.mean(v)),
            "ci_lower": float(np.percentile(boot_stats, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
            "alpha": alpha,
            "n_subjects": n_subjects,
        }

    @staticmethod
    def design_effect(mean_cluster_size: float, icc: float) -> float:
        """Kish design effect DEFF = 1 + (n̄ − 1)·ICC.

        Variance-inflation factor of a clustered sample relative to an iid
        sample of the same size: divide the nominal N by DEFF to get the
        effective sample size for power/sizing calculations.

        Args:
            mean_cluster_size: Mean slices per subject n̄ (≥ 1).
            icc: Intraclass correlation ς in [0, 1].

        Raises:
            ValueError: If mean_cluster_size < 1 or icc outside [0, 1].
        """
        if mean_cluster_size < 1.0:
            raise ValueError(f"mean_cluster_size must be >= 1, got {mean_cluster_size}")
        if not 0.0 <= icc <= 1.0:
            raise ValueError(f"icc must be in [0, 1], got {icc}")
        return 1.0 + (mean_cluster_size - 1.0) * icc

    @staticmethod
    def dkw_required_n(alpha: float, eps: float) -> int:
        """Minimal calibration size n with DKW slack ≤ eps.

        Inverse of ``core.metrics.dkw.dkw_slack``: the smallest
        n such that sqrt(ln(2/α) / (2n)) ≤ ε, i.e. ⌈ln(2/α) / (2ε²)⌉. Used to
        size the trust-calibration set before fitting a certificate.

        Args:
            alpha: DKW confidence budget in (0, 1).
            eps: Target uniform CDF band half-width, > 0.

        Raises:
            ValueError: If alpha outside (0, 1) or eps <= 0.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        return int(math.ceil(math.log(2.0 / alpha) / (2.0 * eps * eps)))

    # ── Full Battery ─────────────────────────────────────────────────

    @staticmethod
    def perform_tests(
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        metric_fn: str = "mse",
        confidence: float = 0.95,
        n_bootstrap: int = 10000,
    ) -> StatisticalReport:
        """Perform full battery of statistical tests on predictions vs targets.

        Computes per-sample metric values, then runs paired t-test, Wilcoxon,
        effect size, and bootstrap CI on the resulting distributions.

        Args:
            predictions: Model outputs [N, ...] (batch of samples).
            targets: Ground truth [N, ...] (same shape as predictions).
            metric_fn: Per-sample metric. Options:
                'mse' — Mean Squared Error per sample.
                'psnr' — Peak Signal-to-Noise Ratio per sample.
                'mae' — Mean Absolute Error per sample.
            confidence: Confidence level for bootstrap CI.
            n_bootstrap: Number of bootstrap resamples.

        Returns:
            StatisticalReport with all test results.
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        predictions = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        # Compute per-sample metrics
        N = predictions.shape[0]
        if N < 2:
            return StatisticalReport(summary="Insufficient samples (N < 2)")

        # Flatten spatial dims for per-sample metric
        pred_flat = predictions.reshape(N, -1)
        tgt_flat = targets.reshape(N, -1)

        if metric_fn == "mse":
            per_sample = np.mean((pred_flat - tgt_flat) ** 2, axis=1)
        elif metric_fn == "mae":
            per_sample = np.mean(np.abs(pred_flat - tgt_flat), axis=1)
        elif metric_fn == "psnr":
            mse_vals = np.mean((pred_flat - tgt_flat) ** 2, axis=1)
            data_range = np.max(tgt_flat) - np.min(tgt_flat)
            per_sample = 10 * np.log10((data_range**2) / (mse_vals + 1e-12))
        else:
            raise ValueError(f"Unknown metric_fn: {metric_fn}")

        # Use zero as baseline (comparing model errors against "no error")
        # More useful: compare two model arrays. Here we report absolute distribution.
        baseline = np.zeros_like(per_sample)

        report = StatisticalReport()

        # Paired t-test (model errors vs. zero)
        report.paired_ttest = StatisticalTests.paired_ttest(per_sample, baseline)

        # Wilcoxon
        report.wilcoxon = StatisticalTests.wilcoxon_signed_rank(per_sample, baseline)

        # Effect size
        report.effect_size = StatisticalTests.cohens_d(per_sample, baseline, paired=True)

        # Bootstrap CI on per-sample metric
        report.bootstrap_ci = StatisticalTests.bootstrap_ci(
            per_sample,
            baseline,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )

        # Summary
        p_t = report.paired_ttest.get("p_value", float("nan"))
        p_w = report.wilcoxon.get("p_value", float("nan"))
        d = report.effect_size.get("cohens_d", float("nan"))
        interp = report.effect_size.get("interpretation", "unknown")
        ci_lo = report.bootstrap_ci.get("ci_lower", float("nan"))
        ci_hi = report.bootstrap_ci.get("ci_upper", float("nan"))

        report.summary = (
            f"N={N} | metric={metric_fn}\n"
            f"  Paired t-test: p={p_t:.4e}\n"
            f"  Wilcoxon: p={p_w:.4e}\n"
            f"  Cohen's d: {d:.3f} ({interp})\n"
            f"  {confidence * 100:.0f}% Bootstrap CI: [{ci_lo:.6f}, {ci_hi:.6f}]"
        )

        return report
