import numpy as np

from spectramr.infrastructure.reporting.plotters import _stats


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 1.0, size=500)
    lo, hi = _stats.bootstrap_ci(x, n_boot=1000, seed=0)
    assert lo < x.mean() < hi
    assert hi - lo < 0.5


def test_paired_bootstrap_pvalue_detects_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=200)
    b = a + 1.0  # b strictly better
    p = _stats.paired_bootstrap_pvalue(a, b, n_boot=2000, seed=0)
    assert p < 0.05


def test_holm_bonferroni_monotone():
    raw = [0.001, 0.02, 0.04]
    adj = _stats.holm_bonferroni(raw)
    assert adj[0] <= adj[1] <= adj[2]
    assert all(0.0 <= q <= 1.0 for q in adj)


def test_t_ci_half_width_matches_student_t_formula():
    from scipy import stats

    vals = [30.0, 31.0, 29.0]  # n=3
    arr = np.asarray(vals)
    expected = float(stats.t.ppf(0.975, df=2)) * arr.std(ddof=1) / np.sqrt(3)
    assert abs(_stats.t_ci_half_width(vals) - expected) < 1e-9


def test_t_ci_half_width_is_wider_than_normal_z_for_small_n():
    # The whole point of the fix: at small n the t interval is strictly wider
    # than the z=1.96 normal approximation that was used before.
    vals = [30.0, 31.0, 29.0]
    arr = np.asarray(vals)
    z_half = 1.96 * arr.std(ddof=1) / np.sqrt(arr.size)
    assert _stats.t_ci_half_width(vals) > z_half
    # At n=3 the factor is t=4.30 vs z=1.96 — more than 2x wider.
    assert _stats.t_ci_half_width(vals) > 2.0 * z_half


def test_t_ci_half_width_undefined_for_single_sample():
    assert _stats.t_ci_half_width([42.0]) == 0.0
    assert _stats.t_ci_half_width([]) == 0.0


def test_t_ci_half_width_ignores_nans():
    assert _stats.t_ci_half_width([1.0, 2.0, 3.0, np.nan]) == _stats.t_ci_half_width(
        [1.0, 2.0, 3.0]
    )
