"""Unit tests for the cross-metric normalization schemes.

The meta-evaluation report mixes metrics on wildly different scales
(PSNR ~0-50 dB, FID 0-400+, LPIPS/SSIM 0-1, MSE 0-inf). These schemes
project a per-metric scalar onto a comparable axis so a single
leaderboard table / figure is readable. Direction-aware min-max is the
headline default; robust-z and percentile are documented alternatives.
"""

from __future__ import annotations

import math

import pytest

from spectramr.core.metrics.meta_evaluation.normalization import (
    direction_aware_minmax,
    normalize,
    percentile_rank,
    robust_zscore,
)

# ---------------------------------------------------------------------------
# direction_aware_minmax
# ---------------------------------------------------------------------------


def test_minmax_higher_is_better_maps_best_to_one() -> None:
    # All higher-is-better: largest -> 1.0, smallest -> 0.0.
    out = direction_aware_minmax(
        {"a": 10.0, "b": 30.0, "c": 20.0},
        higher_is_better={"a": True, "b": True, "c": True},
    )
    assert out["b"] == pytest.approx(1.0)
    assert out["a"] == pytest.approx(0.0)
    assert out["c"] == pytest.approx(0.5)


def test_minmax_flips_lower_is_better() -> None:
    # For a lower-is-better metric (e.g. MSE), the SMALLEST raw value is best
    # and must map to 1.0 after the direction flip.
    out = direction_aware_minmax(
        {"mse_lo": 1.0, "mse_mid": 5.0, "mse_hi": 9.0},
        higher_is_better={"mse_lo": False, "mse_mid": False, "mse_hi": False},
    )
    assert out["mse_lo"] == pytest.approx(1.0)
    assert out["mse_hi"] == pytest.approx(0.0)
    assert out["mse_mid"] == pytest.approx(0.5)


def test_minmax_all_outputs_in_unit_interval() -> None:
    out = direction_aware_minmax(
        {"a": -3.0, "b": 7.5, "c": 0.0, "d": 100.0},
        higher_is_better=None,  # default: larger is better
    )
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_minmax_none_direction_treats_larger_as_better() -> None:
    # When no direction map is given (e.g. consensus z-scores, already
    # direction-normalized), larger == better.
    out = direction_aware_minmax({"x": 0.0, "y": 2.0}, higher_is_better=None)
    assert out["y"] == pytest.approx(1.0)
    assert out["x"] == pytest.approx(0.0)


def test_minmax_constant_cohort_maps_to_half() -> None:
    out = direction_aware_minmax({"a": 4.0, "b": 4.0, "c": 4.0})
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == pytest.approx(0.5)
    assert out["c"] == pytest.approx(0.5)


def test_minmax_nan_passes_through_and_is_ignored_for_bounds() -> None:
    out = direction_aware_minmax(
        {"a": 1.0, "b": float("nan"), "c": 3.0},
        higher_is_better={"a": True, "b": True, "c": True},
    )
    assert math.isnan(out["b"])  # NaN stays NaN
    # Bounds computed from finite values only: a=0.0, c=1.0.
    assert out["a"] == pytest.approx(0.0)
    assert out["c"] == pytest.approx(1.0)


def test_minmax_does_not_mutate_input() -> None:
    src = {"a": 1.0, "b": 2.0}
    _ = direction_aware_minmax(src)
    assert src == {"a": 1.0, "b": 2.0}


def test_minmax_empty_returns_empty() -> None:
    assert direction_aware_minmax({}) == {}


# ---------------------------------------------------------------------------
# robust_zscore
# ---------------------------------------------------------------------------


def test_robust_zscore_centers_on_median() -> None:
    out = robust_zscore({"a": 1.0, "b": 2.0, "c": 3.0})
    # Median is b=2.0 -> exactly zero after centering.
    assert out["b"] == pytest.approx(0.0)
    assert out["a"] < 0.0 < out["c"]


def test_robust_zscore_resists_heavy_tail() -> None:
    # One huge FID-style outlier should not collapse the rest to ~0 the way
    # a mean/std z-score would. The non-outlier spread must stay resolvable.
    out = robust_zscore({"a": 1.0, "b": 2.0, "c": 3.0, "d": 1000.0})
    assert abs(out["a"] - out["c"]) > 0.5


def test_robust_zscore_flips_lower_is_better() -> None:
    out = robust_zscore(
        {"lo": 1.0, "mid": 2.0, "hi": 3.0},
        higher_is_better={"lo": False, "mid": False, "hi": False},
    )
    # Lower raw is better -> 'lo' should score ABOVE 'hi' after the flip.
    assert out["lo"] > out["hi"]


def test_robust_zscore_constant_cohort_is_zero() -> None:
    out = robust_zscore({"a": 5.0, "b": 5.0})
    assert out["a"] == pytest.approx(0.0)
    assert out["b"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# percentile_rank
# ---------------------------------------------------------------------------


def test_percentile_rank_bounds_and_monotonic() -> None:
    out = percentile_rank({"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0})
    assert out["a"] == pytest.approx(0.0)
    assert out["d"] == pytest.approx(1.0)
    assert out["a"] < out["b"] < out["c"] < out["d"]


def test_percentile_rank_flips_lower_is_better() -> None:
    out = percentile_rank(
        {"lo": 1.0, "hi": 9.0},
        higher_is_better={"lo": False, "hi": False},
    )
    assert out["lo"] == pytest.approx(1.0)
    assert out["hi"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# normalize() dispatch
# ---------------------------------------------------------------------------


def test_normalize_dispatches_to_minmax() -> None:
    values = {"a": 1.0, "b": 3.0, "c": 2.0}
    assert normalize(values, "minmax") == direction_aware_minmax(values)


def test_normalize_dispatches_to_each_scheme() -> None:
    values = {"a": 1.0, "b": 3.0, "c": 2.0}
    assert normalize(values, "robust_z") == robust_zscore(values)
    assert normalize(values, "percentile") == percentile_rank(values)


def test_normalize_unknown_scheme_raises() -> None:
    # No silent fallback (CLAUDE.md pitfall #9): a typo must abort.
    with pytest.raises(ValueError, match="scheme"):
        normalize({"a": 1.0}, "minmaxx")  # type: ignore[arg-type]
