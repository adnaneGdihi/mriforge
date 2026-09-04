"""Unit tests for :class:`ConformalCalibrator`.

Targets ``spectramr.infrastructure.calibration.conformal``. This file is part
of the parallel test-coverage push (Unit 14) and is the first set of
tests for the previously-untested ``src/infrastructure/calibration/``
subtree.

Coverage focus
--------------

The Vovk-Gammerman-Shafer split-conformal procedure has three
documented properties that we exercise as black-box property tests:

1. **Marginal validity.** On a held-out test split, the fraction of
   ground-truth points lying inside the prediction band is
   :math:`\\geq 1 - \\alpha` (within statistical noise).
2. **Tightness / monotonicity.** More calibration data should not
   *grow* the band; smaller :math:`\\alpha` (stricter coverage) gives a
   wider band.
3. **Edge cases.** :math:`\\alpha = 0` and :math:`\\alpha = 1` are
   degenerate; the implementation rejects them with a loud
   ``ValueError`` (CLAUDE.md pitfall #9: no silent clipping).

Plus the usual contract tests: ``predict`` before ``fit`` raises, empty
calibration set raises, determinism on identical inputs.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.calibration.conformal import (
    ConformalCalibrator,
    IConformalCalibrator,
)


# ---------------------------------------------------------------------------
# Construction / edge cases on alpha
# ---------------------------------------------------------------------------


def test_default_alpha_is_in_open_unit_interval() -> None:
    """Default constructor produces a usable α (∈ (0, 1))."""
    cal = ConformalCalibrator()
    assert 0.0 < cal.alpha < 1.0


def test_alpha_zero_raises_value_error() -> None:
    """``alpha = 0`` (100 % coverage) is degenerate — rejected, not clipped.

    Edge case from the unit spec. The Vovk-Gammerman-Shafer correction
    is :math:`\\lceil (n + 1)(1 - \\alpha) \\rceil / n`; at ``α = 0``
    this is exactly 1 (the empirical max), but only on the calibration
    set — there is no finite-sample coverage guarantee at α = 0 for
    fresh points. The implementation rejects this rather than silently
    falling back.
    """
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=0.0)


def test_alpha_one_raises_value_error() -> None:
    """``alpha = 1`` (0 % coverage) is degenerate — rejected.

    Edge case from the unit spec.
    """
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=1.0)


@pytest.mark.parametrize("bad_alpha", [-0.01, 1.01, 2.0, -1.0, float("inf")])
def test_alpha_outside_open_unit_interval_raises(bad_alpha: float) -> None:
    """α outside (0, 1) raises ``ValueError`` (fail loud)."""
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=bad_alpha)


# ---------------------------------------------------------------------------
# Fit / predict contract
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises_runtime_error() -> None:
    """Calling ``predict`` before ``fit`` is a contract violation."""
    cal = ConformalCalibrator(alpha=0.1)
    with pytest.raises(RuntimeError, match="not been fitted"):
        cal.predict(torch.zeros(4))


def test_quantile_before_fit_raises_runtime_error() -> None:
    """``quantile`` property raises before ``fit``."""
    cal = ConformalCalibrator(alpha=0.1)
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = cal.quantile


def test_is_fitted_false_before_fit_true_after() -> None:
    """``is_fitted`` flips after the first successful ``fit``."""
    cal = ConformalCalibrator(alpha=0.1)
    assert cal.is_fitted is False
    cal.fit([(torch.zeros(4), torch.zeros(4))])
    assert cal.is_fitted is True


def test_fit_empty_iterable_raises_value_error() -> None:
    """Empty calibration set is rejected with an actionable message."""
    cal = ConformalCalibrator(alpha=0.1)
    with pytest.raises(ValueError, match="empty|at least one"):
        cal.fit([])


def test_fit_returns_self() -> None:
    """``fit`` returns ``self`` for fluent chaining."""
    cal = ConformalCalibrator(alpha=0.1)
    out = cal.fit([(torch.zeros(4), torch.zeros(4))])
    assert out is cal


def test_predict_returns_three_tensors_of_equal_shape() -> None:
    """``predict`` returns ``(y_hat, lower, upper)`` of identical shape."""
    torch.manual_seed(0)
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.randn(64), torch.randn(64))])
    pred = torch.zeros(3, 5)
    y_hat, lower, upper = cal.predict(pred)
    assert y_hat.shape == lower.shape == upper.shape == pred.shape


def test_predict_band_is_symmetric_around_point_estimate() -> None:
    """``y_hat - lower == upper - y_hat == quantile`` (symmetric band)."""
    torch.manual_seed(0)
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.randn(64), torch.randn(64))])
    y_hat, lower, upper = cal.predict(torch.zeros(8))
    q = cal.quantile
    assert torch.allclose(y_hat - lower, torch.full_like(y_hat, q))
    assert torch.allclose(upper - y_hat, torch.full_like(y_hat, q))


def test_n_calibration_reflects_total_flattened_samples() -> None:
    """``n_calibration`` records the post-flatten sample count."""
    cal = ConformalCalibrator(alpha=0.1)
    # Two pairs of shape (10, 3) → 60 flattened samples
    pair1 = (torch.zeros(10, 3), torch.ones(10, 3))
    pair2 = (torch.zeros(10, 3), torch.zeros(10, 3))
    cal.fit([pair1, pair2])
    assert cal.n_calibration == 60


# ---------------------------------------------------------------------------
# Validity: marginal coverage ≥ 1 − α on fresh data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_marginal_coverage_meets_target_within_tolerance(alpha: float) -> None:
    """Empirical coverage on a fresh test split is within 5 pp of ``1 − α``.

    This is the headline Vovk-Gammerman-Shafer guarantee:
    :math:`\\mathbb{P}(y \\in [\\hat y - q, \\hat y + q]) \\geq 1 - \\alpha`
    under the i.i.d. assumption. We use a large synthetic regression so
    the empirical coverage converges to the theoretical value.
    """
    torch.manual_seed(2026)
    n_cal, n_test = 4000, 8000

    # y = x + ε, ε ~ N(0, 1); predictor knows the mean structure.
    cal_x = torch.randn(n_cal)
    cal_target = cal_x + torch.randn(n_cal)
    cal_pred = cal_x

    test_x = torch.randn(n_test)
    test_target = test_x + torch.randn(n_test)
    test_pred = test_x

    cal = ConformalCalibrator(alpha=alpha).fit([(cal_pred, cal_target)])
    coverage = cal.empirical_coverage(test_pred, test_target)
    target = 1.0 - alpha
    # |empirical - target| < 0.05 per unit spec.
    assert abs(coverage - target) < 0.05, (
        f"coverage {coverage:.3f} too far from target {target:.3f} (α={alpha})"
    )


def test_marginal_coverage_lower_bound_holds() -> None:
    """The VGS theorem says coverage ≥ 1 − α; verify on a fresh split.

    Stronger than just "close to" — we explicitly check the inequality
    direction with a small slack to absorb statistical noise.
    """
    torch.manual_seed(7)
    alpha = 0.1
    n_cal, n_test = 5000, 10000

    cal_target = torch.randn(n_cal)
    cal_pred = torch.zeros(n_cal)
    test_target = torch.randn(n_test)
    test_pred = torch.zeros(n_test)

    cal = ConformalCalibrator(alpha=alpha).fit([(cal_pred, cal_target)])
    coverage = cal.empirical_coverage(test_pred, test_target)
    # Theoretical floor is 1 - α; allow 1 pp of statistical slack.
    assert coverage >= (1.0 - alpha) - 0.01


# ---------------------------------------------------------------------------
# Tightness: smaller α → wider band; more data → not wider
# ---------------------------------------------------------------------------


def test_smaller_alpha_gives_wider_band() -> None:
    """Stricter coverage requires a larger quantile (wider band)."""
    torch.manual_seed(1)
    pred = torch.randn(2048)
    target = torch.randn(2048)
    q_loose = ConformalCalibrator(alpha=0.20).fit([(pred, target)]).quantile
    q_tight = ConformalCalibrator(alpha=0.05).fit([(pred, target)]).quantile
    assert q_tight > q_loose


def test_more_calibration_data_does_not_grow_band() -> None:
    """Doubling the calibration size should not noticeably enlarge ``q``.

    The empirical-quantile estimator is consistent — variance shrinks
    with ``n``. We assert the larger-sample quantile is not dramatically
    larger than the small-sample one (it can be either side of the small
    estimate but in expectation does not grow without bound).
    """
    torch.manual_seed(13)
    alpha = 0.1
    # Same distribution, different sample sizes.
    pred_small = torch.randn(500)
    target_small = pred_small + torch.randn(500)
    pred_large = torch.randn(5000)
    target_large = pred_large + torch.randn(5000)

    q_small = ConformalCalibrator(alpha=alpha).fit([(pred_small, target_small)]).quantile
    q_large = ConformalCalibrator(alpha=alpha).fit([(pred_large, target_large)]).quantile

    # The large-sample quantile should be near the small-sample one; we
    # require it doesn't blow up beyond a generous 50 % factor.
    assert q_large < 1.5 * q_small + 0.5


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_fit_is_deterministic_on_same_data() -> None:
    """Same calibration data → same fitted quantile (no hidden randomness)."""
    pred = torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8])
    target = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0])
    q1 = ConformalCalibrator(alpha=0.1).fit([(pred, target)]).quantile
    q2 = ConformalCalibrator(alpha=0.1).fit([(pred, target)]).quantile
    assert q1 == q2


def test_predict_is_deterministic_on_same_input() -> None:
    """Two predictions on identical inputs give bit-identical bands."""
    torch.manual_seed(0)
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.randn(64), torch.randn(64))])
    x = torch.tensor([1.0, 2.0, 3.0])
    a = cal.predict(x)
    b = cal.predict(x)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)


def test_order_invariance_within_batch() -> None:
    """The set of scores (and hence the quantile) is invariant under shuffling.

    Permuting rows of a calibration pair must not change the fitted
    quantile — the score is element-wise and the quantile is order-free.
    """
    torch.manual_seed(99)
    pred = torch.randn(512)
    target = torch.randn(512)
    perm = torch.randperm(pred.numel())
    q_original = ConformalCalibrator(alpha=0.1).fit([(pred, target)]).quantile
    q_shuffled = ConformalCalibrator(alpha=0.1).fit([(pred[perm], target[perm])]).quantile
    assert q_original == pytest.approx(q_shuffled)


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_fitted_calibrator_is_instance_of_protocol() -> None:
    """``ConformalCalibrator`` is a runtime instance of ``IConformalCalibrator``."""
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.zeros(4), torch.zeros(4))])
    assert isinstance(cal, IConformalCalibrator)


def test_unfitted_calibrator_is_also_instance_of_protocol() -> None:
    """Protocol membership is structural — fitness is not a precondition."""
    cal = ConformalCalibrator(alpha=0.1)
    assert isinstance(cal, IConformalCalibrator)


# ---------------------------------------------------------------------------
# Empirical coverage helper
# ---------------------------------------------------------------------------


def test_empirical_coverage_in_unit_interval() -> None:
    """The helper returns a fraction in [0, 1]."""
    torch.manual_seed(0)
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.randn(256), torch.randn(256))])
    cov = cal.empirical_coverage(torch.randn(128), torch.randn(128))
    assert 0.0 <= cov <= 1.0


def test_empirical_coverage_on_perfect_predictor_is_one() -> None:
    """When prediction == target, every point is inside the band."""
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.randn(128), torch.randn(128))])
    target = torch.tensor([0.0, 0.0, 0.0])
    pred = target.clone()
    # Band radius q ≥ 0 (it can be 0 only in degenerate calibration cases),
    # and target lies on top of pred, so coverage == 1.0.
    cov = cal.empirical_coverage(pred, target)
    assert cov == 1.0
