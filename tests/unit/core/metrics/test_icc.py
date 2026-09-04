"""Tests for Intraclass Correlation Coefficient (ICC).

Targets ``spectramr.core.metrics.icc``. ICC measures reliability across raters
/ scanners; used for radiomic-stability evaluation. Verifies the
oneway-random and twoway-mixed formulae against analytical expectations
on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.core.metrics.icc import ICCMetric


def test_perfect_agreement_yields_icc_near_one() -> None:
    """Identical measurements across raters → ICC ≈ 1."""
    metric = ICCMetric(mode="oneway")
    # 5 subjects, 3 raters, all raters give the subject's "true" value.
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    measurements = np.tile(truth.reshape(-1, 1), (1, 3))
    icc = metric(measurements)
    # Floating point noise in MS_within / MS_rows — tolerate.
    assert icc > 0.99


def test_pure_noise_yields_low_icc() -> None:
    """Random measurements with no subject signal → ICC ≈ 0."""
    rng = np.random.RandomState(0)
    metric = ICCMetric(mode="oneway")
    # 50 subjects, 4 raters, all values pure noise (no subject effect).
    measurements = rng.randn(50, 4)
    icc = metric(measurements)
    # ICC can be slightly negative for pure noise; expect near-zero.
    assert abs(icc) < 0.3, f"got ICC = {icc:.3f}"


def test_two_way_mode_runs() -> None:
    """Two-way mixed mode produces a finite ICC."""
    metric = ICCMetric(mode="twoway")
    rng = np.random.RandomState(1)
    truth = rng.randn(20, 1) * 5.0
    measurements = truth + rng.randn(20, 3) * 0.1  # high agreement
    icc = metric(measurements)
    assert np.isfinite(icc)
    assert icc > 0.9


def test_unknown_mode_raises() -> None:
    """Mode other than 'oneway' / 'twoway' fails loud (CLAUDE.md #9)."""
    metric = ICCMetric(mode="three_way_uncertain")
    with pytest.raises(ValueError, match="Unknown mode"):
        metric.calculate(np.zeros((4, 4)))


def test_torch_tensor_input_is_accepted() -> None:
    """Torch tensors are converted to numpy internally."""
    metric = ICCMetric(mode="oneway")
    truth = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    icc = metric(truth)
    assert icc > 0.99


def test_call_dunder_dispatches_to_calculate() -> None:
    """``ICCMetric()(x)`` is equivalent to ``ICCMetric().calculate(x)``."""
    metric = ICCMetric(mode="oneway")
    arr = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    via_call = metric(arr)
    via_calc = metric.calculate(arr)
    assert via_call == via_calc


def test_returns_python_float_not_numpy_scalar() -> None:
    """Return type is Python ``float`` for downstream JSON serialisation."""
    metric = ICCMetric(mode="oneway")
    arr = np.array([[1.0, 1.1], [2.0, 2.1]])
    icc = metric(arr)
    assert isinstance(icc, float)


@pytest.mark.parametrize(
    "shape",
    [(5, 2), (10, 3), (50, 4)],
    ids=lambda s: f"{s[0]}x{s[1]}",
)
def test_shape_matrix_oneway(shape: tuple[int, int]) -> None:
    """One-way ICC accepts arbitrary [N, K] shapes."""
    metric = ICCMetric(mode="oneway")
    measurements = np.random.RandomState(0).randn(*shape)
    icc = metric(measurements)
    assert np.isfinite(icc)


# --- Regression: degenerate shapes must raise, not silently divide by zero ---
# Before the fix, N<2 (single subject) divided MS_rows by (N-1)=0 and K<2
# (single rater) divided MS_within / MS_cols / MS_residual by (K-1)=0, yielding
# a NaN/inf ICC. ``calculate`` now raises ValueError for these degenerate cases.


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["oneway", "twoway"])
def test_single_subject_raises(mode: str) -> None:
    """A (1, K) matrix (N<2) must raise ValueError, not return NaN/inf."""
    metric = ICCMetric(mode=mode)
    measurements = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3): N=1, K=3
    assert measurements.shape == (1, 3)
    with pytest.raises(ValueError, match="N>=2"):
        metric.calculate(measurements)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["oneway", "twoway"])
def test_single_rater_raises(mode: str) -> None:
    """An (N, 1) matrix (K<2) must raise ValueError, not return NaN/inf."""
    metric = ICCMetric(mode=mode)
    measurements = np.array([[1.0], [2.0], [3.0]])  # shape (3, 1): N=3, K=1
    assert measurements.shape == (3, 1)
    with pytest.raises(ValueError, match="K>=2"):
        metric.calculate(measurements)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["oneway", "twoway"])
def test_single_subject_raises_tensor(mode: str) -> None:
    """The (1, K) guard also fires for a torch.Tensor input."""
    metric = ICCMetric(mode=mode)
    measurements = torch.tensor([[1.0, 2.0, 3.0]])  # shape (1, 3)
    with pytest.raises(ValueError, match="N>=2"):
        metric.calculate(measurements)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["oneway", "twoway"])
def test_single_rater_raises_tensor(mode: str) -> None:
    """The (N, 1) guard also fires for a torch.Tensor input."""
    metric = ICCMetric(mode=mode)
    measurements = torch.tensor([[1.0], [2.0], [3.0]])  # shape (3, 1)
    with pytest.raises(ValueError, match="K>=2"):
        metric.calculate(measurements)
