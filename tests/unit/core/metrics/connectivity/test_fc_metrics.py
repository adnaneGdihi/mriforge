"""Coverage for ``geodesic_fc_error`` (connectivity FC metric, fMRI §4).

Asserts the affine-invariant geodesic FC error metric behaves as a
distance: identical SPD inputs -> 0, distinct SPD inputs -> positive
finite, shape mismatch raises, and the optimisation direction
(``higher_is_better is False``) is correct.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.connectivity.fc_metrics import GeodesicFCErrorMetric

pytestmark = pytest.mark.unit


def _spd(n: int, *, seed: int) -> torch.Tensor:
    """Construct a deterministic SPD matrix ``A A^T + n I`` of size ``[n, n]``."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    return a @ a.transpose(-1, -2) + n * torch.eye(n, dtype=torch.float64)


def test_geodesic_fc_error_identical_is_zero() -> None:
    metric = GeodesicFCErrorMetric()
    c = _spd(8, seed=0)
    value = metric(c, c.clone())
    assert isinstance(value, float)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_geodesic_fc_error_distinct_is_positive_finite() -> None:
    metric = GeodesicFCErrorMetric()
    pred = _spd(8, seed=1)
    target = _spd(8, seed=2)
    value = metric(pred, target)
    assert math.isfinite(value)
    assert value > 0.0


def test_geodesic_fc_error_symmetric() -> None:
    """Geodesic distance is symmetric: d(A, B) == d(B, A)."""
    metric = GeodesicFCErrorMetric()
    a = _spd(6, seed=3)
    b = _spd(6, seed=4)
    assert metric(a, b) == pytest.approx(metric(b, a), rel=1e-6, abs=1e-6)


def test_geodesic_fc_error_batched_mean() -> None:
    """A [B, n, n] batch reduces to a single finite scalar mean."""
    metric = GeodesicFCErrorMetric()
    pred = torch.stack([_spd(5, seed=10), _spd(5, seed=11)])
    target = torch.stack([_spd(5, seed=12), _spd(5, seed=13)])
    value = metric(pred, target)
    assert isinstance(value, float)
    assert math.isfinite(value)
    assert value > 0.0


def test_geodesic_fc_error_shape_mismatch_raises() -> None:
    metric = GeodesicFCErrorMetric()
    pred = _spd(8, seed=5)
    target = _spd(6, seed=6)
    with pytest.raises(ValueError, match="shape mismatch"):
        metric(pred, target)


def test_geodesic_fc_error_higher_is_better_false() -> None:
    """An error metric must be minimised."""
    metric = GeodesicFCErrorMetric()
    assert metric.higher_is_better is False


def test_geodesic_fc_error_name() -> None:
    assert GeodesicFCErrorMetric().name == "geodesic_fc_error"
