"""Coverage tests for the MRF quantitative metrics (WS-2).

Targets :mod:`mriforge.core.metrics.quantitative.mrf_metrics`:

* :class:`GeodesicMRFParameterError` — chart-space distance on the
  5-D Bloch manifold; identical input -> 0, distinct -> positive/finite,
  shape mismatch -> ``ValueError``.
* :class:`CosinePreservationScore` — angular-structure agreement;
  identical embedding/raw pairs -> best value ``1.0``, distinct -> finite
  and bounded above by 1.
* :class:`CrossScannerT1T2Concordance` — Lin's CCC over paired T1/T2;
  identical scanners -> ``1.0``, distinct -> finite, shape mismatch ->
  ``ValueError``.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.core.metrics.quantitative.mrf_metrics import (
    CosinePreservationScore,
    CrossScannerT1T2Concordance,
    GeodesicMRFParameterError,
)

pytestmark = pytest.mark.unit


def _valid_theta(n: int = 4, seed: int = 0) -> torch.Tensor:
    """Random but on-manifold ``[n, 5]`` Bloch params (T1>T2>0, M0,B1>0)."""
    g = torch.Generator().manual_seed(seed)
    t1 = 0.5 + torch.rand(n, generator=g)  # T1 in [0.5, 1.5]
    t2 = 0.1 + 0.3 * torch.rand(n, generator=g)  # T2 < T1, both > 0
    m0 = 0.5 + torch.rand(n, generator=g)
    b0 = torch.randn(n, generator=g) * 0.1  # B0 may be any sign
    b1 = 0.8 + 0.4 * torch.rand(n, generator=g)
    return torch.stack([t1, t2, m0, b0, b1], dim=-1)


# --------------------------------------------------------------------------- #
# GeodesicMRFParameterError
# --------------------------------------------------------------------------- #
class TestGeodesicMRFParameterError:
    def test_contract_properties(self) -> None:
        m = GeodesicMRFParameterError()
        assert m.name == "geodesic_mrf_parameter_error"
        assert m.higher_is_better is False

    def test_identical_input_is_zero(self) -> None:
        m = GeodesicMRFParameterError()
        theta = _valid_theta(seed=1)
        val = m(theta, theta.clone())
        assert isinstance(val, float)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_distinct_input_positive_and_finite(self) -> None:
        m = GeodesicMRFParameterError()
        pred = _valid_theta(seed=2)
        target = _valid_theta(seed=3)
        val = m(pred, target)
        assert math.isfinite(val)
        assert val > 0.0

    def test_shape_mismatch_raises(self) -> None:
        m = GeodesicMRFParameterError()
        pred = _valid_theta(n=4, seed=4)
        target = _valid_theta(n=2, seed=5)
        with pytest.raises(ValueError):
            m(pred, target)


# --------------------------------------------------------------------------- #
# CosinePreservationScore
# --------------------------------------------------------------------------- #
class TestCosinePreservationScore:
    def test_contract_properties(self) -> None:
        m = CosinePreservationScore()
        assert m.name == "cosine_preservation_score"
        assert m.higher_is_better is True

    def test_perfect_preservation_is_best(self) -> None:
        # When the embedding equals the raw fingerprint, pairwise cosines
        # match exactly, so the score attains its upper bound of 1.0.
        m = CosinePreservationScore()
        g = torch.Generator().manual_seed(6)
        x = torch.randn(5, 8, generator=g)
        val = m(x, x.clone())
        assert isinstance(val, float)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_distinct_embedding_finite_and_at_most_one(self) -> None:
        m = CosinePreservationScore()
        g = torch.Generator().manual_seed(7)
        embedded = torch.randn(6, 4, generator=g)  # [B, d]
        raw = torch.randn(6, 16, generator=g)  # [B, N]
        val = m(embedded, raw)
        assert math.isfinite(val)
        assert val <= 1.0 + 1e-6
        # Distinct angular structure: score strictly below the optimum.
        assert val < 1.0

    def test_returns_float(self) -> None:
        m = CosinePreservationScore()
        g = torch.Generator().manual_seed(8)
        val = m(torch.randn(4, 4, generator=g), torch.randn(4, 4, generator=g))
        assert isinstance(val, float)


# --------------------------------------------------------------------------- #
# CrossScannerT1T2Concordance
# --------------------------------------------------------------------------- #
class TestCrossScannerT1T2Concordance:
    def test_contract_properties(self) -> None:
        m = CrossScannerT1T2Concordance()
        assert m.name == "cross_scanner_t1t2_concordance"
        assert m.higher_is_better is True

    def test_identical_scanners_perfect_concordance(self) -> None:
        # Lin's CCC == 1 when scanner-A and scanner-B agree exactly.
        m = CrossScannerT1T2Concordance()
        g = torch.Generator().manual_seed(9)
        maps = torch.rand(3, 2, 8, 8, generator=g)  # [B, (T1,T2), H, W]
        val = m(maps, maps.clone())
        assert isinstance(val, float)
        assert val == pytest.approx(1.0, abs=1e-4)

    def test_distinct_scanners_finite_and_bounded(self) -> None:
        m = CrossScannerT1T2Concordance()
        g = torch.Generator().manual_seed(10)
        pred = torch.rand(3, 2, 8, 8, generator=g)
        target = torch.rand(3, 2, 8, 8, generator=g)
        val = m(pred, target)
        assert math.isfinite(val)
        # CCC is bounded in [-1, 1]; uncorrelated maps stay below perfect.
        assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6
        assert val < 1.0

    def test_last_dim_layout_supported(self) -> None:
        # The [..., 2] last-dim path (dim < 4) must also evaluate finitely.
        m = CrossScannerT1T2Concordance()
        g = torch.Generator().manual_seed(11)
        pred = torch.rand(16, 2, generator=g)
        val = m(pred, pred.clone())
        assert val == pytest.approx(1.0, abs=1e-4)

    def test_shape_mismatch_raises(self) -> None:
        m = CrossScannerT1T2Concordance()
        pred = torch.rand(3, 2, 8, 8)
        target = torch.rand(3, 2, 4, 4)
        with pytest.raises(ValueError):
            m(pred, target)
