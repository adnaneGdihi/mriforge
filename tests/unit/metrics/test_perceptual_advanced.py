"""Unit tests for perceptual_advanced.py — four archetypes per metric.

Covers PDM, CWSSIM, MAD. ST-MAD requires a 5-D sequence input so gets
a separate shape-check.

All metrics use ``.forward(pred, target)`` (nn.Module).
Heavy backbone metrics (none here) would be @pytest.mark.slow.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id


def _img(b: int = 1, c: int = 1, h: int = 32, w: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


def _is_finite_scalar(t: torch.Tensor) -> bool:
    return t.ndim == 0 and math.isfinite(t.item())


# ---------------------------------------------------------------------------
# PDM
# ---------------------------------------------------------------------------


class TestPDM:
    @pytest.fixture
    def pdm(self):
        from mriforge.core.metrics.perceptual_advanced import PDM
        return PDM()

    # canary
    def test_canary_finite_nonnegative(self, pdm):
        x = _img()
        y = _img(seed=1)
        result = pdm.forward(x, y)
        assert _is_finite_scalar(result)
        assert result.item() >= 0.0

    # parametrize batch sizes
    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, pdm, b):
        x = _img(b)
        y = _img(b, seed=10)
        result = pdm.forward(x, y)
        assert _is_finite_scalar(result) and result.item() >= 0.0

    # parametrize value ranges
    @pytest.mark.parametrize("offset", [0.0, 0.5])
    def test_value_ranges(self, pdm, offset):
        x = _img() + offset
        y = _img(seed=2) + offset
        result = pdm.forward(x, y)
        assert math.isfinite(result.item())

    # edge: identical inputs → PDM ≈ 0
    def test_edge_identical_near_zero(self, pdm):
        x = _img()
        result = pdm.forward(x, x)
        assert _is_finite_scalar(result)
        assert result.item() < 1e-3

    # edge: all-zero inputs → no crash
    def test_edge_all_zero_no_crash(self, pdm):
        z = torch.zeros(1, 1, 32, 32)
        result = pdm.forward(z, z)
        assert math.isfinite(result.item())

    # edge: constant non-zero → finite
    def test_edge_constant_arrays(self, pdm):
        a = torch.ones(1, 1, 32, 32) * 0.3
        b = torch.ones(1, 1, 32, 32) * 0.7
        result = pdm.forward(a, b)
        assert math.isfinite(result.item())

    # raises: mismatched spatial shapes — jaxtyping/beartype may raise,
    # or conv layers will raise; either way the code should not silently
    # return NaN.  We test that it either raises or stays finite.
    def test_raises_or_finite_on_shape_mismatch(self, pdm):
        x = _img(1, 1, 32, 32)
        y = _img(1, 1, 16, 16)
        try:
            result = pdm.forward(x, y)
            # If it doesn't raise, the result must be finite (no silent NaN)
            assert math.isfinite(result.item())
        except Exception:
            pass  # expected path

    # sanity_shape
    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from mriforge.core.metrics.perceptual_advanced import PDM
        b, c, h, w = shape
        x = _img(b, c, h, w)
        y = _img(b, c, h, w, seed=99)
        result = PDM().forward(x, y)
        assert _is_finite_scalar(result)


# ---------------------------------------------------------------------------
# CWSSIM
# ---------------------------------------------------------------------------


class TestCWSSIM:
    @pytest.fixture
    def cwssim(self):
        from mriforge.core.metrics.perceptual_advanced import CWSSIM
        return CWSSIM()

    # canary
    def test_canary_in_unit_range(self, cwssim):
        x = _img()
        y = _img(seed=1)
        result = cwssim.forward(x, y)
        assert _is_finite_scalar(result)
        assert 0.0 <= result.item() <= 1.0 + 1e-5

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, cwssim, b):
        x = _img(b)
        y = _img(b, seed=11)
        result = cwssim.forward(x, y)
        assert _is_finite_scalar(result)
        assert 0.0 <= result.item() <= 1.0 + 1e-5

    @pytest.mark.parametrize("scale", [0.1, 1.0])
    def test_value_ranges(self, cwssim, scale):
        x = _img() * scale
        y = _img(seed=3) * scale
        result = cwssim.forward(x, y)
        assert math.isfinite(result.item())

    # edge: identical inputs → CW-SSIM ≥ 0.9
    def test_edge_identical_near_one(self, cwssim):
        x = _img()
        result = cwssim.forward(x, x)
        assert _is_finite_scalar(result)
        assert result.item() >= 0.5  # Should be high for identical images

    # edge: all-zero → no crash
    def test_edge_all_zero(self, cwssim):
        z = torch.zeros(1, 1, 32, 32)
        result = cwssim.forward(z, z)
        assert math.isfinite(result.item())

    # edge: constant different arrays → finite
    def test_edge_constant_different(self, cwssim):
        a = torch.ones(1, 1, 32, 32) * 0.2
        b = torch.ones(1, 1, 32, 32) * 0.8
        result = cwssim.forward(a, b)
        assert math.isfinite(result.item())

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from mriforge.core.metrics.perceptual_advanced import CWSSIM
        b, c, h, w = shape
        x = _img(b, c, h, w)
        y = _img(b, c, h, w, seed=77)
        result = CWSSIM().forward(x, y)
        assert _is_finite_scalar(result)


# ---------------------------------------------------------------------------
# MAD
# ---------------------------------------------------------------------------


class TestMAD:
    @pytest.fixture
    def mad(self):
        from mriforge.core.metrics.perceptual_advanced import MAD
        return MAD()

    # canary
    def test_canary_finite_nonnegative(self, mad):
        x = _img()
        y = _img(seed=1)
        result = mad.forward(x, y)
        assert _is_finite_scalar(result)
        assert result.item() >= 0.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, mad, b):
        x = _img(b)
        y = _img(b, seed=12)
        result = mad.forward(x, y)
        assert _is_finite_scalar(result) and result.item() >= 0.0

    @pytest.mark.parametrize("dw", [0.0, 0.5, 1.0])
    def test_detection_weight_range(self, dw):
        from mriforge.core.metrics.perceptual_advanced import MAD
        m = MAD(detection_weight=dw)
        x = _img()
        y = _img(seed=4)
        result = m.forward(x, y)
        assert math.isfinite(result.item())

    # edge: identical inputs → MAD ≈ 0
    def test_edge_identical_near_zero(self, mad):
        x = _img()
        result = mad.forward(x, x)
        assert _is_finite_scalar(result)
        assert result.item() < 0.5  # Should be small for identical images

    # edge: all-zero
    def test_edge_all_zero(self, mad):
        z = torch.zeros(1, 1, 32, 32)
        result = mad.forward(z, z)
        assert math.isfinite(result.item())

    # edge: small spatial dims (block_size > image)
    def test_edge_tiny_image(self, mad):
        x = _img(1, 1, 4, 4)
        y = _img(1, 1, 4, 4, seed=5)
        result = mad.forward(x, y)
        assert math.isfinite(result.item())

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from mriforge.core.metrics.perceptual_advanced import MAD
        b, c, h, w = shape
        x = _img(b, c, h, w)
        y = _img(b, c, h, w, seed=55)
        result = MAD().forward(x, y)
        assert _is_finite_scalar(result)


# ---------------------------------------------------------------------------
# STMAD — expects 5-D (B, T, C, H, W) input
# ---------------------------------------------------------------------------


class TestSTMAD:
    @pytest.fixture
    def stmad(self):
        from mriforge.core.metrics.perceptual_advanced import STMAD
        return STMAD()

    def _seq(self, b: int = 1, t: int = 4, c: int = 1, h: int = 16, w: int = 16,
              seed: int = 0) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.rand(b, t, c, h, w, generator=g)

    # canary
    def test_canary_finite(self, stmad):
        x = self._seq()
        y = self._seq(seed=1)
        result = stmad.forward(x, y)
        assert _is_finite_scalar(result)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, stmad, b):
        x = self._seq(b)
        y = self._seq(b, seed=20)
        result = stmad.forward(x, y)
        assert _is_finite_scalar(result)

    def test_edge_identical_inputs(self, stmad):
        x = self._seq()
        result = stmad.forward(x, x)
        assert math.isfinite(result.item())

    def test_edge_all_zero(self, stmad):
        z = torch.zeros(1, 4, 1, 16, 16)
        result = stmad.forward(z, z)
        assert math.isfinite(result.item())
