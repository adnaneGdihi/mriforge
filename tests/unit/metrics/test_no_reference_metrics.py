"""Unit tests for no_reference_metrics.py — four archetypes per class.

The three classes (TenengradVariance, LaplacianVariance, NIQE, BRISQUE)
are callable objects, not BaseMetric subclasses, and take only
``prediction`` (target is ignored).
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


# ---------------------------------------------------------------------------
# TenengradVariance
# ---------------------------------------------------------------------------


class TestTenengradVariance:
    @pytest.fixture
    def tv(self):
        from spectramr.core.metrics.no_reference_metrics import TenengradVariance
        return TenengradVariance()

    # canary
    def test_canary_finite_nonnegative(self, tv):
        x = _img()
        result = tv(x)
        assert isinstance(result, float)
        assert math.isfinite(result)
        assert result >= 0.0

    # parametrize over batch sizes
    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, tv, b):
        x = _img(b)
        result = tv(x)
        assert math.isfinite(result) and result >= 0.0

    # parametrize over value ranges
    @pytest.mark.parametrize("scale", [0.01, 1.0, 10.0])
    def test_value_range_scale(self, tv, scale):
        x = _img() * scale
        result = tv(x)
        assert math.isfinite(result) and result >= 0.0

    # edge: identical inputs — no crash, variance well defined
    def test_edge_identical_inputs(self, tv):
        x = _img()
        result = tv(x)
        assert math.isfinite(result)

    # edge: all-zero image — variance = 0
    def test_edge_all_zero(self, tv):
        z = torch.zeros(1, 1, 32, 32)
        result = tv(z)
        assert math.isfinite(result)
        assert result >= 0.0

    # edge: constant non-zero array — gradients vanish
    def test_edge_constant_array(self, tv):
        c = torch.ones(1, 1, 32, 32) * 0.5
        result = tv(c)
        assert math.isfinite(result)

    # edge: 2-D (H, W) input — should auto-expand
    def test_edge_2d_input(self, tv):
        x = _img().squeeze(0).squeeze(0)
        result = tv(x)
        assert math.isfinite(result)

    # raises: this metric does NOT raise on wrong inputs (it coerces) — canary only
    def test_property_higher_is_better(self, tv):
        assert tv.higher_is_better is True

    # sanity_shape
    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from spectramr.core.metrics.no_reference_metrics import TenengradVariance
        b, c, h, w = shape
        x = _img(b, c, h, w)
        result = TenengradVariance()(x)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# LaplacianVariance
# ---------------------------------------------------------------------------


class TestLaplacianVariance:
    @pytest.fixture
    def lv(self):
        from spectramr.core.metrics.no_reference_metrics import LaplacianVariance
        return LaplacianVariance()

    def test_canary_finite_nonnegative(self, lv):
        x = _img()
        result = lv(x)
        assert isinstance(result, float)
        assert math.isfinite(result)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, lv, b):
        assert math.isfinite(lv(_img(b)))

    @pytest.mark.parametrize("scale", [0.01, 1.0, 5.0])
    def test_value_range_scale(self, lv, scale):
        assert math.isfinite(lv(_img() * scale))

    def test_edge_all_zero(self, lv):
        z = torch.zeros(1, 1, 32, 32)
        result = lv(z)
        assert math.isfinite(result)

    def test_edge_constant_array(self, lv):
        c = torch.ones(1, 1, 32, 32) * 0.5
        result = lv(c)
        assert math.isfinite(result)

    def test_edge_2d_input(self, lv):
        x = _img().squeeze(0).squeeze(0)
        assert math.isfinite(lv(x))

    def test_property_higher_is_better(self, lv):
        assert lv.higher_is_better is True

    def test_sharp_greater_than_blurry(self, lv):
        """Sharp image should have higher Laplacian variance than blurred."""
        import torch.nn.functional as F
        sharp = _img(1, 1, 32, 32, seed=42)
        blurry = F.avg_pool2d(sharp, kernel_size=5, stride=1, padding=2)
        assert lv(sharp) >= lv(blurry)

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from spectramr.core.metrics.no_reference_metrics import LaplacianVariance
        b, c, h, w = shape
        result = LaplacianVariance()(_img(b, c, h, w))
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# NIQE (requires larger patches — uses images >= 96px for full execution)
# ---------------------------------------------------------------------------


class TestNIQE:
    @pytest.fixture
    def niqe(self):
        from spectramr.core.metrics.no_reference_metrics import NIQE
        return NIQE()

    # canary with large enough image
    def test_canary_finite_nonnegative(self, niqe):
        x = _img(1, 1, 128, 128)
        result = niqe(x)
        assert isinstance(result, float)
        assert math.isfinite(result)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, niqe, b):
        x = _img(b, 1, 128, 128)
        assert math.isfinite(niqe(x))

    def test_edge_small_image_fallback(self, niqe):
        """Image smaller than patch_size should fall back gracefully (returns 0.0)."""
        x = _img(1, 1, 32, 32)
        result = niqe(x)
        assert isinstance(result, float) and math.isfinite(result)

    def test_edge_all_zero_no_crash(self, niqe):
        z = torch.zeros(1, 1, 128, 128)
        result = niqe(z)
        assert math.isfinite(result)

    def test_property_higher_is_better_false(self, niqe):
        assert niqe.higher_is_better is False


# ---------------------------------------------------------------------------
# BRISQUE
# ---------------------------------------------------------------------------


class TestBRISQUE:
    @pytest.fixture
    def brisque(self):
        from spectramr.core.metrics.no_reference_metrics import BRISQUE
        return BRISQUE()

    def test_canary_finite_nonnegative(self, brisque):
        x = _img(1, 1, 64, 64)
        result = brisque(x)
        assert isinstance(result, float)
        assert math.isfinite(result)
        assert result >= 0.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, brisque, b):
        x = _img(b, 1, 64, 64)
        assert math.isfinite(brisque(x))

    def test_edge_all_zero_no_crash(self, brisque):
        z = torch.zeros(1, 1, 32, 32)
        result = brisque(z)
        assert math.isfinite(result)

    def test_edge_constant_array(self, brisque):
        c = torch.ones(1, 1, 32, 32) * 0.5
        result = brisque(c)
        assert math.isfinite(result)

    def test_property_lower_is_better(self, brisque):
        assert brisque.higher_is_better is False

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from spectramr.core.metrics.no_reference_metrics import BRISQUE
        b, c, h, w = shape
        result = BRISQUE()(_img(b, c, h, w))
        assert math.isfinite(result)
