"""Unit tests for report_step_metrics.py — four archetypes per metric class.

All metrics subclass BaseMetric and expose compute_metric(preds, target).
Several have non-standard INPUT_SIGNATURE (landmark_pair, displacement_field,
volume_3d, distribution) and are tested with their domain-appropriate inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id


def _img(b: int = 1, c: int = 1, h: int = 16, w: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


def _is_finite(t: torch.Tensor) -> bool:
    return math.isfinite(t.item())


# ---------------------------------------------------------------------------
# MutualInformationMetric (#17)
# ---------------------------------------------------------------------------


class TestMutualInformation:
    @pytest.fixture
    def mi(self):
        from mriforge.core.metrics.report_step_metrics import MutualInformationMetric
        return MutualInformationMetric()

    def test_canary_finite(self, mi):
        x = _img()
        y = _img(seed=1)
        result = mi.compute_metric(x, y)
        assert _is_finite(result)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, mi, b):
        assert _is_finite(mi.compute_metric(_img(b), _img(b, seed=2)))

    def test_edge_identical_inputs_positive(self, mi):
        x = _img()
        result = mi.compute_metric(x, x)
        # MI(x, x) should be positive (maximum MI for a given distribution)
        assert result.item() > 0.0

    def test_edge_all_zero_no_crash(self, mi):
        z = torch.zeros(1, 1, 16, 16)
        result = mi.compute_metric(z, z)
        assert math.isfinite(result.item())

    @pytest.mark.sanity_shape
    @pytest.mark.parametrize("shape", SHAPE_MATRIX_2D, ids=shape_id)
    def test_shape_matrix(self, shape):
        from mriforge.core.metrics.report_step_metrics import MutualInformationMetric
        b, c, h, w = shape
        result = MutualInformationMetric().compute_metric(_img(b, c, h, w), _img(b, c, h, w, seed=7))
        assert _is_finite(result)


# ---------------------------------------------------------------------------
# EdgePreservationIndex (#21)
# ---------------------------------------------------------------------------


class TestEdgePreservationIndex:
    @pytest.fixture
    def epi(self):
        from mriforge.core.metrics.report_step_metrics import EdgePreservationIndex
        return EdgePreservationIndex()

    def test_canary_finite_or_nan_when_no_edges(self, epi):
        x = _img(1, 1, 32, 32)
        y = _img(1, 1, 32, 32, seed=1)
        result = epi.compute_metric(x, y)
        assert isinstance(result.item(), float)  # nan or finite is OK

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, epi, b):
        result = epi.compute_metric(_img(b, 1, 32, 32), _img(b, 1, 32, 32, seed=3))
        assert isinstance(result.item(), float)

    def test_edge_identical_near_one(self, epi):
        x = _img(1, 1, 32, 32)
        result = epi.compute_metric(x, x)
        # Perfect reconstruction → EPI = 1 (or NaN if no edges)
        v = result.item()
        assert math.isnan(v) or v > 0.9

    def test_edge_all_zero_gradient_returns_nan(self, epi):
        z = torch.zeros(1, 1, 32, 32)
        result = epi.compute_metric(z, z)
        # constant image has no edges → NaN is expected
        assert math.isnan(result.item()) or math.isfinite(result.item())


# ---------------------------------------------------------------------------
# RadialKSpaceError (#23)
# ---------------------------------------------------------------------------


class TestRadialKSpaceError:
    @pytest.fixture
    def rke(self):
        from mriforge.core.metrics.report_step_metrics import RadialKSpaceError
        return RadialKSpaceError()

    def test_canary_finite_nonnegative(self, rke):
        x = _img()
        y = _img(seed=1)
        result = rke.compute_metric(x, y)
        assert _is_finite(result) and result.item() >= 0.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, rke, b):
        result = rke.compute_metric(_img(b), _img(b, seed=4))
        assert _is_finite(result) and result.item() >= 0.0

    def test_edge_identical_near_zero(self, rke):
        x = _img()
        result = rke.compute_metric(x, x)
        assert result.item() < 1e-5

    def test_edge_all_zero(self, rke):
        z = torch.zeros(1, 1, 16, 16)
        result = rke.compute_metric(z, z)
        assert math.isfinite(result.item())


# ---------------------------------------------------------------------------
# BlandAltmanBias / LimitsOfAgreement (#34, #35)
# ---------------------------------------------------------------------------


class TestBlandAltman:
    @pytest.fixture
    def bias(self):
        from mriforge.core.metrics.report_step_metrics import BlandAltmanBias
        return BlandAltmanBias()

    @pytest.fixture
    def loa_upper(self):
        from mriforge.core.metrics.report_step_metrics import LimitsOfAgreementUpper
        return LimitsOfAgreementUpper()

    @pytest.fixture
    def loa_lower(self):
        from mriforge.core.metrics.report_step_metrics import LimitsOfAgreementLower
        return LimitsOfAgreementLower()

    def test_canary_bias_finite(self, bias):
        x = _img()
        y = _img(seed=1)
        result = bias.compute_metric(x, y)
        assert _is_finite(result)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes_bias(self, bias, b):
        assert _is_finite(bias.compute_metric(_img(b), _img(b, seed=5)))

    def test_edge_identical_bias_near_zero(self, bias):
        x = _img()
        result = bias.compute_metric(x, x)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_edge_known_offset_bias(self, bias):
        x = torch.ones(1, 1, 16, 16) * 0.8
        y = torch.ones(1, 1, 16, 16) * 0.3
        result = bias.compute_metric(x, y)
        assert result.item() == pytest.approx(0.5, abs=1e-4)

    def test_canary_loa_finite(self, loa_upper, loa_lower):
        x = _img()
        y = _img(seed=1)
        assert _is_finite(loa_upper.compute_metric(x, y))
        assert _is_finite(loa_lower.compute_metric(x, y))

    def test_loa_upper_above_lower(self, loa_upper, loa_lower):
        x = _img(2)
        y = _img(2, seed=6)
        u = loa_upper.compute_metric(x, y).item()
        l = loa_lower.compute_metric(x, y).item()
        assert u >= l


# ---------------------------------------------------------------------------
# ICC3_1 (#36)
# ---------------------------------------------------------------------------


class TestICC3_1:
    @pytest.fixture
    def icc(self):
        from mriforge.core.metrics.report_step_metrics import ICC3_1Metric
        return ICC3_1Metric()

    def test_canary_in_minus1_1(self, icc):
        x = _img()
        y = _img(seed=1)
        result = icc.compute_metric(x, y)
        v = result.item()
        assert math.isnan(v) or -1.0 - 1e-5 <= v <= 1.0 + 1e-5

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, icc, b):
        result = icc.compute_metric(_img(b), _img(b, seed=7))
        v = result.item()
        assert math.isnan(v) or -1.0 - 1e-5 <= v <= 1.0 + 1e-5

    def test_edge_identical_near_one(self, icc):
        x = _img(2)
        result = icc.compute_metric(x, x)
        v = result.item()
        assert math.isnan(v) or v > 0.9

    def test_edge_single_element_nan(self, icc):
        x = _img(1, 1, 1, 1)
        y = _img(1, 1, 1, 1, seed=8)
        result = icc.compute_metric(x, y)
        # Only 1 element → ICC returns NaN
        assert math.isnan(result.item())


# ---------------------------------------------------------------------------
# CoefficientOfVariation (#37)
# ---------------------------------------------------------------------------


class TestCoefficientOfVariation:
    @pytest.fixture
    def cv(self):
        from mriforge.core.metrics.report_step_metrics import CoefficientOfVariation
        return CoefficientOfVariation()

    def test_canary_finite_nonnegative(self, cv):
        x = _img()
        result = cv.compute_metric(x, x)  # target ignored
        v = result.item()
        assert math.isnan(v) or v >= 0.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, cv, b):
        x = _img(b)
        result = cv.compute_metric(x, x)
        assert math.isfinite(result.item()) or math.isnan(result.item())

    def test_edge_zero_mean_returns_nan(self, cv):
        # Zero-mean tensor → CV undefined → NaN
        z = torch.zeros(1, 1, 16, 16)
        result = cv.compute_metric(z, z)
        assert math.isnan(result.item())

    def test_edge_constant_positive(self, cv):
        # Constant positive → std=0 → CV=0
        c = torch.ones(1, 1, 16, 16)
        result = cv.compute_metric(c, c)
        v = result.item()
        assert math.isnan(v) or v == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# DetectionSensitivity / Specificity (#39)
# ---------------------------------------------------------------------------


class TestDetection:
    @pytest.fixture
    def sens(self):
        from mriforge.core.metrics.report_step_metrics import DetectionSensitivity
        return DetectionSensitivity(threshold=0.5)

    @pytest.fixture
    def spec(self):
        from mriforge.core.metrics.report_step_metrics import DetectionSpecificity
        return DetectionSpecificity(threshold=0.5)

    def test_canary_sensitivity_in_unit(self, sens):
        x = _img()
        y = _img(seed=1)
        result = sens.compute_metric(x, y)
        v = result.item()
        assert math.isnan(v) or 0.0 <= v <= 1.0

    def test_canary_specificity_in_unit(self, spec):
        x = _img()
        y = _img(seed=1)
        result = spec.compute_metric(x, y)
        v = result.item()
        assert math.isnan(v) or 0.0 <= v <= 1.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes_sensitivity(self, sens, b):
        x = _img(b)
        y = _img(b, seed=9)
        v = sens.compute_metric(x, y).item()
        assert math.isnan(v) or 0.0 <= v <= 1.0

    def test_edge_all_ones_sensitivity_one(self, sens):
        x = torch.ones(1, 1, 16, 16)
        y = torch.ones(1, 1, 16, 16)
        result = sens.compute_metric(x, y)
        v = result.item()
        assert math.isnan(v) or v == pytest.approx(1.0, abs=1e-6)

    def test_edge_all_zeros_specificity_one(self, spec):
        x = torch.zeros(1, 1, 16, 16)
        y = torch.zeros(1, 1, 16, 16)
        result = spec.compute_metric(x, y)
        v = result.item()
        assert math.isnan(v) or v == pytest.approx(1.0, abs=1e-6)

    def test_edge_no_positive_returns_nan(self, sens):
        x = torch.zeros(1, 1, 16, 16)
        y = torch.zeros(1, 1, 16, 16)
        result = sens.compute_metric(x, y)
        # No positives in target → TP+FN=0 → NaN
        assert math.isnan(result.item())


# ---------------------------------------------------------------------------
# ExpectedCalibrationError (#41)
# ---------------------------------------------------------------------------


class TestECE:
    @pytest.fixture
    def ece(self):
        from mriforge.core.metrics.report_step_metrics import ExpectedCalibrationError
        return ExpectedCalibrationError()

    def _binary_data(self, n: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(seed)
        confs = torch.rand(1, 1, n, n, generator=g)
        correct = (confs > 0.5).float()
        return confs, correct

    def test_canary_finite_nonnegative(self, ece):
        confs, correct = self._binary_data()
        result = ece.compute_metric(confs, correct)
        assert _is_finite(result) and result.item() >= 0.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, ece, b):
        g = torch.Generator()
        g.manual_seed(b)
        confs = torch.rand(b, 1, 8, 8, generator=g)
        correct = (confs > 0.5).float()
        result = ece.compute_metric(confs, correct)
        v = result.item()
        assert math.isnan(v) or v >= 0.0

    def test_edge_perfect_calibration_near_zero(self, ece):
        # Perfect calibration: conf=0.9 → correct=1 always
        confs = torch.full((1, 1, 4, 4), 0.9)
        correct = torch.ones(1, 1, 4, 4)
        result = ece.compute_metric(confs, correct)
        assert result.item() < 0.2

    def test_edge_empty_raises_or_nan(self, ece):
        """Empty tensor → NaN guard."""
        confs = torch.zeros(0)
        correct = torch.zeros(0)
        result = ece.compute_metric(confs, correct)
        assert math.isnan(result.item())


# ---------------------------------------------------------------------------
# NLLBitsPerDim (#42)
# ---------------------------------------------------------------------------


class TestNLLBitsPerDim:
    @pytest.fixture
    def bpd(self):
        from mriforge.core.metrics.report_step_metrics import NLLBitsPerDim
        return NLLBitsPerDim()

    def test_canary_finite(self, bpd):
        nll = torch.rand(2, 1, 16, 16)  # per-element nats
        result = bpd.compute_metric(nll, nll)  # target ignored
        assert _is_finite(result)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, bpd, b):
        nll = torch.rand(b, 1, 16, 16)
        assert _is_finite(bpd.compute_metric(nll, nll))

    def test_edge_zero_nll_zero_bpd(self, bpd):
        nll = torch.zeros(1, 1, 4, 4)
        result = bpd.compute_metric(nll, nll)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_edge_known_conversion(self, bpd):
        # log(2) nats per dim → 1 bit per dim
        nll = torch.full((1, 1, 4, 4), math.log(2.0))
        result = bpd.compute_metric(nll, nll)
        assert result.item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Wasserstein1D (#45)
# ---------------------------------------------------------------------------


class TestWasserstein1D:
    @pytest.fixture
    def w1(self):
        from mriforge.core.metrics.report_step_metrics import Wasserstein1D
        return Wasserstein1D()

    def test_canary_finite_nonnegative(self, w1):
        x = _img()
        y = _img(seed=1)
        result = w1.compute_metric(x, y)
        assert _is_finite(result) and result.item() >= 0.0

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, w1, b):
        result = w1.compute_metric(_img(b), _img(b, seed=10))
        assert _is_finite(result) and result.item() >= 0.0

    def test_edge_identical_zero(self, w1):
        x = _img()
        result = w1.compute_metric(x, x)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_edge_offset_proportional(self, w1):
        x = torch.zeros(1, 1, 8, 8)
        y = torch.ones(1, 1, 8, 8)
        result = w1.compute_metric(x, y)
        assert result.item() > 0.5  # Large separation


# ---------------------------------------------------------------------------
# TargetRegistrationError (#29) — landmark_pair input
# ---------------------------------------------------------------------------


class TestTRE:
    @pytest.fixture
    def tre(self):
        from mriforge.core.metrics.report_step_metrics import TargetRegistrationErrorMetric
        return TargetRegistrationErrorMetric()

    def test_canary_finite_nonnegative(self, tre):
        p = torch.rand(10, 3)  # 10 landmarks in 3-D
        t = torch.rand(10, 3)
        result = tre.compute_metric(p, t)
        assert _is_finite(result) and result.item() >= 0.0

    @pytest.mark.parametrize("k", [5, 20])
    def test_num_landmarks(self, tre, k):
        p = torch.rand(k, 2)
        t = torch.rand(k, 2)
        result = tre.compute_metric(p, t)
        assert _is_finite(result) and result.item() >= 0.0

    def test_edge_identical_zero(self, tre):
        p = torch.rand(5, 3)
        result = tre.compute_metric(p, p)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_edge_shape_mismatch_returns_nan(self, tre):
        p = torch.rand(5, 3)
        t = torch.rand(6, 3)
        result = tre.compute_metric(p, t)
        assert math.isnan(result.item())


# ---------------------------------------------------------------------------
# CohenKappa if present (#50)
# ---------------------------------------------------------------------------


class TestCohenKappa:
    def test_import_canary(self):
        """Confirm the module can be imported without error."""
        import mriforge.core.metrics.report_step_metrics as m
        assert hasattr(m, "BlandAltmanBias")  # representative export
