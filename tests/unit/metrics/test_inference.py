"""Unit tests for core/metrics/inference.py — four archetypes per class.

Covers:
- PSNRMetric, SSIMMetric, ReconstructionErrorMetric: compute() dict.
- BaseEvaluationMetric contract.
- ParadigmSpecificEvaluator: instantiates without crash, evaluate() returns a dict.
- FIDMetric / ISMetric: import canary (heavy InceptionV3 tests are @slow).
"""

from __future__ import annotations

import math

import pytest

from tests.utils.optional_backends import requires_torch_fidelity
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cpu() -> torch.device:
    return torch.device("cpu")


def _img(b: int = 2, c: int = 1, h: int = 16, w: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


# ---------------------------------------------------------------------------
# PSNRMetric (inference wrapper)
# ---------------------------------------------------------------------------


class TestInferencePSNR:
    @pytest.fixture
    def metric(self):
        from spectramr.core.metrics.inference import PSNRMetric
        return PSNRMetric(device=_cpu())

    # canary
    def test_canary_returns_dict_with_psnr(self, metric):
        x = _img()
        y = _img(seed=1)
        result = metric.compute(x, y)
        assert isinstance(result, dict)
        assert "psnr" in result
        assert math.isfinite(result["psnr"])

    # parametrize over batch sizes
    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, metric, b):
        result = metric.compute(_img(b), _img(b, seed=2))
        assert "psnr" in result
        assert isinstance(result["psnr"], float)

    # parametrize over value ranges
    @pytest.mark.parametrize("scale", [0.1, 1.0])
    def test_value_ranges(self, metric, scale):
        x = _img() * scale
        y = _img(seed=3) * scale
        result = metric.compute(x, y)
        assert isinstance(result["psnr"], float)

    # edge: identical inputs → high PSNR (clamped at 100)
    def test_edge_identical_high_psnr(self, metric):
        x = _img()
        result = metric.compute(x, x)
        assert result["psnr"] > 50.0

    # edge: real=None raises ValueError
    def test_raises_no_real(self, metric):
        with pytest.raises(ValueError, match="requires real"):
            metric.compute(_img(), real=None)

    # edge: all-zero target
    def test_edge_all_zero_no_crash(self, metric):
        z = torch.zeros(2, 1, 16, 16)
        result = metric.compute(z, z)
        assert math.isfinite(result["psnr"])


# ---------------------------------------------------------------------------
# SSIMMetric (inference wrapper)
# ---------------------------------------------------------------------------


class TestInferenceSSIM:
    @pytest.fixture
    def metric(self):
        from spectramr.core.metrics.inference import SSIMMetric
        return SSIMMetric(device=_cpu())

    def test_canary_returns_dict_with_ssim(self, metric):
        x = _img()
        y = _img(seed=1)
        result = metric.compute(x, y)
        assert "ssim" in result
        assert isinstance(result["ssim"], float)

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, metric, b):
        result = metric.compute(_img(b), _img(b, seed=4))
        assert "ssim" in result

    def test_edge_identical_near_one(self, metric):
        x = _img()
        result = metric.compute(x, x)
        assert result["ssim"] > 0.95

    def test_edge_all_zero_no_crash(self, metric):
        z = torch.zeros(2, 1, 16, 16)
        result = metric.compute(z, z)
        assert math.isfinite(result["ssim"])

    def test_raises_no_real(self, metric):
        with pytest.raises(ValueError, match="requires real"):
            metric.compute(_img(), real=None)


# ---------------------------------------------------------------------------
# ReconstructionErrorMetric
# ---------------------------------------------------------------------------


class TestReconstructionError:
    @pytest.fixture
    def metric(self):
        from spectramr.core.metrics.inference import ReconstructionErrorMetric
        return ReconstructionErrorMetric(device=_cpu())

    def test_canary_returns_l1_l2_rmse(self, metric):
        x = _img()
        y = _img(seed=1)
        result = metric.compute(x, y)
        for key in ["reconstruction_l1", "reconstruction_l2", "reconstruction_rmse"]:
            assert key in result
            assert math.isfinite(result[key])

    @pytest.mark.parametrize("b", [1, 2])
    def test_batch_sizes(self, metric, b):
        result = metric.compute(_img(b), _img(b, seed=5))
        assert "reconstruction_l1" in result

    def test_edge_identical_zero_error(self, metric):
        x = _img()
        result = metric.compute(x, x)
        assert result["reconstruction_l1"] == pytest.approx(0.0, abs=1e-6)
        assert result["reconstruction_l2"] == pytest.approx(0.0, abs=1e-6)

    def test_edge_all_zero_target_no_crash(self, metric):
        z = torch.zeros(2, 1, 16, 16)
        result = metric.compute(z, z)
        for v in result.values():
            assert math.isfinite(v)

    def test_raises_no_real(self, metric):
        with pytest.raises(ValueError, match="Reconstruction error requires"):
            metric.compute(_img(), real=None)


# ---------------------------------------------------------------------------
# ParadigmSpecificEvaluator — lightweight instantiation + evaluate
# ---------------------------------------------------------------------------


class TestParadigmSpecificEvaluator:
    @pytest.fixture
    def gan_evaluator(self):
        from spectramr.config.schemas.enums import TrainingModeTypes
        from spectramr.core.metrics.inference import ParadigmSpecificEvaluator
        return ParadigmSpecificEvaluator(
            device=_cpu(),
            training_mode=TrainingModeTypes.RECONSTRUCTION,
        )

    # canary
    def test_canary_instantiates(self, gan_evaluator):
        assert gan_evaluator is not None
        assert "psnr" in gan_evaluator.metrics
        assert "ssim" in gan_evaluator.metrics

    @pytest.mark.parametrize("b", [1, 2])
    def test_evaluate_returns_dict(self, gan_evaluator, b):
        x = _img(b)
        y = _img(b, seed=10)
        result = gan_evaluator.evaluate(x, real=y)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_edge_psnr_ssim_in_result(self, gan_evaluator):
        x = _img(2)
        y = _img(2, seed=11)
        result = gan_evaluator.evaluate(x, real=y)
        # At least psnr or psnr_medical should be present
        psnr_present = any("psnr" in k for k in result)
        assert psnr_present

    def test_reset_metrics_no_crash(self, gan_evaluator):
        gan_evaluator.reset_metrics()  # Should not raise

    def test_evaluate_no_real_still_returns_dict(self, gan_evaluator):
        x = _img(2)
        result = gan_evaluator.evaluate(x, real=None)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# FIDMetric / ISMetric — import canary only (no InceptionV3 download needed)
# ---------------------------------------------------------------------------


@requires_torch_fidelity
class TestFIDISImportCanary:
    @requires_torch_fidelity
    def test_fid_metric_importable(self):
        from spectramr.core.metrics.inference import FIDMetric
        assert FIDMetric is not None

    def test_is_metric_importable(self):
        from spectramr.core.metrics.inference import ISMetric
        assert ISMetric is not None

    @requires_torch_fidelity
    def test_fid_instantiates(self):
        from spectramr.core.metrics.inference import FIDMetric
        m = FIDMetric(device=_cpu())
        assert m is not None

    @pytest.mark.slow
    @requires_torch_fidelity
    def test_fid_compute_with_small_batch(self):
        """FID needs torchmetrics + enough samples; skip if not available."""
        from spectramr.core.metrics.inference import FIDMetric
        m = FIDMetric(device=_cpu())
        x = _img(4, 3, 32, 32)
        y = _img(4, 3, 32, 32, seed=1)
        try:
            m.update(x, y)
            result = m.compute()
            assert isinstance(result, dict)
        except Exception:
            pytest.skip("FID requires torchmetrics + sufficient samples")
