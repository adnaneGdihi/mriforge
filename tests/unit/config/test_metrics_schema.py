import pytest
from pydantic import ValidationError

from spectramr.config.schemas.enums import MetricMode
from spectramr.config.schemas.metrics import MetricsConfigSchema, ProfilingConfigSchema


class TestMetricsConfigSchema:
    def test_defaults(self):
        schema = MetricsConfigSchema()
        assert schema.enable_tracking is True
        assert schema.compute_psnr is True
        assert schema.compute_ssim is True
        assert schema.metric_interval == 1000
        assert schema.best_metric_mode == MetricMode.MIN

    def test_custom_values(self):
        schema = MetricsConfigSchema(
            compute_fid=True,
            metric_interval=500,
            best_metric_name="psnr",
            best_metric_mode=MetricMode.MAX,
        )
        assert schema.compute_fid is True
        assert schema.metric_interval == 500
        assert schema.best_metric_name == "psnr"
        assert schema.best_metric_mode == MetricMode.MAX

    def test_validation(self):
        with pytest.raises(ValidationError):
            MetricsConfigSchema(metric_interval=0)

    def test_paradigm_specific_metrics(self):
        metrics = {"psnr": True, "mae": False}
        schema = MetricsConfigSchema(paradigm_specific_metrics=metrics)
        assert schema.paradigm_specific_metrics == metrics


class TestProfilingConfigSchema:
    def test_defaults(self):
        schema = ProfilingConfigSchema()
        assert schema.enabled is False
        assert schema.profile_interval == 100

    def test_custom_values(self):
        schema = ProfilingConfigSchema(
            enabled=True, profile_memory=True, profile_interval=50
        )
        assert schema.enabled is True
        assert schema.profile_memory is True
        assert schema.profile_interval == 50

    def test_validation(self):
        with pytest.raises(ValidationError):
            ProfilingConfigSchema(profile_interval=0)
