"""Tests for early stopping + best checkpoint integration.

Verifies that:
1. EarlyStoppingService.on_improvement callback fires on metric improvement
2. CheckpointDirector.save_best() produces checkpoint_best.pt
3. Callback is NOT fired when metric does not improve
4. Best checkpoint contains expected metadata (is_best, metric fields)
"""

from unittest.mock import MagicMock

import pytest
import torch

from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
from spectramr.config.schemas.enums import MetricMode


class TestEarlyStoppingBestCheckpointIntegration:
    """Test early stopping service improvement callback behaviour."""

    def _make_service(self, mode=MetricMode.MIN, on_improvement=None, metric=None):
        """Helper to create an enabled EarlyStoppingService.

        The metric now tracks the mode. `val_loss` is declared lower-is-better, so
        pairing it with MAX is the contradiction #712's validator rejects -- these
        tests exercise MAX *mechanics*, and the metric name was incidental.
        """
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        if metric is None:
            metric = "val_psnr" if str(mode).lower().endswith("max") else "val_loss"
        config = EarlyStoppingConfigSchema(
            enabled=True,
            patience=5,
            metric=metric,
            mode=mode,
            min_delta=0.0,
            check_interval=1,
        )
        return EarlyStoppingService(config, on_improvement=on_improvement)

    def test_on_improvement_callback_fires_on_better_metric(self):
        """Callback fires when metric improves (MIN mode: lower is better)."""
        callback = MagicMock()
        svc = self._make_service(mode=MetricMode.MIN, on_improvement=callback)

        svc.update(1.0, iteration=100)
        callback.assert_called_once_with(100, 1.0)

        callback.reset_mock()
        svc.update(0.5, iteration=200)
        callback.assert_called_once_with(200, 0.5)

    def test_on_improvement_callback_not_fired_on_worse_metric(self):
        """Callback does NOT fire when metric does not improve."""
        callback = MagicMock()
        svc = self._make_service(mode=MetricMode.MIN, on_improvement=callback)

        svc.update(1.0, iteration=100)
        callback.reset_mock()

        svc.update(1.5, iteration=200)  # Worse (higher in MIN mode)
        callback.assert_not_called()

    def test_wait_count_resets_on_improvement(self):
        """wait_count should be 0 after an improvement."""
        svc = self._make_service(mode=MetricMode.MIN)

        svc.update(1.0, iteration=100)
        assert svc.wait_count == 0

        svc.update(1.5, iteration=200)  # No improvement
        assert svc.wait_count == 1

        svc.update(0.8, iteration=300)  # Improvement!
        assert svc.wait_count == 0

    def test_on_improvement_callback_fires_max_mode(self):
        """Callback fires when metric improves (MAX mode: higher is better)."""
        callback = MagicMock()
        svc = self._make_service(mode=MetricMode.MAX, on_improvement=callback)

        svc.update(10.0, iteration=100)
        callback.assert_called_once_with(100, 10.0)

        callback.reset_mock()
        svc.update(15.0, iteration=200)  # Better (higher in MAX mode)
        callback.assert_called_once_with(200, 15.0)

        callback.reset_mock()
        svc.update(12.0, iteration=300)  # Worse
        callback.assert_not_called()


class TestCheckpointDirectorSaveBest:
    """Test CheckpointDirector.save_best() produces checkpoint_best.pt."""

    def _make_pipeline_mock(self):
        """Create a mock pipeline with a real generator/optimizer and every
        optional field forced to ``None`` so the CheckpointDirector emits a
        torch.save-able state dict (MagicMock instances pickle-fail)."""
        pipeline = MagicMock()

        # Generator with tiny state dict (real nn.Module so state_dict() works)
        gen = torch.nn.Linear(2, 2)
        pipeline.generator = gen
        pipeline.models = MagicMock()
        pipeline.models.get = MagicMock(
            side_effect=lambda k: gen if k == "generator" else None
        )

        # Real optimizer.
        pipeline.optimizer_g = torch.optim.SGD(gen.parameters(), lr=0.01)

        # Force every optional attribute the director inspects to None so the
        # auto-spec'd MagicMock fallbacks don't leak unpicklable sentinels into
        # the checkpoint dict.
        pipeline.optimizer_d = None
        pipeline.discriminator = None
        pipeline.scheduler_g = None
        pipeline.scheduler_d = None
        pipeline.ema = None
        pipeline.scaler = None

        return pipeline

    def _make_config_mock(self):
        """Create a minimal config mock."""
        config = MagicMock()
        config.checkpoint = MagicMock()
        config.checkpoint.enabled = True
        return config

    def test_save_best_creates_checkpoint_best_pt(self, tmp_path):
        """save_best() should create checkpoint_best.pt."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        config = self._make_config_mock()
        pipeline = self._make_pipeline_mock()

        director = CheckpointDirector(config)
        path = (
            director.with_checkpoint_dir(str(tmp_path))
            .with_pipeline(pipeline)
            .with_epoch(5)
            .with_global_step(1000)
            .with_metrics({"g_total_loss": 0.5})
            .validate()
            .save_best(metric_name="val_psnr", metric_value=32.5)
        )

        assert path == tmp_path / "checkpoint_best.pt"
        assert path.exists()

    def test_save_best_contains_metadata(self, tmp_path):
        """Best checkpoint should contain is_best flag and metric info."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        config = self._make_config_mock()
        pipeline = self._make_pipeline_mock()

        director = CheckpointDirector(config)
        path = (
            director.with_checkpoint_dir(str(tmp_path))
            .with_pipeline(pipeline)
            .with_epoch(3)
            .with_global_step(500)
            .with_metrics({"loss": 0.3})
            .validate()
            .save_best(metric_name="val_loss", metric_value=0.25)
        )

        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["is_best"] is True
        assert data["best_metric_name"] == "val_loss"
        assert data["best_metric_value"] == pytest.approx(0.25)
        assert data["epoch"] == 3
        assert data["global_step"] == 500
        assert "generator" in data

    def test_save_best_overwrites_previous(self, tmp_path):
        """Calling save_best() twice should overwrite the same file."""
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        config = self._make_config_mock()
        pipeline = self._make_pipeline_mock()

        # First save
        director1 = CheckpointDirector(config)
        director1.with_checkpoint_dir(str(tmp_path)).with_pipeline(pipeline).with_epoch(
            1
        ).with_global_step(100).with_metrics({}).validate().save_best(
            metric_name="val_loss", metric_value=1.0
        )

        # Second save (better metric)
        director2 = CheckpointDirector(config)
        director2.with_checkpoint_dir(str(tmp_path)).with_pipeline(pipeline).with_epoch(
            2
        ).with_global_step(200).with_metrics({}).validate().save_best(
            metric_name="val_loss", metric_value=0.5
        )

        # Should only have one checkpoint_best.pt
        best_files = list(tmp_path.glob("checkpoint_best.pt"))
        assert len(best_files) == 1

        # Should contain the second (better) save
        data = torch.load(best_files[0], map_location="cpu", weights_only=False)
        assert data["best_metric_value"] == pytest.approx(0.5)
        assert data["global_step"] == 200
