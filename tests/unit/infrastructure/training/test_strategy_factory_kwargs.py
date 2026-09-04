"""Unit tests for TrainingStrategyFactory metrics_service passing.

This test suite verifies that:
1. metrics_service is properly passed to strategy
2. checkpoint_service is properly passed to strategy
3. kwargs are forwarded correctly
4. Strategy can access services after instantiation
"""

from unittest.mock import MagicMock, patch

import pytest

from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, env, logging_service=None, **kwargs):
        self.env = env
        self.logging_service = logging_service
        self.metrics_service = kwargs.get("metrics_service")
        self.checkpoint_service = kwargs.get("checkpoint_service")
        self.extra_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["metrics_service", "checkpoint_service"]
        }


class TestTrainingStrategyFactoryKwargsForwarding:
    """Test suite for TrainingStrategyFactory kwargs forwarding."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline/environment."""
        pipeline = MagicMock()

        # Create a minimal config
        config_dict = {
            "config_version": "6.0",
            "metadata": {"name": "test"},
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
            },
            "training": {"training_mode": "gan"},
            "optimization": {},
            "data": {},
            "logging": {},
            "checkpoint": {},
        }

        # Create mock config
        pipeline.config = MagicMock()
        pipeline.config.training.training_mode = "gan"
        pipeline.config.model.model_type = "standard_unet"

        return pipeline

    @pytest.fixture
    def factory(self):
        """Create TrainingStrategyFactory."""
        return TrainingStrategyFactory()

    def test_factory_forwards_metrics_service(self, factory, mock_pipeline):
        """Test that metrics_service kwarg is forwarded to strategy."""
        logging_service = MagicMock()
        metrics_service = MagicMock()

        # Patch to use our MockStrategy
        with patch.object(factory, "get_strategy_class", return_value=MockStrategy):
            strategy = factory.create_strategy(
                env=mock_pipeline,
                logging_service=logging_service,
                metrics_service=metrics_service,
            )

        # ✅ metrics_service should be passed through
        assert strategy.metrics_service is metrics_service
        assert strategy.logging_service is logging_service

    def test_factory_forwards_checkpoint_service(self, factory, mock_pipeline):
        """Test that checkpoint_service kwarg is forwarded to strategy."""
        logging_service = MagicMock()
        checkpoint_service = MagicMock()

        with patch.object(factory, "get_strategy_class", return_value=MockStrategy):
            strategy = factory.create_strategy(
                env=mock_pipeline,
                logging_service=logging_service,
                checkpoint_service=checkpoint_service,
            )

        # ✅ checkpoint_service should be passed through
        assert strategy.checkpoint_service is checkpoint_service

    def test_factory_forwards_multiple_kwargs(self, factory, mock_pipeline):
        """Test that multiple kwargs are all forwarded."""
        logging_service = MagicMock()
        metrics_service = MagicMock()
        checkpoint_service = MagicMock()
        extra_service = MagicMock()

        with patch.object(factory, "get_strategy_class", return_value=MockStrategy):
            strategy = factory.create_strategy(
                env=mock_pipeline,
                logging_service=logging_service,
                metrics_service=metrics_service,
                checkpoint_service=checkpoint_service,
                extra_service=extra_service,
            )

        # ✅ All kwargs should be available
        assert strategy.metrics_service is metrics_service
        assert strategy.checkpoint_service is checkpoint_service
        assert strategy.extra_kwargs["extra_service"] is extra_service

    def test_factory_handles_missing_kwargs(self, factory, mock_pipeline):
        """Test that factory works fine when some kwargs are missing."""
        logging_service = MagicMock()

        with patch.object(factory, "get_strategy_class", return_value=MockStrategy):
            strategy = factory.create_strategy(
                env=mock_pipeline,
                logging_service=logging_service,
                # No metrics_service or checkpoint_service provided
            )

        # ✅ Strategy should handle None gracefully
        assert strategy.metrics_service is None
        assert strategy.checkpoint_service is None
        assert strategy.logging_service is logging_service

    def test_factory_signature_accepts_kwargs(self, factory):
        """Test that create_strategy method signature accepts **kwargs."""
        import inspect

        sig = inspect.signature(factory.create_strategy)

        # ✅ Should have **kwargs parameter
        assert "kwargs" in str(sig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
