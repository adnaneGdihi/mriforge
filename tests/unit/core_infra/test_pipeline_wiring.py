"""Unit tests for the top-level pipeline orchestration."""

from unittest.mock import MagicMock, patch

import pytest

from spectramr.config.settings import TrainingSettings
from spectramr.pipelines.train import run_training_pipeline


class TestPipelineWiring:
    """Verify orchestration from config down to strategy execution."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=TrainingSettings)
        config.seed = 42
        config.training = MagicMock()
        config.training.seed = 42
        config.training.output_dir = "/tmp/fake_output"
        config.logging = MagicMock()
        config.logging.run_name = "test_run"
        config.logging.tracking_service = "none"
        config.logging.enable_experiment_tracking = False
        config.metadata = {"name": "test_run"}
        config.model = MagicMock()
        config.model.model_type = "dummy_model"
        config.model.target_domain = "image"
        config.data = MagicMock()
        config.data.loader.batch_size = 32
        config.optimization = MagicMock()
        config.optimization.gradient.clip.value = 1.0
        config.optimization.gradient.detect_anomalies = False
        config.metrics = None
        config.losses = MagicMock()
        config.losses.get_enabled_losses.return_value = {"l1": 1.0}
        return config

    @patch("spectramr.pipelines.train.set_global_seed")
    @patch("spectramr.pipelines.train.validate_config_health")
    @patch("spectramr.pipelines.train.bootstrap")
    @patch("spectramr.pipelines.train.TrainingEnvironmentDirector")
    @patch("spectramr.pipelines.train.TrainingStrategyFactory")
    @patch("spectramr.pipelines.training_loop._execute_training_loop")
    def test_run_training_pipeline_wiring(
        self,
        mock_execute_loop,
        mock_strategy_factory_cls,
        mock_director_cls,
        mock_bootstrap,
        mock_validate,
        mock_set_seed,
        mock_config,
    ):
        """Test that the pipeline successfully instantiates the DI container, strategy, and loop."""

        # Setup mocks. validate_config_health now returns a HealthCheckReport,
        # not a bool — pipeline filters report.errors by check_name to fail
        # fast on domain mismatches.
        mock_report = MagicMock()
        mock_report.passed = True
        mock_report.errors = []
        mock_validate.return_value = mock_report

        mock_container = MagicMock()
        mock_logging = MagicMock()
        mock_checkpoint = MagicMock()
        mock_metrics = MagicMock()
        mock_memory = MagicMock()

        def resolve_side_effect(interface):
            name = interface.__name__
            if "Logging" in name:
                return mock_logging
            if "Checkpoint" in name:
                return mock_checkpoint
            if "Metrics" in name:
                return mock_metrics
            if "Memory" in name:
                return mock_memory
            return MagicMock()

        mock_container.resolve.side_effect = resolve_side_effect
        mock_bootstrap.build_container.return_value = mock_container

        mock_director = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.device = "cpu"
        mock_pipeline.models = {"generator": MagicMock(), "discriminator": MagicMock()}
        mock_pipeline.optimizers = {"main": MagicMock()}
        mock_pipeline.losses = {"primary": MagicMock()}
        mock_pipeline.data_loaders = {"train": [1, 2, 3]}

        mock_director.build_environment.return_value = mock_pipeline
        mock_director_cls.return_value = mock_director

        mock_strategy_factory = MagicMock()
        mock_strategy = MagicMock()
        mock_strategy_factory.create_strategy.return_value = mock_strategy
        mock_strategy_factory_cls.return_value = mock_strategy_factory

        mock_execute_loop.return_value = {"success": True}

        # Run pipeline
        result = run_training_pipeline(mock_config, device="cpu", is_sanity_check=True)

        # Asserts — the loop's success propagates. (run_training_pipeline augments
        # the loop result with wall-clock duration/throughput, so assert the
        # success flag rather than exact-equality with the mock's bare return.)
        assert result["success"] is True

        # 1. Bootstrapping
        mock_bootstrap.build_container.assert_called_once_with(
            mock_config, device="cpu"
        )
        assert mock_container.resolve.call_count >= 4

        # 2. Environment Director
        mock_director_cls.assert_called_once_with(mock_config)
        mock_director.build_environment.assert_called_once()

        # 3. Strategy Factory
        mock_strategy_factory.create_strategy.assert_called_once_with(
            env=mock_pipeline,
            logging_service=mock_logging,
            metrics_service=mock_metrics,
            checkpoint_service=mock_checkpoint,
        )

        # 4. Training Loop execution
        mock_execute_loop.assert_called_once()

        call_args = mock_execute_loop.call_args.args
        call_kwargs = mock_execute_loop.call_args.kwargs

        assert call_args[0] == mock_strategy
        assert call_args[1] == mock_pipeline
        assert call_kwargs["metrics_service"] == mock_metrics
        assert call_kwargs["is_sanity_check"] is True
