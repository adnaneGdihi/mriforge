"""Unit tests for HPO Use Case."""

from unittest.mock import MagicMock, patch

import pytest

from spectramr.application.use_cases.hpo_use_case import HPORequest, HPOUseCase
from spectramr.infrastructure.services.logging_service import LoggingService


@pytest.fixture
def mock_logging_service():
    """Mock logging service."""
    return MagicMock(spec=LoggingService)


@pytest.fixture
def use_case(mock_logging_service):
    """Create HPOUseCase instance."""
    return HPOUseCase(mock_logging_service)


class TestHPOUseCase:
    """Test HPOUseCase functionality."""

    @patch("spectramr.application.use_cases.hpo_use_case.TrainingSettings")
    @patch("spectramr.application.use_cases.hpo_use_case.HPOCoordinator")
    def test_execute_success(self, mock_coordinator_cls, mock_settings_cls, use_case):
        """Test successful execution of HPO."""
        # Mock settings. The path is `data.source.root`, NOT the retired
        # one-level `data.data_root`: the block decomposition moved it, and a
        # bare MagicMock auto-vivifies the wrong spelling instead of raising,
        # so this assertion compared a mock against a string for as long as the
        # stale name sat here.
        mock_settings = MagicMock()
        mock_settings.data.source.root = "/tmp/data"
        mock_settings.training.output_dir = "/tmp/output"
        mock_settings_cls.from_yaml.return_value = mock_settings

        # Mock coordinator
        mock_coordinator = mock_coordinator_cls.return_value
        mock_coordinator.execute_hpo.return_value = {"best_trial": 1}

        request = HPORequest(
            config_path="config.yaml",
            model_types=["unet"],
        )

        response = use_case.execute(request)

        assert response.success is True
        assert response.results == {"best_trial": 1}

        # Verify coordinator called with correct args
        mock_coordinator.execute_hpo.assert_called_once()
        call_kwargs = mock_coordinator.execute_hpo.call_args[1]
        assert call_kwargs["input_lr_dir"] == "/tmp/data"
        # Output goes under <training.output_dir>/hpo so trial artifacts
        # don't collide with the parent training run's output directory.
        assert call_kwargs["output_dir"] == "/tmp/output/hpo"
        # New: HPO knobs forwarded from request to coordinator
        assert call_kwargs["sampler_type"] == "tpe"
        assert call_kwargs["pruner_type"] == "hyperband"
        assert call_kwargs["n_trials"] == 50
        assert call_kwargs["objective_metric"] == "val_loss"
        assert call_kwargs["max_iter_per_trial"] == 30000

    @patch("spectramr.application.use_cases.hpo_use_case.TrainingSettings")
    @patch("spectramr.application.use_cases.hpo_use_case.HPOCoordinator")
    def test_execute_request_overrides_propagate(
        self, mock_coordinator_cls, mock_settings_cls, use_case
    ):
        """Per-request HPO knobs propagate to the coordinator (no hardcoding)."""
        mock_settings = MagicMock()
        # Same canonical path as above. Latently stale rather than failing here
        # only because this case never asserts on ``input_lr_dir``.
        mock_settings.data.source.root = "/tmp/data"
        mock_settings.training.output_dir = "/tmp/output"
        mock_settings_cls.from_yaml.return_value = mock_settings
        mock_coordinator = mock_coordinator_cls.return_value
        mock_coordinator.execute_hpo.return_value = {}

        request = HPORequest(
            config_path="config.yaml",
            model_types=["foo"],
            n_trials=123,
            objective_metric="val_psnr",
            sampler_type="cmaes",
            pruner_type="median",
            storage_url="sqlite:///x.db",
            output_dir="/explicit/dir",
            max_iter_per_trial=12345,
            cost_weight=0.3,
            enable_cost_optimization=True,
        )
        use_case.execute(request)
        call_kwargs = mock_coordinator.execute_hpo.call_args[1]
        assert call_kwargs["n_trials"] == 123
        assert call_kwargs["objective_metric"] == "val_psnr"
        assert call_kwargs["sampler_type"] == "cmaes"
        assert call_kwargs["pruner_type"] == "median"
        assert call_kwargs["storage_url"] == "sqlite:///x.db"
        # Explicit output_dir wins over the config-derived default
        assert call_kwargs["output_dir"] == "/explicit/dir"
        assert call_kwargs["max_iter_per_trial"] == 12345
        assert call_kwargs["cost_weight"] == 0.3
        assert call_kwargs["enable_cost_optimization"] is True

    def test_execute_validation_error(self, use_case):
        """Test request validation."""
        request = HPORequest(config_path="", model_types=[])

        with pytest.raises(ValueError, match="config_path must be provided"):
            use_case.execute(request)

    def test_execute_validation_n_trials(self, use_case):
        """n_trials < 1 raises before any work is done."""
        request = HPORequest(
            config_path="x.yaml", model_types=["foo"], n_trials=0
        )
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            use_case.execute(request)

    @patch("spectramr.application.use_cases.hpo_use_case.TrainingSettings")
    def test_execute_failure(self, mock_settings_cls, use_case):
        """Test handling of execution failure."""
        mock_settings_cls.from_yaml.side_effect = Exception("Config error")

        request = HPORequest(config_path="config.yaml", model_types=["unet"])

        response = use_case.execute(request)

        assert response.success is False
        assert response.results == {}
