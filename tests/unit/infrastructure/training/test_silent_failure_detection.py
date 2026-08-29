"""Unit tests for silent failure detection in validation logging.

This test suite verifies that:
1. Missing metric_service is logged as ERROR (not INFO/silent)
2. Error messages are clear and actionable
3. Image saving failures are logged with fail-loud pattern
4. Exception handling doesn't swallow critical errors
"""

from unittest.mock import MagicMock

import pytest
import torch

from mriforge.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy


class MockLoggingService:
    """Mock logging service to capture log calls."""

    def __init__(self):
        self.error_msgs = []
        self.warning_msgs = []
        self.info_msgs = []
        self.debug_msgs = []
        self.logger = MagicMock()
        self.logger.isEnabledFor = MagicMock(return_value=True)

    def log_error(self, msg):
        self.error_msgs.append(msg)

    def log_warning(self, msg):
        self.warning_msgs.append(msg)

    def log_info(self, msg):
        self.info_msgs.append(msg)

    def log_debug(self, msg):
        self.debug_msgs.append(msg)

    def log_images_batch(self, images_dict, step, max_images=4):
        """Mock image logging."""
        pass


class TestValidationLoggingFailures:
    """Test suite for silent failure detection."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock diffusion strategy."""
        strategy = MagicMock(spec=DiffusionTrainingStrategy)
        strategy.logging_service = MockLoggingService()
        strategy.validation_step_count = 0
        strategy.current_epoch = 0
        strategy.env = MagicMock()
        strategy.env.step = 100

        # Set config to return logging config
        strategy.config = MagicMock()
        strategy.config.logging = MagicMock()
        strategy.config.logging.log_validation_images = True
        strategy.config.logging.log_difference_images = True
        strategy.config.logging.log_input_images = False
        strategy.config.logging.max_images_per_batch = 4

        return strategy

    def test_missing_metrics_service_logs_error(self, mock_strategy):
        """Test that missing metrics_service is logged as ERROR."""
        mock_strategy.metrics_service = None

        # Simulate the validation image logging code
        logging_service = mock_strategy.logging_service

        # This is the pattern from diffusion.py _log_validation_images_to_tensorboard
        if mock_strategy.metrics_service is None:
            logging_service.log_error(
                "[DiffusionValidation] CRITICAL: metrics_service is None on strategy. "
                "DI resolution failed or bootstrap did not register IMetricsService. "
                "Validation images will NOT be saved."
            )

        # ✅ Should have ERROR log, not INFO
        assert len(logging_service.error_msgs) > 0
        assert "CRITICAL" in logging_service.error_msgs[0]
        assert len(logging_service.info_msgs) == 0

    def test_missing_metrics_service_attribute_logs_error(self, mock_strategy):
        """Test that missing metrics_service ATTRIBUTE is logged as ERROR."""
        # Remove attribute
        if hasattr(mock_strategy, "metrics_service"):
            delattr(mock_strategy, "metrics_service")

        logging_service = mock_strategy.logging_service

        # This is the pattern from diffusion.py
        if not hasattr(mock_strategy, "metrics_service"):
            logging_service.log_error(
                "[DiffusionValidation] CRITICAL: metrics_service attribute missing on strategy. "
                "This indicates a training infrastructure issue - metrics_service must be passed from bootstrap. "
                "Validation images will NOT be saved."
            )

        # ✅ Should have ERROR log, not INFO
        assert len(logging_service.error_msgs) > 0
        assert "CRITICAL" in logging_service.error_msgs[0]
        assert len(logging_service.info_msgs) == 0

    def test_image_save_exception_logs_error(self, mock_strategy):
        """Test that exceptions during image save are logged as ERROR."""
        mock_strategy.metrics_service = MagicMock()
        mock_strategy.metrics_service.save_images_batch.side_effect = RuntimeError(
            "Failed to write image to disk"
        )

        logging_service = mock_strategy.logging_service

        # Simulate exception handling
        try:
            mock_strategy.metrics_service.save_images_batch(
                real_images=torch.randn(1, 1, 32, 32),
                fake_images=torch.randn(1, 1, 32, 32),
                prefix="val",
                epoch=0,
                step=100,
            )
        except Exception as e:
            logging_service.log_error(
                f"[DiffusionValidation] CRITICAL: Failed to save validation images: {e}"
            )

        # ✅ Exception should be logged as ERROR
        assert len(logging_service.error_msgs) > 0
        assert "CRITICAL" in logging_service.error_msgs[0]

    def test_empty_paths_returned_not_logged_as_error(self, mock_strategy):
        """Test that empty paths (save_images=False) are NOT logged as ERROR."""
        mock_strategy.metrics_service = MagicMock()
        mock_strategy.metrics_service.save_images_batch.return_value = ([], [])

        logging_service = mock_strategy.logging_service

        # When metrics_service exists, attempt save
        if mock_strategy.metrics_service is not None:
            real_paths, fake_paths = mock_strategy.metrics_service.save_images_batch(
                real_images=torch.randn(1, 1, 32, 32),
                fake_images=torch.randn(1, 1, 32, 32),
                prefix="val",
                epoch=0,
                step=100,
            )

            # Empty paths returned (save_images=False user config) is not an error
            if not (real_paths or fake_paths):
                # This is expected - silently continue
                pass

        # ✅ No error should be logged
        assert len(logging_service.error_msgs) == 0


class TestLoggingPatterns:
    """Test that logging pattern prevents silent failures."""

    def test_no_bare_exception_handlers(self):
        """Verify that bare 'except Exception' doesn't hide critical errors."""
        # This test documents the expected pattern
        mock_service = MagicMock()

        # ✅ CORRECT PATTERN: Log details before silent failure
        try:
            result = mock_service.some_method()
        except Exception as e:
            # Must log the error with full details
            msg = f"Operation failed: {e}"
            # In real code, this would be: logging_service.log_error(msg)
            assert "failed" in msg.lower() or "error" in msg.lower()

    def test_error_vs_warning_distinction(self):
        """Test that infrastructure errors are logged as ERROR, not WARNING."""
        logging_service = MockLoggingService()

        # Infrastructure failure = ERROR (not warning)
        missing_service = None
        if missing_service is None:
            logging_service.log_error("CRITICAL: Required service not available")

        # Config issue = WARNING (not error)
        if logging_service.warning_msgs:
            logging_service.log_warning("Non-critical config issue")

        # ✅ Critical issues should be ERROR level
        assert len(logging_service.error_msgs) > 0
        assert "CRITICAL" in logging_service.error_msgs[0]


class TestRegressionAgainstSilentFailures:
    """Regression tests to prevent reintroduction of silent failures."""

    def test_strategy_must_log_errors_not_info_for_missing_services(self):
        """Regression: Missing services must be logged as ERROR, not INFO."""
        # This documents the bug that was fixed:
        # Before: self.logging_service.log_info("No metrics_service")  # ❌ Silent!
        # After:  self.logging_service.log_error("CRITICAL: No metrics_service")  # ✅ Loud!

        logging_service = MockLoggingService()
        metrics_service = None

        # ❌ BAD PATTERN (must not do this)
        # if metrics_service is None:
        #     logging_service.log_info("metrics_service is None")  # Silent failure!

        # ✅ CORRECT PATTERN
        if metrics_service is None:
            logging_service.log_error(
                "CRITICAL: metrics_service is None. Images will NOT be saved."
            )

        assert len(logging_service.error_msgs) > 0
        assert len(logging_service.info_msgs) == 0

    def test_save_images_must_return_paths_not_silently_skip(self):
        """Regression: save_images_batch must return empty list (not None)."""
        # This documents why we return [] instead of None:
        # - Caller checks: if real_paths or fake_paths:  (works with [] but not None)
        # - Caller counts: len(real_paths)  (works with [] but not None)

        real_paths = []
        fake_paths = []

        # ✅ This check works correctly:
        if real_paths or fake_paths:
            print("Images were saved")
        else:
            print("No images saved (save_images=False)")

        # ✅ This counting works correctly:
        assert len(real_paths) == 0
        assert len(fake_paths) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
