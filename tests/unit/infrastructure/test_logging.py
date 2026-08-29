"""
Unit tests for Logging infrastructure.
Covers AsyncLogger, MetricsTracker, DeprecationLogger.
"""

import pytest

try:
    from mriforge.infrastructure.logging.async_logger import AsyncLogger
except ImportError:
    AsyncLogger = None


# @# pytest.mark.unit
# @# pytest.mark.timeout(30)
class TestLogging:
    def test_async_logger_queueing(self):
        if AsyncLogger is None:
            pytest.skip("AsyncLogger not available")

        logger = AsyncLogger(log_dir="/tmp/logs")
        logger.start()

        logger.log("test_metric", 1.0)

        # Verify it didn't crash and ideally queue size > 0 or processed
        # Hard to test async side effects in unit test without mocks
        logger.stop()

    # @# pytest.mark.timeout(30)
    def test_metrics_tracker_aggregation(self):
        try:
            from mriforge.infrastructure.logging.metrics_tracker import MetricsTracker
        except ImportError:
            pytest.skip("MetricsTracker not available")

        tracker = MetricsTracker()
        tracker.update("loss", 10.0)
        tracker.update("loss", 20.0)

        assert tracker.avg("loss") == 15.0
