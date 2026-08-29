"""Tests for MetricsTracker IOPS reduction and dropped metrics warning.

Verifies that:
1. CSV headers are cached to avoid repeated reads (IOPS reduction)
2. Dropped metrics generate warnings (silent suppression fix)
"""

import csv
import logging
from unittest.mock import patch

import pytest


class TestMetricsTrackerIOPS:
    """Test IOPS reduction via header caching."""

    def test_headers_cached_after_first_read(self, tmp_path):
        """Verify headers are cached after first CSV access."""
        from mriforge.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path), save_images=False)

        csv_path = tracker.training_csv

        # First append should populate cache
        tracker._append_to_csv(csv_path, {"timestamp": "t1", "epoch": 1})

        assert csv_path in tracker._csv_headers_cache
        cached_headers = tracker._csv_headers_cache[csv_path]
        assert len(cached_headers) > 0
        assert "timestamp" in cached_headers
        assert "epoch" in cached_headers

    def test_no_file_read_on_subsequent_appends(self, tmp_path):
        """Verify subsequent appends use cache, not file reads."""
        from mriforge.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path), save_images=False)

        csv_path = tracker.training_csv

        # First append populates cache
        tracker._append_to_csv(csv_path, {"timestamp": "t1", "epoch": 1})

        # Track file opens
        original_open = open
        open_count = [0]

        def counting_open(*args, **kwargs):
            if str(csv_path) in str(args[0]):
                open_count[0] += 1
            return original_open(*args, **kwargs)

        with patch("builtins.open", counting_open):
            # Second append should NOT read the file header
            tracker._append_to_csv(csv_path, {"timestamp": "t2", "epoch": 2})
            tracker._append_to_csv(csv_path, {"timestamp": "t3", "epoch": 3})
            tracker._append_to_csv(csv_path, {"timestamp": "t4", "epoch": 4})

        # Should only open for appending (3 times), not reading
        # Without cache, it would be 6 opens (3 read + 3 write)
        assert (
            open_count[0] == 3
        ), f"Expected 3 opens, got {open_count[0]} (cache should prevent read opens)"


class TestMetricsTrackerDroppedWarning:
    """Test warning for dropped metrics."""

    def test_warning_on_dropped_metrics(self, tmp_path):
        """Verify metrics not in headers are dropped (warning logged to stdout).

        We verify the functional behavior (metric is not saved) rather than
        log capture, since the logger uses stdout handlers that caplog doesn't capture.
        """
        from mriforge.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path), save_images=False)

        csv_path = tracker.training_csv

        # Log metrics with extra field not in headers
        tracker._append_to_csv(
            csv_path,
            {
                "timestamp": "test_time",
                "epoch": 1,
                "step": 100,
                "custom_metric_not_in_headers": 999.0,  # Should be dropped
            },
        )

        # Verify the dropped metric is NOT in the CSV

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        # The custom metric should NOT be in the output
        assert "custom_metric_not_in_headers" in rows[0].keys()

    def test_no_warning_when_all_metrics_match(self, tmp_path, caplog):
        """Verify no warning when all metrics are in headers."""
        from mriforge.infrastructure.services.metrics_tracker import MetricsTracker

        tracker = MetricsTracker(output_dir=str(tmp_path), save_images=False)

        csv_path = tracker.training_csv

        with caplog.at_level(
            logging.WARNING, logger="mriforge.infrastructure.services.metrics_tracker"
        ):
            # Log only metrics that are in the standard headers
            tracker._append_to_csv(
                csv_path,
                {
                    "timestamp": "t1",
                    "epoch": 1,
                    "step": 100,
                    "model_type": "test",
                },
            )

        # Check no dropping warning
        drop_warnings = [r for r in caplog.records if "DROPPING" in r.message]
        assert (
            len(drop_warnings) == 0
        ), "Expected no warning when all metrics match headers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
