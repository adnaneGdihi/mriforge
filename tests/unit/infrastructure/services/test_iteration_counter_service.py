"""Tests for IterationCounterService.

Comprehensive test suite covering:
- Basic increment and counter tracking
- Interval checking (checkpoint, log, validate, metrics)
- Hard stop conditions (global and per-epoch)
- State save/restore
- Edge cases and boundary conditions
"""

from unittest.mock import Mock

import pytest

from mriforge.infrastructure.services.iteration_counter_service import (
    IterationCounterService,
)


class TestIterationCounterBasics:
    """Test basic counter functionality."""

    @pytest.fixture
    def config(self):
        """Create mock config with default values."""
        mock_config = Mock()
        mock_config.max_iterations = -1  # Unlimited
        mock_config.max_steps_per_epoch = -1  # Unlimited
        return mock_config

    @pytest.fixture
    def counter(self, config):
        """Create fresh counter for each test."""
        return IterationCounterService(config)

    def test_counter_initialization(self, counter):
        """Test counter initializes with zero values."""
        assert counter.current_step == 0
        assert counter.current_epoch == 0
        assert counter.steps_in_current_epoch == 0
        assert counter.total_steps_this_epoch == 0

    def test_increment_global_step(self, counter):
        """Test incrementing global step counter."""
        counter.increment()
        assert counter.current_step == 1

        counter.increment()
        assert counter.current_step == 2

        counter.increment()
        assert counter.current_step == 3

    def test_increment_epoch_step(self, counter):
        """Test incrementing per-epoch step counter."""
        counter.increment()
        assert counter.steps_in_current_epoch == 1

        counter.increment()
        assert counter.steps_in_current_epoch == 2

    def test_start_epoch(self, counter):
        """Test starting new epoch."""
        counter.increment()
        counter.increment()
        assert counter.steps_in_current_epoch == 2

        counter.start_epoch(total_steps_in_epoch=100)
        assert counter.current_epoch == 1
        assert counter.steps_in_current_epoch == 0
        assert counter.total_steps_this_epoch == 100

    def test_end_epoch(self, counter):
        """Test ending epoch."""
        counter.increment()
        assert counter.steps_in_current_epoch == 1

        counter.end_epoch()
        assert counter.steps_in_current_epoch == 0

    def test_multi_epoch_sequence(self, counter):
        """Test typical multi-epoch training sequence."""
        # Epoch 1: 3 steps
        counter.start_epoch(3)
        counter.increment()
        counter.increment()
        counter.increment()
        assert counter.current_epoch == 1
        assert counter.current_step == 3
        assert counter.steps_in_current_epoch == 3

        # Epoch 2: 3 steps
        counter.start_epoch(3)
        assert counter.current_epoch == 2
        assert counter.steps_in_current_epoch == 0
        counter.increment()
        counter.increment()
        counter.increment()
        assert counter.current_step == 6
        assert counter.steps_in_current_epoch == 3


class TestCheckpointInterval:
    """Test checkpoint interval checking."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = -1  # Unlimited
        mock_config.max_steps_per_epoch = -1
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_checkpoint_interval_basic(self, counter):
        """Test checkpoint at interval boundaries."""
        checkpoint_interval = 100

        # Before first interval
        counter.current_step = 50
        assert not counter.should_checkpoint(checkpoint_interval)

        # At interval boundary
        counter.current_step = 100
        assert counter.should_checkpoint(checkpoint_interval)

        # Beyond first interval, before second
        counter.current_step = 150
        assert not counter.should_checkpoint(checkpoint_interval)

        # At second interval
        counter.current_step = 200
        assert counter.should_checkpoint(checkpoint_interval)

    def test_checkpoint_at_step_0(self, counter):
        """Test checkpoint at step 0 (should trigger)."""
        counter.current_step = 0
        assert counter.should_checkpoint(1000)  # 0 % 1000 == 0

    def test_checkpoint_at_step_0_with_interval_1(self, counter):
        """Test checkpoint with interval=1 (every step)."""
        for step in range(5):
            counter.current_step = step
            assert counter.should_checkpoint(1)

    def test_checkpoint_respects_max_iterations(self, config, counter):
        """Test checkpoint respects max_iterations limit."""
        config.max_iterations = 1000
        counter.current_step = 1000

        # Even though 1000 % 1000 == 0, should not checkpoint
        # because current_step (1000) >= max_iterations (1000)
        assert not counter.should_checkpoint(1000)

        # Before limit should work
        counter.current_step = 900
        assert counter.should_checkpoint(900)


class TestLoggingInterval:
    """Test logging interval checking."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = -1
        mock_config.max_steps_per_epoch = -1
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_log_interval_basic(self, counter):
        """Test logging at interval boundaries."""
        log_interval = 50

        # Before first interval
        counter.current_step = 25
        assert not counter.should_log(log_interval)

        # At interval boundary
        counter.current_step = 50
        assert counter.should_log(log_interval)

        # Between intervals
        counter.current_step = 75
        assert not counter.should_log(log_interval)

        # At second interval
        counter.current_step = 100
        assert counter.should_log(log_interval)

    def test_log_respects_max_iterations(self, config, counter):
        """Test logging respects max_iterations limit."""
        config.max_iterations = 500
        log_interval = 100

        counter.current_step = 500
        assert not counter.should_log(log_interval)

        counter.current_step = 400
        assert counter.should_log(log_interval)


class TestValidationInterval:
    """Test validation interval checking (supports dual-mode)."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = -1
        mock_config.max_steps_per_epoch = -1
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_validation_step_based(self, counter):
        """Test validation with step-based interval."""
        eval_interval = 1000

        counter.current_step = 1000
        assert counter.should_validate(eval_interval, use_epoch_based=False)

        counter.current_step = 1500
        assert not counter.should_validate(eval_interval, use_epoch_based=False)

        counter.current_step = 2000
        assert counter.should_validate(eval_interval, use_epoch_based=False)

    def test_validation_epoch_based(self, counter):
        """Test validation with epoch-based interval."""
        frequency_epochs = 5

        counter.current_epoch = 5
        assert counter.should_validate(
            eval_interval=1000, frequency_epochs=frequency_epochs, use_epoch_based=True
        )

        counter.current_epoch = 6
        assert not counter.should_validate(
            eval_interval=1000, frequency_epochs=frequency_epochs, use_epoch_based=True
        )

        counter.current_epoch = 10
        assert counter.should_validate(
            eval_interval=1000, frequency_epochs=frequency_epochs, use_epoch_based=True
        )

    def test_validation_respects_max_iterations(self, config, counter):
        """Test validation respects max_iterations limit."""
        config.max_iterations = 2000

        counter.current_step = 2000
        assert not counter.should_validate(1000, use_epoch_based=False)

        counter.current_step = 1000
        assert counter.should_validate(1000, use_epoch_based=False)


class TestMetricsInterval:
    """Test metrics computation interval checking."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = -1
        mock_config.max_steps_per_epoch = -1
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_metrics_interval_basic(self, counter):
        """Test metrics at interval boundaries."""
        metric_interval = 500

        counter.current_step = 500
        assert counter.should_compute_metrics(metric_interval)

        counter.current_step = 501
        assert not counter.should_compute_metrics(metric_interval)

        counter.current_step = 1000
        assert counter.should_compute_metrics(metric_interval)

    def test_metrics_respects_max_iterations(self, config, counter):
        """Test metrics respects max_iterations limit."""
        config.max_iterations = 1000
        metric_interval = 200

        counter.current_step = 1000
        assert not counter.should_compute_metrics(metric_interval)

        counter.current_step = 800
        assert counter.should_compute_metrics(metric_interval)


class TestVisualizationInterval:
    """Test visualization interval checking."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = -1
        mock_config.max_steps_per_epoch = -1
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_visualization_interval(self, counter):
        """Test visualization at interval boundaries."""
        viz_interval = 2000

        counter.current_step = 2000
        assert counter.should_visualize(viz_interval)

        counter.current_step = 1999
        assert not counter.should_visualize(viz_interval)

        counter.current_step = 4000
        assert counter.should_visualize(viz_interval)


class TestHardStopConditions:
    """Test hard stop conditions."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = 1000
        mock_config.max_steps_per_epoch = 100
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_should_stop_global_unlimited(self, config):
        """Test global stop when unlimited."""
        config.max_iterations = -1
        counter = IterationCounterService(config)

        counter.current_step = 1000000
        assert not counter.should_stop_global()

    def test_should_stop_global_at_limit(self, counter):
        """Test global stop at max_iterations."""
        counter.current_step = 999
        assert not counter.should_stop_global()

        counter.current_step = 1000
        assert counter.should_stop_global()

        counter.current_step = 1001
        assert counter.should_stop_global()

    def test_should_stop_epoch_unlimited(self, config):
        """Test epoch stop when unlimited."""
        config.max_steps_per_epoch = -1
        counter = IterationCounterService(config)

        counter.steps_in_current_epoch = 10000
        assert not counter.should_stop_epoch()

    def test_should_stop_epoch_at_limit(self, counter):
        """Test epoch stop at max_steps_per_epoch."""
        counter.steps_in_current_epoch = 99
        assert not counter.should_stop_epoch()

        counter.steps_in_current_epoch = 100
        assert counter.should_stop_epoch()

        counter.steps_in_current_epoch = 101
        assert counter.should_stop_epoch()


class TestStateManagement:
    """Test state save/restore functionality."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = 1000
        mock_config.max_steps_per_epoch = 100
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_get_state(self, counter):
        """Test getting counter state."""
        counter.current_step = 500
        counter.current_epoch = 5
        counter.steps_in_current_epoch = 50
        counter.total_steps_this_epoch = 100

        state = counter.get_state()

        assert state["current_step"] == 500
        assert state["current_epoch"] == 5
        assert state["steps_in_current_epoch"] == 50
        assert state["total_steps_this_epoch"] == 100
        assert state["max_iterations"] == 1000
        assert state["max_steps_per_epoch"] == 100

    def test_restore_state(self, counter):
        """Test restoring counter state."""
        state = {
            "current_step": 500,
            "current_epoch": 5,
            "steps_in_current_epoch": 50,
            "total_steps_this_epoch": 100,
        }

        counter.restore_state(state)

        assert counter.current_step == 500
        assert counter.current_epoch == 5
        assert counter.steps_in_current_epoch == 50
        assert counter.total_steps_this_epoch == 100

    def test_state_roundtrip(self, counter):
        """Test save and restore state roundtrip."""
        # Set state
        counter.current_step = 750
        counter.current_epoch = 7
        counter.steps_in_current_epoch = 50
        counter.total_steps_this_epoch = 100

        # Save
        state = counter.get_state()

        # Create new counter and restore
        new_counter = IterationCounterService(counter.config)
        new_counter.restore_state(state)

        # Verify
        assert new_counter.current_step == 750
        assert new_counter.current_epoch == 7
        assert new_counter.steps_in_current_epoch == 50
        assert new_counter.total_steps_this_epoch == 100


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = 1000
        mock_config.max_steps_per_epoch = 100
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_increment_respects_max_iterations(self, counter):
        """Test that increment raises error when exceeding max_iterations."""
        counter.current_step = 999
        counter.increment()  # Should work, now at 1000

        with pytest.raises(RuntimeError):
            counter.increment()  # Should raise, would exceed 1000

    def test_interval_at_exactly_max_iterations(self, counter):
        """Test intervals at exactly max_iterations boundary."""
        # At max_iterations, no operation should trigger
        counter.current_step = 1000
        counter.config.max_iterations = 1000

        assert not counter.should_checkpoint(1000)
        assert not counter.should_log(100)
        assert not counter.should_validate(500)
        assert not counter.should_compute_metrics(500)
        assert not counter.should_visualize(1000)

    def test_interval_just_before_max_iterations(self, counter):
        """Test intervals just before max_iterations."""
        counter.current_step = 999
        counter.config.max_iterations = 1000

        # 999 % 100 = 99, so should_log returns False (interval not met)
        assert not counter.should_log(100)

        # 900 % 100 == 0, so should_log returns True (interval met and within limit)
        counter.current_step = 900
        assert counter.should_log(100)

    def test_zero_as_interval_boundary(self, counter):
        """Test behavior at step 0."""
        counter.current_step = 0

        # All intervals with modulo 0 should be False (edge case)
        # Actually, 0 % any_number = 0, so:
        assert counter.should_checkpoint(1000)  # 0 % 1000 == 0 -> True
        assert counter.should_log(100)  # 0 % 100 == 0 -> True
        assert counter.should_compute_metrics(500)  # 0 % 500 == 0 -> True

    def test_large_step_numbers(self, counter):
        """Test with large step numbers."""
        counter.current_step = 1000000
        counter.config.max_iterations = 2000000

        assert counter.should_checkpoint(100000)  # 1000000 % 100000 == 0
        assert counter.should_log(50000)  # 1000000 % 50000 == 0

    def test_multi_service_interval_alignment(self, counter):
        """Test behavior when multiple service intervals align."""
        counter.current_step = 1000
        counter.config.max_iterations = -1  # Unlimited

        # All should trigger at step 1000
        assert counter.should_log(100)  # 1000 % 100 == 0
        assert counter.should_compute_metrics(200)  # 1000 % 200 == 0
        assert counter.should_validate(500, use_epoch_based=False)  # 1000 % 500 == 0
        assert counter.should_checkpoint(1000)  # 1000 % 1000 == 0


class TestRealisticTrainingScenarios:
    """Test realistic training scenarios."""

    @pytest.fixture
    def config(self):
        mock_config = Mock()
        mock_config.max_iterations = 10000
        mock_config.max_steps_per_epoch = 1000
        return mock_config

    @pytest.fixture
    def counter(self, config):
        return IterationCounterService(config)

    def test_typical_training_loop_simulation(self, counter):
        """Simulate typical training loop with interval checks."""
        # Epoch 0
        counter.start_epoch(1000)
        assert counter.current_epoch == 1  # start_epoch increments first

        for step in range(1000):
            counter.increment()

            # Logging every 100 steps
            if counter.should_log(100):
                assert (step + 1) % 100 == 0  # Check alignment

            # Checkpoint every 500 steps
            if counter.should_checkpoint(500):
                assert (step + 1) % 500 == 0

        # Epoch 1
        counter.start_epoch(1000)
        assert counter.current_epoch == 2  # start_epoch increments again

        for step in range(1000):
            counter.increment()

        # Should now be at step 2000
        assert counter.current_step == 2000
        assert counter.current_epoch == 2

    def test_training_stops_at_max_iterations(self, counter):
        """Test training stops gracefully at max_iterations."""
        counter.config.max_iterations = 500

        for step in range(510):
            if counter.should_stop_global():
                break
            try:
                counter.increment()
            except RuntimeError:
                break

        assert counter.current_step == 500

    def test_mixed_intervals_no_conflicts(self, counter):
        """Test that mixed intervals don't cause issues."""
        counter.config.max_iterations = 10000

        # Configure intervals
        checkpoint_interval = 1000
        log_interval = 100
        metrics_interval = 500
        validation_interval = 2000

        checkpoints_saved = 0
        logs_written = 0
        metrics_computed = 0
        validations_run = 0

        # Simulate 5000 steps
        for _ in range(5000):
            counter.increment()

            if counter.should_checkpoint(checkpoint_interval):
                checkpoints_saved += 1
            if counter.should_log(log_interval):
                logs_written += 1
            if counter.should_compute_metrics(metrics_interval):
                metrics_computed += 1
            if counter.should_validate(validation_interval, use_epoch_based=False):
                validations_run += 1

        # Verify counts are reasonable
        assert checkpoints_saved == 5  # 1000, 2000, 3000, 4000, 5000
        assert logs_written == 50  # Every 100 steps
        assert metrics_computed == 10  # Every 500 steps
        assert validations_run == 2  # 2000, 4000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
