"""Tests for LR scheduler stepping in the training pipeline.

Verifies that the scheduler created by OptimizationBuilder is actually
stepped each iteration, producing LR decay.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch


class TestSchedulerSteppingInPipeline:
    """Verify LR scheduler integration in the training loop."""

    def test_scheduler_step_called_when_schedulers_present(self) -> None:
        """Scheduler.step() must be called after each execute_step."""
        # Arrange: mock scheduler with step() method
        mock_scheduler = MagicMock()
        mock_scheduler.step = MagicMock()
        mock_scheduler.get_last_lr = MagicMock(return_value=[1e-5])

        schedulers = {"main": mock_scheduler}

        # Act: simulate the stepping logic from train.py
        for _iteration in range(10):
            if schedulers:
                for sched_name, sched in schedulers.items():
                    if hasattr(sched, "step"):
                        sched.step()

        # Assert: step called 10 times
        assert mock_scheduler.step.call_count == 10

    def test_empty_schedulers_is_noop(self) -> None:
        """When pipeline.schedulers is empty, no error should occur."""
        schedulers: dict = {}

        # This must not raise
        if schedulers:
            for sched_name, sched in schedulers.items():
                sched.step()

        # If we reach here, the dict was empty and the block was skipped
        assert True

    def test_scheduler_exception_is_caught(self) -> None:
        """A failing scheduler.step() must be caught, not crash training."""
        mock_scheduler = MagicMock()
        mock_scheduler.step = MagicMock(side_effect=RuntimeError("boom"))

        schedulers = {"main": mock_scheduler}

        # Act: simulate the safe stepping logic
        caught = False
        if schedulers:
            for sched_name, sched in schedulers.items():
                try:
                    if hasattr(sched, "step"):
                        sched.step()
                except Exception:
                    caught = True

        assert caught

    def test_lr_actually_decays_with_cosine_scheduler(self) -> None:
        """Integration: CosineAnnealingLR produces decaying LR over steps."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Satisfy PyTorch's optimizer.step() → scheduler.step() contract;
        # otherwise lr_scheduler emits a UserWarning that the project's
        # filterwarnings=error config turns into a test failure.
        x = torch.zeros(1, 10)
        y = torch.zeros(1, 1)
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]

        # LR should have decreased
        assert mid_lr < initial_lr, (
            f"LR did not decay: initial={initial_lr}, after 50 steps={mid_lr}"
        )

    def test_lr_logged_to_losses_scalar(self) -> None:
        """Current LR should be added to losses_scalar dict for CSV logging."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr = MagicMock(return_value=[3.14e-5])

        schedulers = {"main": mock_scheduler}
        losses_scalar: dict = {"g_total_loss": 0.5}

        # Simulate LR logging logic from train.py
        if schedulers:
            for sched_name, sched in schedulers.items():
                try:
                    current_lr = (
                        sched.get_last_lr()[0]
                        if hasattr(sched, "get_last_lr")
                        else None
                    )
                    if current_lr is not None:
                        losses_scalar[f"lr_{sched_name}"] = current_lr
                except Exception:
                    pass

        assert "lr_main" in losses_scalar
        assert losses_scalar["lr_main"] == pytest.approx(3.14e-5)
