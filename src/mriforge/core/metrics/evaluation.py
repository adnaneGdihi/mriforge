"""Evaluation Metrics for MRIForge Models

This module provides evaluation metrics computation for model validation
and testing in MRIForge training.

MIGRATION NOTE (Nov 2025):
This module has been refactored to use the ValidationMetricsComputer and
MetricsRegistry (SSOT). All metric computations route through the
centralized registry.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from mriforge.core.metrics.computer import create_validation_metrics_computer

logger = logging.getLogger(__name__)


class EvaluationMetrics(nn.Module):
    """Evaluation metrics computation module for MRIForge models.

    Provides comprehensive evaluation metrics for assessing model performance
    during validation and testing phases.

    NOTE: This is now a thin wrapper around ValidationMetricsComputer.
    """

    def __init__(self, device: str = "cpu", config: Any = None):
        """Initialize evaluation metrics with config-driven flags."""
        super().__init__()
        self.device = device
        self.config = config

        # Use modern factory pattern (config should have metrics as list)
        self.computer = create_validation_metrics_computer(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (not used for evaluation).

        forward method for EvaluationMetrics.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return x

    def compute_all(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute all available evaluation metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of metric names to values
        """
        return self.computer.compute(predictions, targets)

    def to(self, device: str) -> "EvaluationMetrics":
        """Move module to specified device."""
        self.device = device
        self.computer.device = device

        return super().to(device)
