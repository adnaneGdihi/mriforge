"""Stability Manager
=================

Manages training stability through:
1. Gradient Clipping (Safe)
2. Anomaly Detection (NaNs/Infs)
3. Divergence Monitoring

Usage:
    manager = StabilityManager(clip_grad_norm=1.0)
    metrics = manager.handle_stability(model, step)
"""

import logging

import torch
import torch.nn as nn

from .safe_stabilization import safe_gradient_clipping

logger = logging.getLogger(__name__)


class StabilityManager:
    """Manages model stability during training."""

    def __init__(self, clip_grad_norm: float = 1.0, detect_anomalies: bool = True):
        """Initialize Stability Manager.

        Args:
            clip_grad_norm: Maximum gradient norm. Set to <= 0 to disable.
            detect_anomalies: Whether to enable PyTorch's anomaly detection.
        """
        self.clip_grad_norm = clip_grad_norm
        self.detect_anomalies = detect_anomalies

        if self.detect_anomalies:
            logger.info("Enabling PyTorch Autograd Anomaly Detection")
            torch.autograd.set_detect_anomaly(True)

    def handle_stability(self, model: nn.Module, step: int) -> dict[str, float]:
        """Apply stability measures and return metrics.

        Args:
            model: The model to stabilize (gradients must be computed).
            step: Current training step.

        Returns:
            Dict containing stability metrics (e.g., 'grad_norm').
        """
        metrics = {}

        # Gradient Clipping
        if self.clip_grad_norm > 0:
            # Use safe version to avoid slice hashability issues
            total_norm = safe_gradient_clipping(model, max_norm=self.clip_grad_norm)
            metrics["grad_norm"] = total_norm

            # Log high gradients
            if total_norm > self.clip_grad_norm * 0.9:
                logger.debug(
                    f"High gradient norm allowed: {total_norm:.4f} (Clip: {self.clip_grad_norm})"
                )

        return metrics

    def check_divergence(self, loss: torch.Tensor, step: int) -> bool:
        """Check for training divergence (NaN/Inf).

        Returns:
            True if divergent (NaN or Inf detected).
        """
        if not torch.isfinite(loss):
            logger.critical(f"DIVERGENCE DETECTED at step {step}: Loss is {loss.item()}")
            return True
        return False
