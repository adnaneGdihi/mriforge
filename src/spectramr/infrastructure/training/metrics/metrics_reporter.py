import logging as std_logging

import numpy as np
import torch

from spectramr.infrastructure.training.strategy_interfaces import IMetricsReporter


class MetricsReporter(IMetricsReporter):
    """Metrics reporter for training and validation."""

    def __init__(self):
        """__init__."""
        self.logger = std_logging.getLogger(__name__)

    def report_losses(
        self,
        losses: dict[str, float | torch.Tensor],
        epoch: int,
        step_type: str = "train",
    ) -> dict[str, float]:
        """Report loss metrics, converting tensors to scalars.

        Optimized to batch GPU->CPU transfers.
        """
        scalar_losses = {}
        tensor_keys = []
        tensor_values = []

        # 1. Separate scalars and tensors
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                tensor_keys.append(key)
                tensor_values.append(value)
            else:
                scalar_losses[key] = float(value)

        # 2. Batch convert tensors if any
        if tensor_values:
            # Ensure all are 0-dim or 1-dim scalars
            reshaped_values = []
            for t in tensor_values:
                if t.numel() != 1:
                    # Fallback for non-scalars (should be rare for losses)
                    reshaped_values.append(t.mean().reshape(1))
                else:
                    reshaped_values.append(t.reshape(1))

            # Stack and move to CPU in one go
            try:
                stacked = torch.cat(reshaped_values).detach().cpu()
                # Convert to list
                floats = stacked.tolist()

                # Map back to keys
                for key, val in zip(tensor_keys, floats, strict=False):
                    scalar_losses[key] = val
            except Exception as e:
                # Fallback to slow loop if stacking fails (e.g. dtype mismatch)
                self.logger.warning(f"Batch loss conversion failed, falling back to loop: {e}")
                for key, val in zip(tensor_keys, tensor_values, strict=False):
                    scalar_losses[key] = val.item()

        # Log significant losses
        if step_type == "train" and epoch % 100 == 0:
            self.logger.debug(f"Epoch {epoch} losses: {scalar_losses}")

        return scalar_losses

    def detect_anomalies(
        self,
        losses: dict[str, float | torch.Tensor],
    ) -> tuple[bool, bool]:
        """Detect NaN/Inf values in losses with minimal synchronization."""
        nan_detected = False
        inf_detected = False

        tensor_values = [v for v in losses.values() if isinstance(v, torch.Tensor)]

        # Fast path: check all tensors at once
        if tensor_values:
            try:
                # Flatten and concat all tensors to check in one kernel
                # We detach to avoid graph overhead, though isnan usually doesn't need grad
                flat_tensors = [t.detach().reshape(-1) for t in tensor_values]
                combined = torch.cat(flat_tensors)

                # Check combined (triggers 1 sync)
                if torch.isnan(combined).any():
                    nan_detected = True
                if torch.isinf(combined).any():
                    inf_detected = True
            except Exception:
                # Fallback if cat fails
                for v in tensor_values:
                    if torch.isnan(v).any():
                        nan_detected = True
                    if torch.isinf(v).any():
                        inf_detected = True

        # If detected (or if we have non-tensors), perform detailed check to log WHICH key
        # This is the slow path, but only runs when anomaly exists
        if nan_detected or inf_detected:
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        self.logger.error(f"NaN detected in {key}: {value}")
                    if torch.isinf(value).any():
                        self.logger.error(f"Inf detected in {key}: {value}")
                elif isinstance(value, (int, float)):
                    if np.isnan(value):
                        nan_detected = True  # Ensure set if we missed it in tensor check
                        self.logger.error(f"NaN detected in {key}: {value}")
                    if np.isinf(value):
                        inf_detected = True  # Ensure set if we missed it
                        self.logger.error(f"Inf detected in {key}: {value}")

        return nan_detected, inf_detected

    def compute_validation_metrics(
        self,
        hr_fakes: torch.Tensor,
        hr_reals: torch.Tensor,
        device: str,
    ) -> dict[str, float]:
        """Compute validation metrics."""
        from spectramr.core.metrics.evaluation import EvaluationMetrics

        eval_metrics = EvaluationMetrics(device=device, config=self.config)
        return eval_metrics.compute_all(hr_fakes, hr_reals)
