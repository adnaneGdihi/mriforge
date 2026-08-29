#!/usr/bin/env python
"""Gradient Poison Protection System
Prevents bad gradients from corrupting optimizer state
"""

import logging
import math

import torch
from torch import nn

logger = logging.getLogger(__name__)


class GradientPoisonProtector:
    """Protects against gradient poisoning that can corrupt optimizer state"""

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        poison_detection_threshold: float = 10.0,
        nan_inf_protection: bool = True,
        gradient_smoothing: bool = True,
        smoothing_momentum: float = 0.9,
    ):
        """__init__.

        Args:
            max_grad_norm (float): Description.
            poison_detection_threshold (float): Description.
            nan_inf_protection (bool): Description.
            gradient_smoothing (bool): Description.
            smoothing_momentum (float): Description.
        """
        self.max_grad_norm = max_grad_norm
        self.poison_detection_threshold = poison_detection_threshold
        self.nan_inf_protection = nan_inf_protection
        self.gradient_smoothing = gradient_smoothing
        self.smoothing_momentum = smoothing_momentum

        # Track gradient history for smoothing
        self.gradient_ema = {}
        self.poison_count = 0
        self.total_steps = 0

    def safe_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        loss: torch.Tensor,
    ) -> dict[str, any]:
        """Perform a safe optimizer step with fused gradient protection.

        Optimized: Uses torch.nn.utils.clip_grad_norm_ for fused norm computation
        and NaN/Inf detection in a single kernel launch.
        """
        self.total_steps += 1

        # 1. Compute Global Norm (Fused Kernel)
        # This calculates norms for all params without intermediate syncs
        # error_if_nonfinite=False ensures we get a tensor result (NaN/Inf) instead of crash
        try:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_grad_norm, error_if_nonfinite=False
            )
        except Exception as e:
            # Fallback for older PyTorch versions or specific errors
            logger.debug(f"Gradient check failed: {e}")
            optimizer.zero_grad()
            return {"step_taken": False, "error": str(e)}

        # 2. Single Synchronization Point
        # We check the scalar status ONCE for the entire model
        total_norm_cpu = total_norm.item()

        issues = {
            "poisoned": False,
            "has_nan": False,
            "has_inf": False,
            "exploding": False,
        }

        if not math.isfinite(total_norm_cpu):
            issues["poisoned"] = True
            if math.isnan(total_norm_cpu):
                issues["has_nan"] = True
            if math.isinf(total_norm_cpu):
                issues["has_inf"] = True

            self.poison_count += 1
            # print(f"Gradient poison detected at step {self.total_steps} (Norm: {total_norm_cpu})")

            # Skip step and zero grads
            optimizer.zero_grad()
            return {
                "step_taken": False,
                "poison_detected": True,
                "issues": issues,
                "fixes_applied": {"skipped": 1},
            }

        # Check for exploding gradients (post-clipping norm is capped, but pre-clipping might be huge)
        # Actually clip_grad_norm_ returns the PRE-clipping norm.
        if total_norm_cpu > self.poison_detection_threshold:
            issues["exploding"] = True
            # We already clipped, so we can proceed, but log it.
            # If it's WAY too high, maybe skip?
            # For now, trust the clipper.

        # If finite, step is safe
        try:
            optimizer.step()
            return {
                "step_taken": True,
                "poison_detected": False,
                "issues": issues,
                "fixes_applied": {"clipped": 1 if issues["exploding"] else 0},
            }
        except Exception as e:
            logger.debug(f"Optimizer step failed: {e}")
            optimizer.zero_grad()
            return {
                "step_taken": False,
                "poison_detected": True,
                "error": str(e),
                "issues": issues,
            }

    def get_stats(self) -> dict[str, float]:
        """Get protection statistics"""
        return {
            "poison_rate": self.poison_count / max(1, self.total_steps),
            "total_poisons": self.poison_count,
            "total_steps": self.total_steps,
        }


def create_gradient_protector(**kwargs) -> GradientPoisonProtector:
    """Factory function to create gradient protector"""
    return GradientPoisonProtector(**kwargs)
