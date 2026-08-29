"""Safe Model Stabilization Module

This module provides safe gradient analysis without slice hashability issues.
"""

from typing import Any

import torch
from torch import nn


def safe_gradient_analysis(
    model: nn.Module,
    gradient_dict: dict | None = None,
) -> dict[str, Any]:
    """Safely analyze gradients without slice hashability issues.

    Args:
        model: PyTorch model to analyze
        gradient_dict: Optional existing gradient dictionary

    Returns:
        Dictionary with gradient analysis (string keys only)

    """
    if gradient_dict is None:
        gradient_dict = {}

    analysis = {}

    for name, param in model.named_parameters():
        # Always convert to string to avoid slice issues
        safe_name = str(name)

        if param.grad is not None:
            grad_data = param.grad.data
            analysis[safe_name] = {
                "norm": float(torch.norm(grad_data)),
                "mean": float(grad_data.mean()),
                "std": float(grad_data.std()),
                "max": float(grad_data.max()),
                "min": float(grad_data.min()),
                "shape": list(grad_data.shape),
                "requires_grad": param.requires_grad,
            }
        else:
            analysis[safe_name] = {
                "norm": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
                "shape": list(param.shape) if hasattr(param, "shape") else [],
                "requires_grad": param.requires_grad,
            }

    return analysis


def safe_parameter_modification(
    config: dict[str, Any],
    modifications: dict[str, Any],
) -> dict[str, Any]:
    """Safely modify configuration parameters with string keys only.

    Args:
        config: Base configuration dictionary
        modifications: Modifications to apply

    Returns:
        Modified configuration with safe string keys

    """
    modified_config = config.copy()

    for param, value in modifications.items():
        # Ensure all keys are strings
        safe_param = str(param)

        # Handle special parameter types
        if "_multiplier" in safe_param:
            base_param = safe_param.replace("_multiplier", "")
            if base_param in modified_config:
                modified_config[base_param] = modified_config[base_param] * float(value)
            else:
                modified_config[safe_param] = float(value)
        elif "_divisor" in safe_param:
            base_param = safe_param.replace("_divisor", "")
            if base_param in modified_config:
                modified_config[base_param] = modified_config[base_param] / float(value)
            else:
                modified_config[safe_param] = float(value)
        else:
            modified_config[safe_param] = value

    return modified_config


def safe_gradient_clipping(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """Safely clip gradients without slice issues.

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use

    Returns:
        Total norm of gradients

    """
    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return float(total_norm)


class SafeStabilizationManager:
    """Safe stabilization manager that avoids slice errors"""

    def __init__(self):
        """__init__."""
        self.gradient_history = {}
        self.parameter_history = {}

    def analyze_gradients(self, model: nn.Module, step: int) -> dict[str, Any]:
        """Analyze gradients safely"""
        analysis = safe_gradient_analysis(model)

        # Store with string key
        step_key = str(step)
        self.gradient_history[step_key] = analysis

        return analysis

    def apply_stabilization(
        self,
        model: nn.Module,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply stabilization fixes safely"""
        # Analyze current gradients
        current_analysis = safe_gradient_analysis(model)

        # Create stabilization recommendations
        recommendations = {}

        for param_name, grad_info in current_analysis.items():
            safe_name = str(param_name)

            if grad_info["norm"] > 10.0:  # High gradient norm
                recommendations[f"{safe_name}_clip"] = True
                recommendations[f"{safe_name}_lr_reduce"] = 0.5
            elif grad_info["norm"] < 1e-6:  # Vanishing gradients
                recommendations[f"{safe_name}_lr_increase"] = 2.0

        return recommendations

    def get_stability_metrics(self) -> dict[str, Any]:
        """Get stability metrics with safe keys"""
        if not self.gradient_history:
            return {}

        metrics = {}

        # Analyze gradient trends
        recent_steps = list(self.gradient_history.keys())[-5:]  # Last 5 steps

        for step in recent_steps:
            step_data = self.gradient_history[step]
            step_key = f"step_{step}"

            total_norm = sum(info["norm"] for info in step_data.values())
            metrics[step_key] = {
                "total_gradient_norm": total_norm,
                "num_parameters": len(step_data),
                "avg_gradient_norm": total_norm / len(step_data) if step_data else 0.0,
            }

        return metrics


# Global instance for easy access
stabilization_manager = SafeStabilizationManager()
