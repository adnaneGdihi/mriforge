"""Gradient Safety Wrapper

This module provides safe wrappers for gradient operations that prevent
slice hashability errors and manage memory efficiently.
"""

import threading
from collections import deque
from typing import Any

import torch
from torch import nn


def ensure_string_keys(data: dict[Any, Any]) -> dict[str, Any]:
    """Ensure all dictionary keys are strings"""
    return {str(k): v for k, v in data.items()}


def safe_named_parameters(model: nn.Module) -> list[tuple]:
    """Get named parameters with guaranteed string names"""
    return [(str(name), param) for name, param in model.named_parameters()]


def safe_gradient_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Create a safe gradient dictionary with string keys"""
    grad_dict = {}

    for name, param in model.named_parameters():
        safe_name = str(name)
        if param.grad is not None:
            grad_dict[safe_name] = param.grad.clone()
        else:
            grad_dict[safe_name] = torch.zeros_like(param)

    return grad_dict


def safe_update_dict(
    target_dict: dict[str, Any],
    updates: dict[Any, Any],
) -> dict[str, Any]:
    """Safely update dictionary ensuring string keys"""
    safe_updates = ensure_string_keys(updates)

    result = target_dict.copy()
    result.update(safe_updates)

    return result


class SafeGradientTracker:
    """Safe gradient tracking that avoids slice errors and manages memory.

    **MEMORY-SAFE:** Implements a ring buffer (deque) to maintain bounded history.
    Prevents unbounded memory growth in long-running training sessions.

    **THREAD-SAFE [FORENSIC FIX]:** Uses RLock to protect history buffers from
    concurrent read/write race conditions.
    """

    def __init__(self, max_history: int = 1000):
        """Initialize tracker with bounded history.

        Args:
            max_history: Maximum number of steps to retain in history.
                        Older steps are evicted automatically.
        """
        self.max_history = max_history
        self._history: deque[str] = deque(maxlen=max_history)  # Ring buffer of step keys
        self.gradient_norms: dict[str, dict[str, float]] = {}
        self.parameter_shapes: dict[str, dict[str, list[int]]] = {}
        self._lock = threading.RLock()  # [FORENSIC FIX] Thread safety

    def track_gradients(self, model: nn.Module, step: int) -> None:
        """Track gradients safely with automatic history eviction.

        OLD ENTRIES: When history exceeds max_history, the oldest step
        is automatically evicted from gradient_norms and parameter_shapes.
        """
        step_key = str(step)

        with self._lock:  # [FORENSIC FIX] Lock during write
            # Detect if we're about to evict the oldest entry
            if len(self._history) == self.max_history:
                # Evict the oldest step (about to be pushed out by deque)
                oldest_step = self._history[0]
                self.gradient_norms.pop(oldest_step, None)
                self.parameter_shapes.pop(oldest_step, None)

            # Record this step in the history ring buffer
            self._history.append(step_key)

            # Track gradients for this step
            self.gradient_norms[step_key] = {}
            self.parameter_shapes[step_key] = {}

            for name, param in safe_named_parameters(model):
                if param.grad is not None:
                    grad_norm = float(torch.norm(param.grad))
                    self.gradient_norms[step_key][name] = grad_norm
                    self.parameter_shapes[step_key][name] = list(param.shape)

    def get_gradient_stats(self, steps: int = 10) -> dict[str, Any]:
        """Get gradient statistics for recent steps.

        Only returns statistics for steps that are still in history.
        """
        with self._lock:  # [FORENSIC FIX] Lock during read
            # Copy to list for safe iteration
            recent_steps = [s for s in list(self._history)[-steps:] if s in self.gradient_norms]

            stats: dict[str, Any] = {}
            for step in recent_steps:
                # Double check existence inside lock
                if step not in self.gradient_norms:
                    continue
                step_norms = self.gradient_norms[step]
                stats[step] = {
                    "mean_norm": (
                        sum(step_norms.values()) / len(step_norms) if step_norms else 0.0
                    ),
                    "max_norm": max(step_norms.values()) if step_norms else 0.0,
                    "min_norm": min(step_norms.values()) if step_norms else 0.0,
                    "num_params": len(step_norms),
                }

            return stats

    def get_history_size(self) -> int:
        """Return current number of tracked steps (always <= max_history)."""
        with self._lock:
            return len(self._history)


# Global tracker instance
gradient_tracker = SafeGradientTracker(max_history=1000)
