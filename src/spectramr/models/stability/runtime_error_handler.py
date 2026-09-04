#!/usr/bin/env python
"""Comprehensive Error Handler

Provides robust error handling and recovery for all runtime issues.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RuntimeErrorHandler:
    """Comprehensive runtime error handler"""

    @staticmethod
    def safe_lower(obj: Any) -> str:
        """Safely call .lower() on any object"""
        if obj is None:
            return ""
        if hasattr(obj, "lower"):
            return str(obj.lower())
        return str(obj).lower()

    @staticmethod
    def ensure_device(tensor: Any, device: Any) -> Any:
        """Safely ensure tensor is on correct device"""
        if tensor is None:
            return tensor
        if not hasattr(tensor, "device"):
            return tensor
        if tensor.device != device:
            try:
                return tensor.to(device)
            except Exception as e:
                logger.debug(f"Warning: Could not move tensor to {device}: {e}")
                return tensor
        return tensor

    @staticmethod
    def safe_slice_key(slice_obj: Any) -> str:
        """Convert slice to hashable key"""
        if isinstance(slice_obj, slice):
            return f"slice_{slice_obj.start}_{slice_obj.stop}_{slice_obj.step}"
        return str(slice_obj)

    @staticmethod
    def safe_model_call(model: Any, *args: Any, **kwargs: Any) -> Any:
        """Safely call model with error recovery"""
        try:
            return model(*args, **kwargs)
        except RuntimeError as e:
            if "device" in str(e).lower():
                logger.debug(f"Device error detected, attempting recovery: {e}")
                # Try to move all tensors to same device as model
                if hasattr(model, "parameters"):
                    model_device = next(model.parameters()).device
                    args = tuple(
                        (
                            RuntimeErrorHandler.ensure_device(arg, model_device)
                            if hasattr(arg, "device")
                            else arg
                        )
                        for arg in args
                    )
                    kwargs = {
                        k: (
                            RuntimeErrorHandler.ensure_device(v, model_device)
                            if hasattr(v, "device")
                            else v
                        )
                        for k, v in kwargs.items()
                    }
                    return model(*args, **kwargs)
            raise

    @staticmethod
    def handle_gradient_error(model: Any, error: Any) -> None:
        """Handle gradient-related errors"""
        logger.debug(f"Gradient error: {error}")
        if hasattr(model, "zero_grad"):
            try:
                model.zero_grad()
                logger.debug("Cleared gradients for recovery")
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)


# Global error handler instance
error_handler = RuntimeErrorHandler()
