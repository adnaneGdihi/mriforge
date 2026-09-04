"""Device Policy Helper for Centralized Device Management

This module provides a centralized device policy helper that standardizes
device assignment across the spectraMR project. It replaces scattered .to(device)
calls with a consistent, testable interface.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

from spectramr.core.compute_device import resolve_torch_device

logger = logging.getLogger(__name__)


class DevicePolicyType(Enum):
    """Device policy types for different use cases."""

    AUTO = "auto"  # Auto-select best available device
    CUDA = "cuda"  # Force CUDA if available
    CPU = "cpu"  # Force CPU
    SPECIFIC = "specific"  # Specific device (e.g., cuda:1)


@dataclass(frozen=True)
class DevicePolicyConfig:
    """Configuration for device policy.

    ``fallback_to_cpu`` defaults to **False**: silently degrading a GPU run to
    CPU is the failure mode this project refuses (CLAUDE.md pitfall #9, and the
    accelerated-run contract in :mod:`spectramr.core.compute_device`). A caller
    that genuinely wants CPU must ask for it — ``DevicePolicyType.CPU`` or
    ``fallback_to_cpu=True`` explicitly.
    """

    policy_type: DevicePolicyType
    fallback_to_cpu: bool = False
    specific_device: str | None = None
    memory_threshold_mb: int | None = None


class DevicePolicy:
    """Centralized device policy helper for consistent device management.

    This class provides a single point of control for device assignment,
    replacing scattered .to(device) calls throughout the codebase.

    ### Device Resolution Ladder
    ```mermaid
    graph TD
        A[Resolution Start] --> B{Policy Type?}
        B -- CPU --> C[Return CPU]
        B -- CUDA --> D{CUDA Avail?}
        D -- No --> E{Fallback CPU?}
        E -- Yes --> C
        E -- No --> F[Raise Error]
        D -- Yes --> G[Return CUDA]
        B -- SPECIFIC --> H{Device Avail?}
        H -- Yes --> I[Return Device]
        H -- No --> J{Fallback CPU?}
        J -- Yes --> C
        J -- No --> F
        B -- AUTO --> K{CUDA Avail?}
        K -- Yes --> L{Memory OK?}
        L -- Yes --> G
        L -- No --> M{Fallback CPU?}
        M -- Yes --> C
        M -- No --> F
        K -- No --> C
    ```
    """

    def __init__(self, config: DevicePolicyConfig | None = None):
        """Initialize device policy.

        Args:
            config: Device policy configuration. If None, uses AUTO policy.

        """
        if config is None:
            config = DevicePolicyConfig(policy_type=DevicePolicyType.AUTO)

        self.config = config
        self._device_cache: dict[str, torch.device] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the device policy configuration."""
        if self.config.policy_type == DevicePolicyType.SPECIFIC:
            if not self.config.specific_device:
                raise ValueError("specific_device must be provided for SPECIFIC policy")

        if self.config.memory_threshold_mb is not None and not torch.cuda.is_available():
            raise ValueError("Memory threshold only supported for CUDA devices")

    def get_device(self) -> torch.device:
        """Get the target device based on policy.

        Returns:
            torch.device: The target device

        """
        cache_key = f"{self.config.policy_type.value}_{self.config.specific_device}"

        if cache_key in self._device_cache:
            return self._device_cache[cache_key]

        device = self._resolve_device()
        self._device_cache[cache_key] = device
        return device

    def _resolve_device(self) -> torch.device:
        """Resolve the actual device based on policy."""
        if self.config.policy_type == DevicePolicyType.CPU:
            return torch.device("cpu")

        if self.config.policy_type == DevicePolicyType.CUDA:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if self.config.fallback_to_cpu:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
            raise RuntimeError("CUDA requested but not available")

        if self.config.policy_type == DevicePolicyType.SPECIFIC:
            device_str = self.config.specific_device
            if device_str is None:
                raise ValueError("specific_device must be provided for SPECIFIC policy")

            if device_str.startswith("cuda"):
                if not torch.cuda.is_available():
                    if self.config.fallback_to_cpu:
                        logger.warning(
                            f"CUDA device {device_str} requested but not available, "
                            "falling back to CPU",
                        )
                        return torch.device("cpu")
                    msg = f"CUDA device {device_str} requested but CUDA not available"
                    raise RuntimeError(msg)
                return torch.device(device_str)
            return torch.device(device_str)

        if self.config.policy_type == DevicePolicyType.AUTO:
            if torch.cuda.is_available():
                # Check memory threshold if specified
                if self.config.memory_threshold_mb is not None:
                    if self._check_memory_threshold():
                        return torch.device("cuda")
                    if self.config.fallback_to_cpu:
                        logger.warning(
                            f"GPU memory threshold not met ({self.config.memory_threshold_mb}MB), "
                            "falling back to CPU",
                        )
                        return torch.device("cpu")
                    msg = "Memory threshold not met and fallback disabled"
                    raise RuntimeError(msg)
                return torch.device("cuda")
            # No CUDA under an AUTO policy. Defer to the accelerated-run
            # contract rather than assuming CPU is acceptable: it raises unless
            # the caller opted in (fallback_to_cpu / FORCE_CPU).
            if self.config.fallback_to_cpu:
                return torch.device("cpu")
            decision = resolve_torch_device("auto", pipeline="train", source="DevicePolicy(AUTO)")
            return torch.device(decision.device)

        raise ValueError(f"Unknown policy type: {self.config.policy_type}")

    def _check_memory_threshold(self) -> bool:
        """Check if available GPU memory meets the threshold."""
        if not torch.cuda.is_available():
            return False

        try:
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory - torch.cuda.memory_allocated(0)
            available_mb = available_memory / (1024 * 1024)
            threshold = self.config.memory_threshold_mb
            return available_mb >= threshold if threshold is not None else True
        except (RuntimeError, AttributeError, ValueError):
            return False

    def move_to_device(
        self,
        data: Any,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Any:
        """Move data to the target device.

        Args:
            data: Data to move (tensor, model, list, dict, etc.)
            dtype: Optional dtype conversion
            non_blocking: Whether to use non-blocking transfer

        Returns:
            Data moved to target device

        """
        device = self.get_device()
        return self._move_data_to_device(data, device, dtype, non_blocking)

    def _move_data_to_device(
        self,
        data: Any,
        device: torch.device,
        dtype: torch.dtype | None,
        non_blocking: bool,
    ) -> Any:
        """Internal method to move data to device."""
        if data is None:
            return data

        # Handle tensors
        if isinstance(data, torch.Tensor):
            if dtype is not None:
                return data.to(device=device, dtype=dtype, non_blocking=non_blocking)
            return data.to(device=device, non_blocking=non_blocking)

        # Handle models
        if isinstance(data, nn.Module):
            return self._move_model_to_device(data, device, dtype, non_blocking)

        # Handle collections
        if isinstance(data, (list, tuple)):
            moved_items = [
                self._move_data_to_device(item, device, dtype, non_blocking) for item in data
            ]
            return type(data)(moved_items)

        if isinstance(data, dict):
            return {
                key: self._move_data_to_device(value, device, dtype, non_blocking)
                for key, value in data.items()
            }

        # Return as-is if not movable
        return data

    def _move_model_to_device(
        self,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype | None,
        non_blocking: bool,
    ) -> nn.Module:
        """Move model to device with proper parameter/buffer handling."""
        # Move model
        model = model.to(device=device, non_blocking=non_blocking)

        # Convert dtype if specified
        if dtype is not None:
            model = model.to(dtype=dtype)

        # Ensure all parameters are on target device
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device=device, non_blocking=non_blocking)

        # Ensure all buffers are on target device
        for buffer in model.buffers():
            if buffer.device != device:
                buffer.data = buffer.data.to(device=device, non_blocking=non_blocking)

        return model

    def is_device_available(self, device: torch.device | None = None) -> bool:
        """Check if a device is available.

        Args:
            device: Device to check. If None, checks the policy's device.

        Returns:
            bool: True if device is available

        """
        if device is None:
            try:
                device = self.get_device()
            except RuntimeError:
                return False

        if device.type == "cpu":
            return True
        if device.type == "cuda":
            return torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count()

        return False

    def get_device_info(self) -> dict[str, Any]:
        """Get information about the current device.

        Returns:
            dict with device information

        """
        device = self.get_device()
        info = {
            "device": str(device),
            "type": device.type,
            "policy": self.config.policy_type.value,
        }

        if device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(device.index or 0)
                info.update(
                    {
                        "name": str(props.name),
                        "total_memory_mb": str(int(props.total_memory / (1024 * 1024))),
                        "major": str(props.major),
                        "minor": str(props.minor),
                    },
                )
            except Exception as e:
                info["error"] = str(e)

        return info

    def clear_cache(self) -> None:
        """Clear device cache."""
        self._device_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Convenience functions for common use cases
def create_auto_device_policy() -> DevicePolicy:
    """Create a device policy that auto-selects the best available device."""
    return DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.AUTO))


def create_cpu_device_policy() -> DevicePolicy:
    """Create a device policy that forces CPU usage."""
    return DevicePolicy(DevicePolicyConfig(policy_type=DevicePolicyType.CPU))


def create_cuda_device_policy(fallback_to_cpu: bool = False) -> DevicePolicy:
    """Create a device policy that prefers CUDA with optional CPU fallback."""
    return DevicePolicy(
        DevicePolicyConfig(
            policy_type=DevicePolicyType.CUDA,
            fallback_to_cpu=fallback_to_cpu,
        ),
    )


def create_specific_device_policy(device: str) -> DevicePolicy:
    """Create a device policy for a specific device."""
    return DevicePolicy(
        DevicePolicyConfig(
            policy_type=DevicePolicyType.SPECIFIC,
            specific_device=device,
        ),
    )


# Global default policy instance
_default_device_policy: DevicePolicy | None = None


def get_default_device_policy() -> DevicePolicy:
    """Get the default device policy instance."""
    global _default_device_policy
    if _default_device_policy is None:
        _default_device_policy = create_auto_device_policy()
    return _default_device_policy


def set_default_device_policy(policy: DevicePolicy) -> None:
    """Set the default device policy instance."""
    global _default_device_policy
    _default_device_policy = policy


def move_to_device_auto(data: Any, **kwargs: Any) -> Any:
    """Move data to device using the default auto policy.

    Args:
        data: Data to move
        **kwargs: Additional arguments for move_to_device

    Returns:
        Data moved to device

    """
    return get_default_device_policy().move_to_device(data, **kwargs)
