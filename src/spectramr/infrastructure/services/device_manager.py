"""Device management service implementation."""

from typing import Any

import torch

from spectramr.domain.interfaces.service_interfaces import IDeviceManager
from spectramr.infrastructure.services.device_policy import (
    DevicePolicy,
    DevicePolicyConfig,
    DevicePolicyType,
)


class DeviceManager(IDeviceManager):
    """Concrete implementation of device management."""

    def __init__(self, preferred_device: str | None = None):
        """__init__.

        Args:
            preferred_device (Optional[str]): Description.
        """
        self._device = self._setup_device(preferred_device)
        self._device_policy = DevicePolicy(
            DevicePolicyConfig(
                policy_type=DevicePolicyType.SPECIFIC,
                specific_device=str(self._device),
            ),
        )

    def _setup_device(self, preferred_device: str | None) -> torch.device:
        """Setup the appropriate device.

        Per CLAUDE.md #9 (silent fallbacks are forbidden), an explicitly
        requested device that is unavailable raises rather than silently
        downgrading to CPU. ``None`` is the only ambiguous input — it is
        treated as "auto-detect" and falls back to CPU when CUDA is
        unavailable. See TODO/audit/08_training_services_orchestration.md F14.
        """
        # ``None`` means the caller didn't express a preference (e.g.
        # default-construction in test fixtures). Auto-detect.
        if preferred_device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if preferred_device == "cpu":
            return torch.device("cpu")

        if preferred_device == "cuda" or preferred_device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device requested ({preferred_device!r}) but CUDA is not available. "
                    "Set device=cpu or device=auto explicitly if you want CPU fallback."
                )
            # Validate index for cuda:N
            if preferred_device.startswith("cuda:"):
                try:
                    idx = int(preferred_device.split(":", 1)[1])
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"Malformed CUDA device {preferred_device!r}; expected 'cuda' or 'cuda:N'."
                    ) from exc
                if idx >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"CUDA device index {idx} out of range "
                        f"(available: {torch.cuda.device_count()} GPUs)."
                    )
            return torch.device(preferred_device)

        if preferred_device == "mps":
            mps_available = getattr(torch.backends, "mps", None)
            if mps_available is None or not torch.backends.mps.is_available():
                raise RuntimeError("MPS device requested but MPS backend is not available.")
            return torch.device("mps")

        raise ValueError(
            f"Unsupported device {preferred_device!r}. Expected one of: cpu, cuda, cuda:N, mps."
        )

    def get_device(self) -> torch.device:
        """Get the current device."""
        return self._device

    def move_to_device(self, data: Any) -> Any:
        """Move data to the current device."""
        return self._device_policy.move_to_device(data)

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def set_device(self, device: str) -> None:
        """Set the current device."""
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            self._device = torch.device(device)
            self._device_policy = DevicePolicy(
                DevicePolicyConfig(
                    policy_type=DevicePolicyType.SPECIFIC,
                    specific_device=device,
                ),
            )
        elif device == "cpu":
            self._device = torch.device("cpu")
            self._device_policy = DevicePolicy(
                DevicePolicyConfig(policy_type=DevicePolicyType.CPU),
            )
        else:
            raise ValueError(f"Unsupported device: {device}")

    def list_devices(self) -> list[str]:
        """list available devices."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        return devices
