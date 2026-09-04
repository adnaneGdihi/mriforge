import logging

import torch


class DeviceManager:
    """
    Centralized Device Management.
    Handles CUDA, MPS (Apple Silicon), CPU, and multi-gpu configurations.
    """

    def __init__(self, override_device: str = None):
        """__init__.

        Args:
            override_device (str): Description.
        """
        self.device = self._detect_device(override_device)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized DeviceManager with device: {self.device}")

    def _detect_device(self, override: str = None) -> torch.device:
        """_detect_device.

        Args:
            override (str): Description.
        Returns:
            torch.device: Description.
        """
        if override:
            return torch.device(override)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def get_device(self) -> torch.device:
        """get_device.

        Returns:
            torch.device: Description.
        """
        return self.device

    def to_device(self, data, non_blocking=True):
        """Move tensor or dict/list of tensors to device."""
        if hasattr(data, "to"):
            return data.to(self.device, non_blocking=non_blocking)
        elif isinstance(data, dict):
            return {k: self.to_device(v, non_blocking) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_device(v, non_blocking) for v in data]
        elif isinstance(data, tuple):
            return tuple(self.to_device(v, non_blocking) for v in data)
        return data
