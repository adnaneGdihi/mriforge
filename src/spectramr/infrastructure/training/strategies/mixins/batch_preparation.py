"""Batch Preparation Mixin for Training Strategies.

Provides common batch unpacking and device transfer utilities.
Follows Single Responsibility Principle (SKILL_ARCHITECTURE.md).
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class BatchPreparationMixin:
    """Mixin for batch data preparation and device transfer.

    Encapsulates batch unpacking and tensor-to-device operations,
    allowing strategies to focus on training logic rather than data handling.

    Used by: All training strategies that need batch preparation
    """

    device: torch.device

    def _to_device(self, data: Any) -> Any:
        """Recursively move data to the active device.

        Handles:
        - torch.Tensor: Move to device with non-blocking transfer
        - uint8 tensors: Auto-cast to float (PNG/JPG inputs)
        - dict/list/tuple: Recursively process containers
        - Other types: Return as-is

        Args:
            data: Data to move (tensor, dict, list, tuple, or scalar)

        Returns:
            Data on self.device with appropriate dtype conversions
        """
        if isinstance(data, torch.Tensor):
            # Auto-cast uint8 to float (common for PNG/JPG inputs)
            if data.dtype == torch.uint8:
                data = data.float() / 255.0
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._to_device(v) for v in data)
        return data

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Unpack batch into (input_batch, target_batch).

        Handles multiple input formats:
        - tuple/list: [input, target, ...]
        - TrainingBatch dataclass: .input and .target attributes
        - dict: {"input": ..., "target": ...}
        - dict-like objects with __getitem__
        - TorchIO/Monai nested dicts with 'data' key

        Args:
            batch: Input batch in any supported format

        Returns:
            Tuple of (input_batch, target_batch) or (None, None) if unpacking fails

        Raises:
            Only returns None on missing tensors, never raises
        """
        input_batch, target_batch = None, None

        # Case 1: tuple/list unpacking
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_batch, target_batch = batch[0], batch[1]

        # Case 2: TrainingBatch dataclass (has .input and .target attributes)
        elif hasattr(batch, "input") and hasattr(batch, "target"):
            input_batch = batch.input
            target_batch = batch.target

        # Case 3: Dict-like access - STRICT: canonical keys only
        elif isinstance(batch, dict):
            # STRICT: Require canonical 'input' and 'target' keys
            if "input" in batch:
                input_batch = batch["input"]
            if "target" in batch:
                target_batch = batch["target"]

        # Case 4: Object with __getitem__ (dict-like but not dict)
        elif hasattr(batch, "__getitem__"):
            try:
                # Try numeric unpacking first
                input_batch = batch[0]
                target_batch = batch[1]
            except (KeyError, TypeError, IndexError):
                # Try canonical string keys only
                try:
                    input_batch = batch["input"]
                except (KeyError, TypeError) as _exc:
                    logger.debug("Suppressed exception: %s", _exc)
                try:
                    target_batch = batch["target"]
                except (KeyError, TypeError) as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

        # Handle TorchIO/Monai Image/Subject dicts (nested dicts with 'data' key)
        if (
            isinstance(input_batch, dict)
            and "data" in input_batch
            and isinstance(input_batch["data"], torch.Tensor)
        ):
            input_batch = input_batch["data"]

        if (
            isinstance(target_batch, dict)
            and "data" in target_batch
            and isinstance(target_batch["data"], torch.Tensor)
        ):
            target_batch = target_batch["data"]

        return input_batch, target_batch


__all__ = ["BatchPreparationMixin"]
