"""Core type definitions for the framework.

This module provides standard type aliases and protocols to enforce
type safety and shape sovereignty across the codebase.
"""

from typing import Any, Protocol, TypeVar

import torch

# --- Tensor Types ---
# Generic Tensor
Tensor = torch.Tensor

# For clear signatures, we define aliases.
# Future hardening: Switch to jaxtyping when dependencies allow.
Image = Tensor  # Represents: [B, C, H, W]
KSpace = Tensor  # Represents: [B, C, H, W] (complex)
Mask = Tensor  # Represents: [B, H, W]
CoilSens = Tensor  # Represents: [B, C, H, W]

# Scalar Types
Scalar = float | int
Device = str | torch.device


# --- Protocols ---
class IMetrics(Protocol):
    """Protocol for metric calculators."""

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """__call__.

        Args:
            pred (Tensor): Description.
            target (Tensor): Description.
        Returns:
            Tensor: Description.
        """
        ...


class IForwardModel(Protocol):
    """Protocol for physics forward models."""

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """forward.

        Args:
            x (Tensor): Description.
        Returns:
            Tensor: Description.
        """
        ...


# --- Type Vars ---
T = TypeVar("T")
