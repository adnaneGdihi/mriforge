"""Type definitions for MRI training and inference.

Provides type hints for:
- Batch structures (input/target with masks, sensitivity maps)
- Model outputs (losses, predictions, uncertainties)
- Optimization state (optimizers, schedulers)
- Configuration objects

This module enables strict type checking and better IDE support
across the MRI reconstruction pipeline.
"""

from typing import Protocol, TypedDict

import torch
from torch.optim import Optimizer

from spectramr.config.schemas.enums import OptimizerType as _OptimizerType

# ============================================================================
# BATCH STRUCTURES (TypedDict)
# ============================================================================


class MRIBatchDict(TypedDict, total=False):
    """MRI training batch with k-space and spatial domain data.

    All tensors use the following conventions:
    - Complex data: shape [B, C, H, W, 2] (real/imaginary channels)
    - Multi-coil: shape [B, Coils, H, W] or [B, Coils, H, W, 2]
    - k-space: shape [B, Coils, H_k, W_k] (FFT domain)
    - Image space: shape [B, C, H, W] (spatial domain)
    """

    input: torch.Tensor  # Undersampled/degraded image or k-space
    target: torch.Tensor  # Ground truth (fully sampled or high-quality)
    mask: torch.Tensor | None  # k-space sampling mask [B, 1, H, W]
    sensitivity_maps: torch.Tensor | None  # SENSE coil maps [B, Coils, H, W]
    kspace: torch.Tensor | None  # Raw k-space measurements [B, Coils, H_k, W_k]
    filename: str | None  # Source file for provenance
    slice_idx: int | None  # Slice number in 3D volume


class ModelOutputDict(TypedDict, total=False):
    """Model predictions with losses and metrics."""

    logits: torch.Tensor  # Output predictions
    loss: torch.Tensor  # Scalar loss value
    losses: dict[str, torch.Tensor]  # Loss components {name: value}
    metrics: dict[str, float]  # Aggregated metrics
    uncertainty: torch.Tensor | None  # Aleatoric uncertainty estimate


# ============================================================================
# OPTIMIZER & SCHEDULER TYPES
# ============================================================================

#: Re-exported from the config layer, which owns the vocabulary
#: (``domain/`` -> ``config/`` is rightward, so this import is legal).
#:
#: This name used to be an independent ``Literal["adam", "sgd", "adamw"]`` — a
#: third disagreeing optimizer vocabulary alongside the 6-member ``OptimizerType``
#: enum and the 7-name ``OptimizerRegistry``, none of which was the SSOT. It was
#: exported but referenced nowhere, so collapsing it onto the enum regresses
#: nothing and leaves exactly one list.
OptimizerType = _OptimizerType
OptimizerOrTuple = Optimizer | tuple[Optimizer, ...] | dict[str, Optimizer]


class OptimizerStateDict(TypedDict):
    """Checkpoint state for optimizer persistence."""

    optimizer_state: dict
    scheduler_state: dict | None
    epoch: int
    step: int


# ============================================================================
# CONFIG PROTOCOLS (for runtime type checking without importing TrainingSettings)
# ============================================================================


class TrainingConfigProtocol(Protocol):
    """Protocol defining minimal training config interface."""

    training_mode: str
    epochs: int
    batch_size: int
    learning_rate: float
    device: str | torch.device


# ============================================================================
# TYPE ALIASES
# ============================================================================

TensorOrFloat = torch.Tensor | float
TensorDict = dict[str, torch.Tensor]
MetricsDict = dict[str, float]
LossesDict = dict[str, torch.Tensor]

__all__ = [
    "LossesDict",
    "MRIBatchDict",
    "MetricsDict",
    "ModelOutputDict",
    "OptimizerOrTuple",
    "OptimizerStateDict",
    "OptimizerType",
    "TensorDict",
    "TensorOrFloat",
    "TrainingConfigProtocol",
]
