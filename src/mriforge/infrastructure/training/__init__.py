"""Training infrastructure public API.

This package exposes the canonical training state, strategy classes, and
schema-driven strategy factory for the v5+/Phase-2 architecture.
"""

from __future__ import annotations

from .state import TrainingState
from .strategies import (
    BaseTrainingStrategy,
    DiffusionTrainingStrategy,
    DomainAdaptationTrainingStrategy,
    GANTrainingStrategy,
    MAEPretrainingStrategy,
    ReconstructionTrainingStrategy,
    VAETrainingStrategy,
    VQVAETrainingStrategy,
)
from .strategy_factory import TrainingStrategyFactory


def get_training_strategy_factory() -> TrainingStrategyFactory:
    """Return the canonical schema-driven training strategy factory."""
    return TrainingStrategyFactory()


__all__ = [
    "BaseTrainingStrategy",
    "DiffusionTrainingStrategy",
    "DomainAdaptationTrainingStrategy",
    "GANTrainingStrategy",
    "MAEPretrainingStrategy",
    "ReconstructionTrainingStrategy",
    "TrainingState",
    "TrainingStrategyFactory",
    "VAETrainingStrategy",
    "VQVAETrainingStrategy",
    "get_training_strategy_factory",
]
