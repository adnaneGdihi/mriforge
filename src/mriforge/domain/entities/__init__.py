#!/usr/bin/env python
"""Domain Entities Package"""

from .data import DatasetEntity

# 3D Generation Entities
from .entities_3d import (
    GenerationRequest,
    GenerationResult,
    MRIVolume,
    SliceToVolumeRequest,
    VolumeGenerationResponse,
)

# Cached-cascade WS-X: resolved-semantic experiment IR
from .experiment_context import (
    DataProfile,
    PhysicsProfile,
    ResolvedExperimentContext,
)
from .model import ModelEntity
from .training import (
    DomainEvent,
    ExperimentEntity,
    ModelValidatedEvent,
    TrainingCompletedEvent,
    TrainingSessionEntity,
    TrainingStartedEvent,
    ValidationResultEntity,
)

__all__ = [
    # New SOLID entities
    "ModelEntity",
    "TrainingSessionEntity",
    "ExperimentEntity",
    "DatasetEntity",
    "ValidationResultEntity",
    "DomainEvent",
    "TrainingStartedEvent",
    "TrainingCompletedEvent",
    "ModelValidatedEvent",
    # 3D Generation Entities
    "MRIVolume",
    "GenerationRequest",
    "GenerationResult",
    "SliceToVolumeRequest",
    "VolumeGenerationResponse",
    # Cached-cascade WS-X: resolved experiment IR
    "ResolvedExperimentContext",
    "DataProfile",
    "PhysicsProfile",
]
