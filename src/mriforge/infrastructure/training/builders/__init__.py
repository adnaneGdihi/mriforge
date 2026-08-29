"""Builder Pattern Infrastructure for Training

This module provides the Builder Pattern implementation for creating
training environments in MRIForge. The pattern eliminates 1,200+ lines
of duplication across 15+ training strategies.

Public API:
    - TrainingEnvironment: Immutable container for all training components
    - Builder: Abstract base class for all builders
    - ModelBuilder: Creates neural network models
    - OptimizationBuilder: Creates optimizers, schedulers, and gradient scalers
    - LossBuilder: Creates loss functions
    - PhysicsBuilder: Creates MRI physics operators
    - TrainingEnvironmentDirector: Orchestrates all builders

Example:
    >>> from mriforge.infrastructure.training.builders import TrainingEnvironmentDirector
    >>> from mriforge.config.settings import TrainingSettings
    >>>
    >>> config = TrainingSettings.from_yaml("experiments/training/my_config.yaml")
    >>> director = TrainingEnvironmentDirector(config)
    >>> env = director.build_environment()
    >>>
    >>> # Access components
    >>> generator = env.models["generator"]
    >>> optimizer = env.optimizers["opt_g"]
    >>> l1_loss = env.losses["l1"]
    >>> fft = env.physics["fft"]
"""

from .base import Builder
from .data_builder import DataBuilder
from .director import TrainingEnvironmentDirector
from .environment import TrainingEnvironment
from .infrastructure_builder import InfrastructureBuilder
from .loss_builder import LossBuilder
from .model_builder import ModelBuilder
from .optimization_builder import OptimizationBuilder
from .physics_builder import PhysicsBuilder

# Alias for backward compatibility
DataLoaderBuilder = DataBuilder

__all__ = [
    "Builder",
    "DataBuilder",
    "DataLoaderBuilder",
    "InfrastructureBuilder",
    "LossBuilder",
    "ModelBuilder",
    "OptimizationBuilder",
    "PhysicsBuilder",
    "TrainingEnvironment",
    "TrainingEnvironmentDirector",
]
