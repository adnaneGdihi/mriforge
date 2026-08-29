"""Data loading builders - enforce builder pattern for all dataset creation.

This module provides builders for:
- Datasets (FastMRI, ULF, Paired)
- TorchIO Transforms (train, val, test)
- TorchIO Queues (train, val, test)
- Collate Strategies (robust, graph, physics)
- Validation and error handling

Key principle: No fallback datasets, no mock data. Real data or fail immediately.
"""

from mriforge.data.builders.torchio_queue_builder import (
    TorchIOQueueBuilder,
    TorchIOQueueConfig,
)
from mriforge.data.builders.torchio_subject_builder import (
    FastMRISubjectBuilder,
    PreprocessedSubjectBuilder,
    SubjectBuilder,
    SubjectBuilderFactory,
)
from mriforge.data.builders.torchio_transform_builder import (
    TorchIOTransformBuilder,
    TorchIOTransformConfig,
)
from mriforge.data.collation.strategies import (
    CollateStrategy,
    CollateStrategyFactory,
    GraphCollateStrategy,
    PhysicsCollateStrategy,
    RobustCollateStrategy,
)

from .exceptions import (
    DatasetNotFoundError,
    DatasetTypeNotSupportedError,
    DatasetValidationError,
)

__all__ = [
    "CollateStrategy",
    "CollateStrategyFactory",
    "DatasetNotFoundError",
    "DatasetTypeNotSupportedError",
    "DatasetValidationError",
    "FastMRISubjectBuilder",
    "GraphCollateStrategy",
    "PhysicsCollateStrategy",
    "PreprocessedSubjectBuilder",
    "RobustCollateStrategy",
    "SubjectBuilder",
    "SubjectBuilderFactory",
    "TorchIOQueueBuilder",
    "TorchIOQueueConfig",
    "TorchIOTransformBuilder",
    "TorchIOTransformConfig",
]
