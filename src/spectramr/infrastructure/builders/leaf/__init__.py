"""Phase 1: Leaf Component Builders

Fluent builders for creating individual training components:
- Model Builders: GeneratorBuilder, DiscriminatorBuilder, EncoderBuilder, DecoderBuilder
- Optimizer Builders: OptimizerBuilder
- Data Builders: DatasetBuilder, DataLoaderBuilder
- Physics Builders: FFTBuilder, MaskBuilder, DataConsistencyBuilder, PhysicsBuilder

Loss builders are NOT here. The single owner is
``infrastructure/training/builders/loss_builder.py::LossBuilder``, reached via
``LossRegistry`` -> ``create_loss``. ``SingleLossBuilder`` (and the ``LossBuilder``
alias this package exported, which collided with that name) had no consumer
outside the removed directors.

``SchedulerBuilder``, ``GradScalerBuilder`` and ``DataPipelineBuilder`` are gone
for the same reason: ``TrainingPipelineDirector`` was their only caller.
"""

from spectramr.infrastructure.builders.leaf.data_builders import (
    DataLoaderBuilder,
    DatasetBuilder,
)
from spectramr.infrastructure.builders.leaf.model_builders import (
    DecoderBuilder,
    DiscriminatorBuilder,
    EncoderBuilder,
    GeneratorBuilder,
)
from spectramr.infrastructure.builders.leaf.optimizer_builders import (
    OptimizerBuilder,
)
from spectramr.infrastructure.builders.leaf.physics_builders import (
    DataConsistencyBuilder,
    FFTBuilder,
    MaskBuilder,
    PhysicsBuilder,
)

__all__ = [
    # Model builders
    "GeneratorBuilder",
    "DiscriminatorBuilder",
    "EncoderBuilder",
    "DecoderBuilder",
    # Optimizer builders
    "OptimizerBuilder",
    # Data builders
    "DatasetBuilder",
    "DataLoaderBuilder",
    # Physics builders
    "FFTBuilder",
    "MaskBuilder",
    "DataConsistencyBuilder",
    "PhysicsBuilder",
]
