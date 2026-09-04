"""Training Strategy Mixins.

Provides composable mixin classes for training strategies following
Single Responsibility Principle (SKILL_ARCHITECTURE.md):
- BatchPreparationMixin: Batch unpacking and device transfer
- ValidationMixin: Config validation and error handling
- OptimizerMixin: Optimizer operations and gradient management
"""

from .batch_preparation import BatchPreparationMixin
from .optimizer import OptimizerMixin
from .validation import ValidationMixin

__all__ = [
    "BatchPreparationMixin",
    "OptimizerMixin",
    "ValidationMixin",
]
