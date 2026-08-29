"""One-shot multi-parameter ULF mapping (idea 10).

Provides the multi-task ``(T1, T2, PD, segmentation)`` head module
used by ``OneShotMultiParameterStrategy``.
"""

from .multi_parameter_mapping import MultiParameterHead, UncertaintyWeightedMultiTaskLoss

__all__ = ["MultiParameterHead", "UncertaintyWeightedMultiTaskLoss"]
