"""Downstream-task model adapters (PR-5 / H3).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-5.

Pre-trained classifiers / segmenters / detectors that the
reconstruction pipeline hands off its output to. Each adapter
implements the :class:`IDownstreamModelService` protocol and is
registered into the DI container by :mod:`bootstrap`.
"""

from .classification_adapter import BrainClassificationAdapter
from .pipeline import DownstreamEvaluationPipeline, DownstreamMetricResult
from .segmentation_adapter import BrainSegmentationAdapter

__all__ = [
    "BrainClassificationAdapter",
    "BrainSegmentationAdapter",
    "DownstreamEvaluationPipeline",
    "DownstreamMetricResult",
]
