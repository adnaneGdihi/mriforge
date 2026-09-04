"""Uncertainty / certificate metrics (PR-CC and siblings).

Importing this sub-package fires the ``@register_metric`` decorators inside it.
The parent ``core.metrics`` walk-discovery skips sub-PACKAGES, so this package
is imported explicitly from ``spectramr.core.metrics.__init__``.
"""

from __future__ import annotations

from spectramr.core.metrics.uncertainty import phys_residual_conformal
from spectramr.core.metrics.uncertainty.metrics import UncertaintyMetrics

__all__ = ["UncertaintyMetrics", "phys_residual_conformal"]
