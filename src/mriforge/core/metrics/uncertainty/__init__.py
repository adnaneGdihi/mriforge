"""Uncertainty / certificate metrics (PR-CC and siblings).

Importing this sub-package fires the ``@register_metric`` decorators inside it.
The parent ``core.metrics`` walk-discovery skips sub-PACKAGES, so this package
is imported explicitly from ``mriforge.core.metrics.__init__``.
"""

from __future__ import annotations

from mriforge.core.metrics.uncertainty import phys_residual_conformal
from mriforge.core.metrics.uncertainty.metrics import UncertaintyMetrics

__all__ = ["UncertaintyMetrics", "phys_residual_conformal"]
