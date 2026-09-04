"""
[DEPRECATED] Training Strategy Collaborators

This module is deprecated. Please import directly from the modules below:
- Optimizers: spectramr.infrastructure.training.optimizers.standard_optimizer_stepper
- Metrics: spectramr.infrastructure.training.metrics.metrics_reporter

This shim will be removed in v7.0. Update your imports now.
"""

# NOTE: this is a deprecated re-export shim (see the docstring), but it must NOT
# emit a module-level ``DeprecationWarning`` at import time. The project's strict
# ``filterwarnings`` promotes ``DeprecationWarning`` from ``spectramr.*`` to an
# error, so a warn-on-import would break any (current or future) importer or
# star-import of this still-first-class shim (see WS-7 finding TE-004 /
# the migration-warning incident). The deprecation stays documented in the
# docstring; the re-exports below remain fully functional.

# Only re-export non-loss-computer utilities
from spectramr.infrastructure.training.metrics.metrics_reporter import MetricsReporter
from spectramr.infrastructure.training.optimizers.amp_policy import AMPPolicy
from spectramr.infrastructure.training.optimizers.standard_optimizer_stepper import (
    StandardOptimizerStepper,
)

__all__ = [
    "AMPPolicy",
    "MetricsReporter",
    "StandardOptimizerStepper",
]
