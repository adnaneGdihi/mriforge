"""Adapters for prior-method baseline replications.

Each subclass under this package wraps an upstream baseline (vendored
under ``external/baselines/<name>/``) so it plugs into this repo's
Builder + Strategy + Registry chain without duplicating common
infrastructure (mask generation, coil compression, ESPIRiT, FFT
centering, metric computation).

See ``TODO/backlog_baseline_replication_experiment_11.md`` for the
replication contract and ``BaselineAdapter`` for the declarative
attributes every adapter must expose.
"""

from mriforge.models.baselines._base import BaselineAdapter, CoilHandling, FFTNorm

__all__ = ["BaselineAdapter", "CoilHandling", "FFTNorm"]
