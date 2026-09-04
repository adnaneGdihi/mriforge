"""Tests for meta-evaluation dataclasses (``types.py``).

Focus: the ``DegradationSample.assets`` carrier added so real acquisition assets
(coil maps, k-space, reconstructor) can ride alongside a (clean, degraded) pair.
"""

from __future__ import annotations

import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.meta_evaluation.types import DegradationSample


def _sample(assets=None) -> DegradationSample:
    x = torch.zeros(1, 1, 8, 8)
    return DegradationSample(
        clean=x,
        degraded=x + 0.1,
        degradation_family="gaussian_noise",
        severity={"theta": 0.5},
        content_id="subj0",
        seed=0,
        assets=assets,
    )


def test_assets_defaults_to_none():
    s = DegradationSample(
        clean=torch.zeros(1, 1, 8, 8),
        degraded=torch.zeros(1, 1, 8, 8),
        degradation_family="gaussian_noise",
        severity={"theta": 0.0},
        content_id="subj0",
        seed=0,
    )
    assert s.assets is None


def test_assets_carries_metric_context():
    ctx = MetricContext(coil_maps=torch.ones(1, 4, 8, 8), acceleration=1.0)
    s = _sample(assets=ctx)
    assert s.assets is ctx
    assert s.assets.coil_maps is not None
    # The residual property still works with assets attached.
    assert s.residual_energy >= 0.0
