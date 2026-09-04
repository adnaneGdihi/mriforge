"""Tests for the SFA gradient estimators' context threading.

The SFA ranker recomputes metric gradients live; NR/physics metrics need their
measurement context there too, or the gradient is NaN and the alignment
collapses to 0. Context must be forwarded as a keyword *only when present*, so a
context-free metric (or a summary-metric placeholder lambda without a ``context``
parameter) is never handed an unexpected kwarg.
"""

from __future__ import annotations

import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.meta_evaluation.gradient import metric_gradient


def test_metric_gradient_forwards_context_keyword():
    seen = {}

    def metric(pred, target, *, context=None):  # keyword-only, like registry metrics
        seen["ctx"] = context
        return float(pred.float().mean())  # non-diff float -> SPSA path

    ctx = MetricContext(coil_maps=torch.ones(1, 4, 8, 8))
    g = metric_gradient(
        metric,
        torch.zeros(1, 8, 8),
        torch.ones(1, 8, 8),
        differentiable_hint=False,
        n_samples=2,
        context=ctx,
    )
    assert seen["ctx"] is ctx
    assert g.shape == (1, 8, 8)


def test_metric_gradient_omits_context_when_none():
    # A metric WITHOUT a context parameter must still work (context not passed) —
    # otherwise summary-metric placeholder lambdas would TypeError.
    def metric(pred, target):
        return float(pred.float().mean())

    g = metric_gradient(
        metric,
        torch.zeros(1, 8, 8),
        torch.ones(1, 8, 8),
        differentiable_hint=False,
        n_samples=2,
        context=None,
    )
    assert g.shape == (1, 8, 8)  # no TypeError from an unexpected kwarg
