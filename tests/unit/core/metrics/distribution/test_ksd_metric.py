"""Regression tests for ``KernelisedSteinDiscrepancyMetric`` (WS-2 core metrics).

Guards the no-silent-fallback fix (CLAUDE.md pitfall #9): a missing
``score_fn`` kwarg must RAISE ``ValueError`` rather than degrade to the
Gaussian score ``-x``. A supplied ``score_fn`` must still produce a finite
float.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.distribution.ksd_metric import (
    KernelisedSteinDiscrepancyMetric,
)

pytestmark = pytest.mark.unit


def test_ksd_missing_score_fn_raises_value_error() -> None:
    """Omitting ``score_fn`` must raise — no silent Gaussian-score fallback."""
    torch.manual_seed(0)
    samples = torch.randn(16, 3)
    target = torch.zeros_like(samples)  # protocol stub; unused by KSD.

    metric = KernelisedSteinDiscrepancyMetric(n_bootstrap=0)

    with pytest.raises(ValueError, match="score_fn"):
        metric(samples, target)


def test_ksd_with_score_fn_returns_finite_float() -> None:
    """A supplied ``score_fn`` yields a finite scalar KSD² value."""
    torch.manual_seed(0)
    samples = torch.randn(16, 3)
    target = torch.zeros_like(samples)

    metric = KernelisedSteinDiscrepancyMetric(n_bootstrap=0)
    value = metric(samples, target, score_fn=lambda s: -s)

    assert isinstance(value, float)
    assert math.isfinite(value)


def test_ksd_declares_prior_model_need() -> None:
    """KSD must declare the score model (prior_model) it requires.

    KSD needs a score_fn ∇ₓ log p_θ(x) — a trained density/score model, i.e. the
    un-providable ``prior_model``. Declaring it makes sim2rank classify KSD as
    ``not_applicable`` (expected) rather than ``never_computed`` (a defect) on a
    sweep that has no such model. The metric still RAISES without a score_fn
    (no silent fallback, CLAUDE.md #9) — this only fixes the classification.
    """
    from spectramr.core.metrics.registry import MetricsRegistry

    needs = set(MetricsRegistry.needs("kernelised_stein_discrepancy"))
    assert "prior_model" in needs, f"ksd must declare prior_model; got {needs}"
