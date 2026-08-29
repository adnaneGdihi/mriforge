"""Tests for the SFA ranker's per-sample context threading (WS5a).

The SFA ranker recomputes metric gradients live. A no-reference metric must
receive its measurement context there or its gradient is NaN and the alignment
collapses to SFA=0. This asserts the ranker builds and forwards context for
context-needing metrics, and passes ``None`` for context-free ones.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.context import MetricContext
from mriforge.core.metrics.meta_evaluation.rankers import sfa as sfa_mod
from mriforge.core.metrics.meta_evaluation.rankers.sfa import (
    SFAConfig,
    SFARanker,
    _resolve_needs,
)
from mriforge.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
    MetricSet,
)


def test_resolve_needs_reads_registry():
    needs, needs_ctx = _resolve_needs("ndcr")
    assert needs_ctx is True
    assert "coil_maps" in needs
    # A non-registry name (summary placeholder) is safely ((), False).
    assert _resolve_needs("not_a_metric") == ((), False)


def _dataset_with_assets():
    h = w = 12
    c = 4
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    phantom = torch.exp(-(yy**2 + xx**2) / 0.3).clamp(0, 1)[None, None].float()
    from mriforge.infrastructure.physics.asset_preparation import prepare_metric_context

    smaps = torch.ones(1, c, h, w, dtype=torch.complex64) / (c**0.5)
    asset = prepare_metric_context(
        coil_images=phantom.to(torch.complex64) * smaps, smaps=smaps, p99=1.0
    )
    samples = []
    for i in range(4):
        g = torch.Generator().manual_seed(i)
        deg = (phantom + 0.1 * torch.randn(phantom.shape, generator=g)).squeeze(0)
        samples.append(
            DegradationSample(
                clean=phantom.squeeze(0),
                degraded=deg,
                degradation_family="gaussian_noise",
                severity={"theta": 0.5},
                content_id=f"subj_{i % 2}",
                seed=i,
                assets=asset,
            )
        )
    return MetricEvaluationDataset(samples=samples, metric_values={}), asset


def test_sfa_forwards_context_only_for_context_needing_metrics(monkeypatch):
    from mriforge.core.metrics import get_metric

    dataset, asset = _dataset_with_assets()
    captured: dict[int, object] = {}

    def _spy(metric, clean, hat, **kw):
        captured[id(metric)] = kw.get("context")
        return torch.zeros_like(hat.float())

    monkeypatch.setattr(sfa_mod, "metric_gradient", _spy)

    ms = MetricSet(
        metrics={"ndcr": get_metric("ndcr"), "snr": get_metric("snr")},
        higher_is_better={"ndcr": False, "snr": True},
        differentiable={"ndcr": True, "snr": True},
    )
    SFARanker(SFAConfig(max_pairs=4, spsa_samples=2)).rank(ms, dataset)

    # ndcr needs measurement context -> a real MetricContext was built & passed.
    ndcr_ctx = captured[id(ms.metrics["ndcr"])]
    assert isinstance(ndcr_ctx, MetricContext)
    assert ndcr_ctx.coil_maps is asset.coil_maps  # the real prepared maps
    # snr is context-free -> None (never handed an unexpected kwarg).
    assert captured[id(ms.metrics["snr"])] is None
