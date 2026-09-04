"""Tests for MetaEvaluationPipeline.run asset threading.

Focus: ``run(..., assets_by_content=...)`` forwards the real acquisition assets
down to ``run_simulator`` (which attaches them per content). Kept light — the
ranker/aggregation stack is exercised elsewhere; here we only assert the seam.
"""

from __future__ import annotations

import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.meta_evaluation import pipeline as pipeline_mod
from spectramr.core.metrics.meta_evaluation.pipeline import MetaEvaluationPipeline
from spectramr.core.metrics.meta_evaluation.types import MetricSet


def test_run_forwards_assets_by_content_to_simulator(monkeypatch):
    captured = {}

    def _fake_run_simulator(clean_volumes, config, assets_by_content=None):
        captured["assets_by_content"] = assets_by_content
        raise RuntimeError("short-circuit after capturing the seam")

    monkeypatch.setattr(pipeline_mod, "run_simulator", _fake_run_simulator)

    pipe = MetaEvaluationPipeline.from_defaults()
    metric_set = MetricSet(metrics={"noop": lambda p, t, **k: 0.0})
    assets = {"subj0": MetricContext(coil_maps=torch.ones(1, 4, 8, 8))}

    try:
        pipe.run(
            metric_set, [("subj0", torch.zeros(1, 8, 8))], assets_by_content=assets
        )
    except RuntimeError:
        pass

    assert captured["assets_by_content"] is assets
