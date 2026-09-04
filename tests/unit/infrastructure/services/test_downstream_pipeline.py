"""Unit tests for the downstream-task evaluation pipeline (PR-5 / H3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.services.downstream import (
    BrainClassificationAdapter,
    BrainSegmentationAdapter,
    DownstreamEvaluationPipeline,
)


class _IdentitySegmenter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # echo input as a 4-class logit map (C=4)
        return x.expand(-1, 4, -1, -1)


class _IdentityClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3-class linear
        return x.flatten(start_dim=1).mean(dim=1, keepdim=True).repeat(1, 3)


def test_segmentation_adapter_uses_ssot_evaluator() -> None:
    """Adapter delegates metric computation to SegmentationEvaluator (SSOT).

    For multi-class segmentation the SSOT returns ``dice_macro`` and
    ``jaccard_macro``; for binary it returns ``dice`` and ``jaccard``.
    Both shapes prove the adapter routed through the SSOT rather than
    re-rolling the math.
    """
    adapter = BrainSegmentationAdapter(_IdentitySegmenter(), num_classes=4)
    recon = torch.randn(2, 1, 8, 8)
    gt = torch.randint(0, 4, (2, 8, 8))
    m = adapter.compute_metric(recon, gt)
    # Multi-class macro keys from SegmentationEvaluator
    assert "dice_macro" in m
    assert "jaccard_macro" in m
    assert 0.0 <= m["dice_macro"] <= 1.0
    assert m["task"] == "brain_seg"


def test_classification_adapter_returns_balanced_metrics() -> None:
    adapter = BrainClassificationAdapter(_IdentityClassifier(), num_classes=3)
    recon = torch.randn(4, 1, 16, 16)
    gt = torch.tensor([0, 1, 2, 0])
    m = adapter.compute_metric(recon, gt)
    for key in ("sensitivity", "specificity", "balanced_accuracy"):
        assert 0.0 <= m[key] <= 1.0


def test_pipeline_aggregates_across_batches() -> None:
    """Two segmentation adapters share the same gt schema → can be batched together."""
    seg_a = BrainSegmentationAdapter(_IdentitySegmenter(), num_classes=4, task_name="brain_seg_a")
    seg_b = BrainSegmentationAdapter(_IdentitySegmenter(), num_classes=4, task_name="brain_seg_b")
    pipe = DownstreamEvaluationPipeline([seg_a, seg_b])

    recons = [torch.randn(2, 1, 8, 8) for _ in range(3)]
    seg_gts = [torch.randint(0, 4, (2, 8, 8)) for _ in range(3)]
    out = pipe.evaluate(recons, seg_gts)
    assert "brain_seg_a" in out and "brain_seg_b" in out
    assert out["brain_seg_a"].n_samples == 3
    # SegmentationEvaluator (SSOT) returns dice_macro for multi-class.
    assert "dice_macro" in out["brain_seg_a"].metrics


def test_pipeline_rejects_empty_adapters() -> None:
    import pytest
    with pytest.raises(ValueError, match="zero adapters"):
        DownstreamEvaluationPipeline([])


def test_protocol_compatibility() -> None:
    """Adapters satisfy the IDownstreamModelService runtime-checkable protocol."""
    from spectramr.domain.interfaces.i_downstream_model_service import IDownstreamModelService

    seg = BrainSegmentationAdapter(_IdentitySegmenter(), num_classes=4)
    clf = BrainClassificationAdapter(_IdentityClassifier(), num_classes=3)
    assert isinstance(seg, IDownstreamModelService)
    assert isinstance(clf, IDownstreamModelService)
