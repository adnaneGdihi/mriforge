"""Brain-segmentation downstream adapter.

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-5 (H3).

Wraps a pre-trained segmenter and exposes
:class:`IDownstreamModelService`. The adapter routes every metric
computation through the metrics SSOT
(:class:`mriforge.core.metrics.segmentation.SegmentationEvaluator`) — it
never reimplements Dice / Jaccard / accuracy / precision / recall.

The adapter:

- Accepts ``[B, C, H, W]`` reconstructions (real-stacked or magnitude).
- Reduces to magnitude if the input is complex.
- Hands segmentation logits + ground-truth labels to the SSOT
  evaluator and returns its dict unchanged (plus the task tag).

Backed by an injected ``nn.Module`` so the test surface can swap in a
small mock segmenter without dragging a real model through the test
suite.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mriforge.core.metrics.segmentation import SegmentationEvaluator


def _to_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Coerce complex / 2-channel input to magnitude."""
    if torch.is_complex(x):
        return x.abs()
    if x.shape[1] == 2:
        return torch.linalg.norm(x, dim=1, keepdim=True)
    return x


class BrainSegmentationAdapter(nn.Module):
    """Frozen brain-segmenter adapter.

    All metric arithmetic delegates to
    :class:`mriforge.core.metrics.segmentation.SegmentationEvaluator` — the
    canonical SSOT. This class never recomputes Dice / IoU; it only
    wires the input shape coercion + the wrapped segmenter forward
    pass.
    """

    def __init__(
        self,
        segmenter: nn.Module,
        *,
        task_name: str = "brain_seg",
        num_classes: int = 4,
        ignore_index: int = -1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter.eval()
        for p in self.segmenter.parameters():
            p.requires_grad = False
        self._task_name = task_name
        self._num_classes = num_classes
        self._device = device or torch.device("cpu")
        # Single SSOT evaluator instance, reused across calls.
        self._evaluator = SegmentationEvaluator(num_classes=num_classes, ignore_index=ignore_index)

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    @torch.no_grad()
    def forward(self, reconstruction: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = _to_magnitude(reconstruction).to(self._device)
        return self.segmenter(x)

    @torch.no_grad()
    def compute_metric(
        self,
        reconstruction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        logits = self.forward(reconstruction)
        gt = ground_truth.to(self._device)
        # SegmentationEvaluator handles argmax + ignore_index + numpy
        # conversion internally; we just hand off (preds, target).
        results = self._evaluator(logits, gt)
        results["task"] = self._task_name
        return results
