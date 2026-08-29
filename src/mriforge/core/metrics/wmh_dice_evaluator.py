"""White-Matter-Hyperintensity (WMH) Dice downstream evaluator.

Plan §11 P5 deliverable: report Dice between predicted and ground-
truth WMH masks computed by an external segmenter (LST-AI or
SynthSeg-WMH) on the model's reconstructed FLAIR volume.

The evaluator does **not** ship a learned segmenter — it wraps an
adapter pattern so any external pipeline (subprocess CLI, REST
endpoint, embedded torch model) can be plugged in. The default
``LazySegmenterAdapter`` runs a simple intensity-threshold
"bright FLAIR voxels in white matter" baseline that gives a
diagnostic but not a clinical Dice — useful for smoke-testing the
plumbing on synthetic data before a real segmenter is wired in.

Usage::

    evaluator = WMHDiceEvaluator(segmenter=LazySegmenterAdapter())
    score = evaluator(pred_volume, target_volume, mask=brain_mask)

Returns a single Dice score (or per-batch list when called on a
batched tensor).
"""

from __future__ import annotations

import logging
from typing import Protocol

import torch

logger = logging.getLogger(__name__)


class WMHSegmenterAdapter(Protocol):
    """Protocol any downstream WMH segmenter must satisfy.

    The adapter receives a 3D magnitude volume and an optional
    brain-foreground mask, and returns a 3D binary tensor of the
    same spatial shape with WMH voxels = 1, background = 0.
    """

    def segment(
        self, volume: torch.Tensor, *, brain_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class LazySegmenterAdapter:
    """Diagnostic baseline: hyper-intensity threshold inside the brain mask.

    Not a clinical segmenter. Picks the top ``percentile`` of voxels
    inside the brain as "WMH-like". Used only for plumbing tests.
    """

    def __init__(self, percentile: float = 0.97) -> None:
        if not 0.0 < percentile < 1.0:
            raise ValueError(f"percentile must be in (0, 1); got {percentile}")
        self.percentile = percentile

    def segment(
        self, volume: torch.Tensor, *, brain_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if volume.ndim != 3:
            raise ValueError(
                f"LazySegmenterAdapter expects 3D (D, H, W); got {tuple(volume.shape)}"
            )
        flat = volume.reshape(-1).float()
        if brain_mask is not None:
            if brain_mask.shape != volume.shape:
                raise ValueError(
                    "brain_mask shape must match volume; "
                    f"got {tuple(brain_mask.shape)} vs {tuple(volume.shape)}"
                )
            mask_flat = brain_mask.reshape(-1).float()
            sample = flat[mask_flat > 0]
        else:
            sample = flat
        if sample.numel() == 0:
            return torch.zeros_like(volume)
        threshold = torch.quantile(sample, self.percentile)
        # Use a strict ``>`` only when the threshold sits strictly below
        # the image maximum; otherwise fall back to ``>=`` so a "saturated"
        # bright block (where the top-percentile voxel equals the image
        # max) is still segmented. Without this guard, a binary block of
        # 1s makes ``threshold == 1.0`` and ``volume > threshold`` becomes
        # the empty set — leading to two empty segmentations and a
        # degenerate Dice of 1.0 for completely disjoint bright blocks
        # (smooth / smooth).
        sample_max = sample.max()
        if threshold < sample_max:
            seg = (volume > threshold).to(volume.dtype)
        else:
            seg = (volume >= threshold).to(volume.dtype)
        if brain_mask is not None:
            seg = seg * (brain_mask > 0).to(seg.dtype)
        return seg


class WMHDiceEvaluator:
    """Compute WMH Dice between prediction and target reconstructions.

    Args:
        segmenter: any object satisfying :class:`WMHSegmenterAdapter`.
            Defaults to :class:`LazySegmenterAdapter` so the plumbing
            is testable without external deps.
        smooth: numerical stabiliser added to numerator and
            denominator of the Dice formula.
    """

    def __init__(
        self,
        segmenter: WMHSegmenterAdapter | None = None,
        smooth: float = 1.0,
    ) -> None:
        self.segmenter = segmenter or LazySegmenterAdapter()
        self.smooth = float(smooth)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a per-batch tensor of WMH Dice scores."""
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must match shapes; got "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pred.ndim != 5:
            raise ValueError(f"WMHDiceEvaluator expects (B, C, D, H, W); got {pred.ndim}D")
        B, C = pred.shape[:2]
        if C != 1:
            logger.info(
                "[WMHDiceEvaluator] input has %d channels; using channel 0 (magnitude proxy).",
                C,
            )

        scores = []
        for b in range(B):
            vol_pred = pred[b, 0]
            vol_target = target[b, 0]
            brain = self._to_3d(mask[b]) if mask is not None and mask.ndim >= 4 else None
            seg_pred = self.segmenter.segment(vol_pred.detach(), brain_mask=brain)
            seg_target = self.segmenter.segment(vol_target.detach(), brain_mask=brain)
            scores.append(self._dice(seg_pred, seg_target))
        return torch.stack(scores)

    @staticmethod
    def _to_3d(mask_b: torch.Tensor) -> torch.Tensor | None:
        if mask_b.ndim == 4:
            return mask_b[0]
        if mask_b.ndim == 3:
            return mask_b
        return None

    def _dice(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = (a > 0).float()
        b = (b > 0).float()
        intersection = (a * b).sum()
        return (2.0 * intersection + self.smooth) / (a.sum() + b.sum() + self.smooth)


__all__ = ["LazySegmenterAdapter", "WMHDiceEvaluator", "WMHSegmenterAdapter"]
