"""Pluggable brain segmentation backends for the Dice-risk certificate (B-1.7).

The MICCAI MRIxFields2026 structural metric is SynthSeg-Dice over 14 deep
grey-matter nuclei. This module provides a backend abstraction so the Dice metric
and the Dice-risk certificate work both on the cluster (real SynthSeg, when
installed) and locally (a real label-Dice proxy — NOT a no-op): a deterministic
intensity-quantile segmentation that yields a genuine multi-region label map, so
Dice between two reconstructions is a meaningful structural-agreement measure.

``get_segmenter`` raises on an unknown backend (no silent fallback, pitfall #9).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ISegmenter(Protocol):
    """Segment a magnitude image ``[B, 1, H, W]`` into a label map ``[B, H, W]``."""

    n_classes: int

    def segment(self, image: torch.Tensor) -> torch.Tensor: ...


class LabelDiceBackend:
    """Deterministic intensity-quantile segmentation proxy (real, not a no-op).

    Bins each image into ``n_classes`` regions by per-image intensity quantiles
    (class 0 = background / lowest bin). Two structurally-similar images produce
    similar label maps ⇒ high Dice; a hallucinated/degraded image diverges ⇒ lower
    Dice. Used locally and as the CI/test backend.
    """

    def __init__(self, n_classes: int = 5) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2; got {n_classes}")
        self.n_classes = int(n_classes)

    def segment(self, image: torch.Tensor) -> torch.Tensor:
        if image.is_complex():
            raise ValueError("LabelDiceBackend expects magnitude (real) images.")
        x = image
        if x.ndim == 4:  # [B, C, H, W] -> [B, H, W] (magnitude; take first channel)
            x = x[:, 0]
        b = x.shape[0]
        # Quantilise over FOREGROUND voxels only (exactly-zero background -> class 0),
        # with strictly-increasing edges so every foreground class is reachable. On
        # mostly-background brain MRI a global-quantile scheme ties the lower edges to
        # 0 and collapses classes 1..k to empty (a degenerate proxy that would mask
        # hallucination); foreground-only quantiles avoid that.
        qs = torch.linspace(0.0, 1.0, self.n_classes + 1, device=x.device, dtype=x.dtype)[1:-1]
        labels = torch.zeros_like(x, dtype=torch.long)
        for i in range(b):
            xi = x[i]
            fg = xi > 0
            if not bool(fg.any()):
                continue
            fg_vals = xi[fg]
            if fg_vals.max() > fg_vals.min():
                edges = torch.quantile(fg_vals, qs)
                edges = torch.cummax(edges, dim=0).values  # monotone non-decreasing
                eps = (fg_vals.max() - fg_vals.min()) * 1e-4 + 1e-8
                for k in range(1, edges.numel()):  # force strictly increasing
                    if edges[k] <= edges[k - 1]:
                        edges[k] = edges[k - 1] + eps
                lab_fg = (torch.bucketize(fg_vals, edges) + 1).clamp_max(self.n_classes - 1)
            else:
                lab_fg = torch.ones_like(fg_vals, dtype=torch.long)
            li = torch.zeros_like(xi, dtype=torch.long)
            li[fg] = lab_fg
            labels[i] = li
        return labels


class SynthSegBackend:
    """Real SynthSeg backend slot (cluster). Raises clearly if unavailable.

    SynthSeg is not a pip dependency; this backend wraps an injected, already-loaded
    segmentation callable (e.g. a torch model returning per-voxel class logits).
    When neither an injected model nor the optional ``synthseg`` import is present it
    RAISES (no silent fallback) — callers select ``label_dice`` for environments
    without SynthSeg.
    """

    def __init__(self, n_classes: int = 14, model: object | None = None) -> None:
        self.n_classes = int(n_classes)
        self._model = model
        if model is None:
            raise RuntimeError(
                "SynthSegBackend requires an injected segmentation model (or a "
                "SynthSeg install). None available — use backend='label_dice' on "
                "environments without SynthSeg."
            )

    def segment(self, image: torch.Tensor) -> torch.Tensor:
        logits = self._model(image)  # type: ignore[operator]
        return torch.argmax(logits, dim=1)


_BACKENDS = {"label_dice": LabelDiceBackend, "synthseg": SynthSegBackend}


def get_segmenter(backend: str, **kwargs: object) -> ISegmenter:
    """Resolve a segmenter backend by name; raise on unknown (no silent fallback)."""
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown segmenter backend {backend!r}; available: {sorted(_BACKENDS)}.")
    return _BACKENDS[backend](**kwargs)  # type: ignore[arg-type]


def dice_score(
    seg_a: torch.Tensor,
    seg_b: torch.Tensor,
    n_classes: int,
    labels: tuple[int, ...] | Sequence[int] | None = None,
) -> torch.Tensor:
    """Mean multi-class Dice per image.

    Args:
        seg_a, seg_b: integer label maps ``[B, H, W]``.
        n_classes: number of label classes; used only when ``labels`` is ``None``
            (class 0 is background, excluded → iterates ``range(1, n_classes)``).
        labels: explicit iterable of integer label IDs to score.  When provided,
            iterates over these IDs directly instead of ``range(1, n_classes)``.
            Use this for non-contiguous label sets such as FreeSurfer aseg IDs
            emitted natively by SynthSeg 2.0 (DGM_LABELS_14).  When ``None``,
            the contiguous ``range(1, n_classes)`` semantics are preserved
            exactly (the ``label_dice`` proxy depends on this).

    Returns:
        ``[B]`` per-image mean Dice in ``[0, 1]``.
    """
    b = seg_a.shape[0]
    out = torch.zeros(b, device=seg_a.device, dtype=torch.float32)
    present = torch.zeros(b, device=seg_a.device, dtype=torch.float32)
    # Use explicit label IDs when provided (e.g. FreeSurfer DGM IDs for SynthSeg 2.0);
    # fall back to contiguous range(1, n_classes) for the label_dice proxy.
    class_ids: Sequence[int] = labels if labels is not None else range(1, n_classes)
    for c in class_ids:
        a = (seg_a == c).reshape(b, -1).float()
        bb = (seg_b == c).reshape(b, -1).float()
        inter = (a * bb).sum(dim=1)
        denom = a.sum(dim=1) + bb.sum(dim=1)
        # Average over classes PRESENT in at least one image — do NOT credit a free
        # Dice=1 for absent-in-both, which (with any region collapse) would inflate
        # Dice and MASK hallucination (the certificate's whole purpose is to surface
        # it). A class absent in both contributes nothing to numerator or count.
        out += torch.where(denom > 0, 2.0 * inter / denom, torch.zeros_like(denom))
        present += (denom > 0).float()
    # Degenerate case (no foreground class anywhere, e.g. a blank prediction) scores
    # 0 (max risk), not a free 1.0.
    return out / present.clamp_min(1.0)


__all__ = [
    "ISegmenter",
    "LabelDiceBackend",
    "SynthSegBackend",
    "dice_score",
    "get_segmenter",
]
