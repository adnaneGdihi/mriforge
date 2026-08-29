"""SynthSeg-Dice structural-fidelity metrics (MICCAI MRIxFields2026, B-1.7).

The challenge grades synthesis on SynthSeg-Dice over 14 deep grey-matter nuclei
(:data:`~mriforge.core.metrics.quantitative.challenge_metrics.DGM_LABELS_14`).
``synthseg_dice`` segments the prediction and target (pluggable backend — real
SynthSeg on the cluster, a real label-Dice proxy locally) and returns the mean
multi-class Dice; ``synthseg_dice_risk`` returns the complementary risk
``1 - Dice`` that the Dice-risk RCPS certificate controls. Both are emitted at
validation (so the headline structural metric is genuinely exercised, not a facade).

**14-label parity:** when ``segmenter_backend="synthseg"`` is requested and no
explicit ``n_classes`` is given, :func:`_mean_dice` defaults to
``n_classes=len(DGM_LABELS_14)`` (14) so the Dice is scored over all 14 challenge
nuclei.  The local ``label_dice`` proxy keeps ``n_classes=5``.  An explicit
``n_classes`` kwarg always overrides.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.quantitative.challenge_metrics import DGM_LABELS_14
from mriforge.core.metrics.quantitative.segmentation import (
    LabelDiceBackend,
    dice_score,
    get_segmenter,
)
from mriforge.core.metrics.registry import register_metric

# Default n_classes per backend.  The SynthSeg backend is calibrated for the
# MRIxFields2026 challenge's 14 DGM nuclei (DGM_LABELS_14); the local proxy
# uses 5 intensity-quantile bins (sufficient for structural-agreement tracking).
# An explicit ``n_classes`` kwarg always overrides this table (pitfall #15).
_DEFAULT_N_CLASSES_BY_BACKEND: dict[str, int] = {
    "synthseg": len(DGM_LABELS_14),  # 14 — challenge parity
    "label_dice": 5,  # local proxy default
}


def _resolve_segmenter(kwargs: dict, n_classes: int):
    """Use an injected segmenter (kwargs/context) or the local label-Dice proxy."""
    seg = kwargs.get("segmenter")
    if seg is not None:
        return seg
    backend = kwargs.get("segmenter_backend")
    if backend is not None:
        return get_segmenter(str(backend), n_classes=n_classes)
    return LabelDiceBackend(n_classes=n_classes)


def _mean_dice(prediction: torch.Tensor, target: torch.Tensor, kwargs: dict) -> float:
    # Resolve backend-aware default BEFORE the explicit kwarg check so that an
    # explicit n_classes always wins (pitfall #15 — every knob read and validated).
    backend = kwargs.get("segmenter_backend")
    if backend is not None:
        backend_key = str(backend)
        if backend_key not in _DEFAULT_N_CLASSES_BY_BACKEND:
            raise ValueError(
                f"Unknown segmenter_backend {backend_key!r}; "
                f"known: {sorted(_DEFAULT_N_CLASSES_BY_BACKEND)}."
            )
        default_n_classes = _DEFAULT_N_CLASSES_BY_BACKEND[backend_key]
    else:
        backend_key = None
        default_n_classes = 5
    n_classes = int(kwargs.get("n_classes", default_n_classes))
    segmenter = _resolve_segmenter(kwargs, n_classes)
    seg_p = segmenter.segment(prediction)
    seg_t = segmenter.segment(target)
    # SynthSeg 2.0 emits native FreeSurfer aseg IDs; score the 14 DGM nuclei
    # directly rather than contiguous 1..13 (metric↔claim fix, pitfall #18).
    labels = DGM_LABELS_14 if backend_key == "synthseg" else None
    return float(dice_score(seg_p, seg_t, segmenter.n_classes, labels=labels).mean().item())


@register_metric("synthseg_dice", aliases=["seg_dice"])
class SynthSegDiceMetric:
    """Mean SynthSeg-Dice between prediction and target (higher is better)."""

    def __init__(self, device: object = None, **_: object) -> None:
        # The metrics computer constructs every metric as ``get(name, device=...)``;
        # accept (and ignore) device/extra kwargs so instantiation never fails
        # (an uncaught constructor error is silently stored as NaN at validation).
        self._device = device

    @property
    def name(self) -> str:
        return "synthseg_dice"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs: object) -> float:
        return _mean_dice(prediction, target, kwargs)


@register_metric("synthseg_dice_risk", aliases=["seg_dice_risk"])
class SynthSegDiceRiskMetric:
    """Mean Dice-risk ``1 - Dice`` (lower is better) — the RCPS-controlled risk."""

    def __init__(self, device: object = None, **_: object) -> None:
        self._device = device

    @property
    def name(self) -> str:
        return "synthseg_dice_risk"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs: object) -> float:
        return 1.0 - _mean_dice(prediction, target, kwargs)


__all__ = ["DGM_LABELS_14", "SynthSegDiceMetric", "SynthSegDiceRiskMetric"]
