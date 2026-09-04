"""How a metric reduces a batch, and how an epoch reduces its batches.

One owner for a single invariant (non-negotiable 17): **a published metric is a
mean over samples, never a mean over batches, and never a non-linear function of
a batch-level reduction.**

Both halves are needed, and neither is sufficient alone:

1. A *range-sensitive, non-linear* metric must reduce **per sample** and then
   average. PSNR is ``20*log10(DR / sqrt(MSE))``; ``log`` is concave, so by
   Jensen ``mean_i(PSNR_i) != PSNR(mean_i(MSE_i))``. Reducing MSE over a whole
   batch first and taking the log once makes the answer a function of how the
   loader happened to group the samples.
2. The epoch value must weight each batch by its **sample count**. Dividing a
   running sum by the *batch* count also weights a short final batch
   (``drop_last=False``) exactly like a full one.

Composed, they were worth **14.3 dB** on bit-identical predictions across
``batch_size`` 1 -> 24 on a heterogeneous set (issue #1347). The effect is driven
by heterogeneity across samples and very nearly vanishes on uniform synthetic
data, which is why every fixture in the suite was blind to it.

The convention is stamped into ``provenance.json`` via :func:`aggregation_provenance`
because it restates numbers the corpus has already recorded: a run from before
this change and a run after it are not comparable, and nothing else in the
artifact would say so.

Not in scope here, and still batch-composition dependent: metrics that are a
*ratio* of batch-level reductions (``NRMSE``, ``NMSE``). They inherit the
sample-weighted epoch mean but keep their own Jensen term.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "PSNR_REDUCTION",
    "SAMPLE_AXIS_MIN_NDIM",
    "VALIDATION_EPOCH_WEIGHTING",
    "aggregation_provenance",
    "per_sample_flat",
    "per_sample_peak",
]

#: How :class:`~spectramr.core.metrics.evaluation_metrics.PSNR` reduces a batch.
PSNR_REDUCTION = "per_sample_mean"

#: How ``_run_validation`` reduces its per-batch metric values into an epoch value.
VALIDATION_EPOCH_WEIGHTING = "sample"

#: A tensor with fewer dims than this has **no** sample axis and is one sample.
#:
#: ``reduction="mean"`` is dimension-agnostic; a per-sample reduction is not, so
#: the sample axis has to be *decided* rather than assumed. An image-pair metric's
#: contract is ``(B, C, H, W)``, so 4-D and above put the batch at dim 0. A direct
#: caller handing in a bare ``(C, H, W)`` or ``(H, W)`` image would otherwise have
#: its channel or row axis silently reinterpreted as a batch of samples -- which is
#: exactly the class of defect this module exists to close, re-introduced one layer
#: down. Below the threshold the whole tensor is one sample, which reproduces the
#: pre-#1347 formula bit-for-bit.
SAMPLE_AXIS_MIN_NDIM = 4


def per_sample_flat(tensor: Tensor) -> Tensor:
    """View ``tensor`` as ``(N, -1)`` with dim 0 the sample axis.

    Args:
        tensor: Any real or complex tensor.

    Returns:
        A ``(N, K)`` view. ``N == tensor.shape[0]`` when the tensor has a sample
        axis (see :data:`SAMPLE_AXIS_MIN_NDIM`), otherwise ``N == 1``.
    """
    if tensor.ndim < SAMPLE_AXIS_MIN_NDIM:
        return tensor.reshape(1, -1)
    return tensor.reshape(tensor.shape[0], -1)


def per_sample_peak(flat: Tensor, *, floor: float, empty_fallback: float) -> Tensor:
    """Per-sample peak magnitude, floored -- the per-image ``data_range`` modes.

    Vectorised deliberately: the scalar spellings this replaces each called
    ``.item()`` on a device tensor, i.e. one GPU sync per range-sensitive metric
    per validation batch (non-negotiable 9).

    Args:
        flat: ``(N, K)`` tensor from :func:`per_sample_flat`. May be complex.
        floor: Lower bound applied to every sample's peak. Use ``0.0`` for none.
        empty_fallback: Peak substituted for a sample whose peak is ``<= floor``
            *and* whose peak is not positive -- an all-zero image, where the
            ratio would otherwise be undefined rather than merely small.

    Returns:
        A ``(N,)`` real tensor of peaks.
    """
    peak = flat.abs().amax(dim=1) if flat.is_complex() else flat.amax(dim=1)
    peak = torch.where(peak > 0, peak, torch.full_like(peak, empty_fallback))
    return peak.clamp(min=floor) if floor > 0.0 else peak


def aggregation_provenance() -> dict[str, str]:
    """The reduction convention, for stamping into ``provenance.json``.

    Read by :func:`spectramr.infrastructure.logging.provenance.collect_run_provenance`.
    A constant record, not a measurement: it names the convention the code in this
    tree implements, so a reader can tell a run made under it from one made before.

    Returns:
        ``{"psnr_reduction": ..., "validation_epoch_weighting": ...}``.
    """
    return {
        "psnr_reduction": PSNR_REDUCTION,
        "validation_epoch_weighting": VALIDATION_EPOCH_WEIGHTING,
    }
