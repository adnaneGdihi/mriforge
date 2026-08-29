"""Fairness / subgroup-disparity metrics (PR-3, frontier-gap audit 2026-05).

Standard reconstruction metrics (PSNR, SSIM) report a single pooled number
over a heterogeneous cohort, which can hide systematic under-performance on a
protected or under-represented subgroup — a different field strength, a
scanner site, a sex, or an age band. A model that reconstructs the majority
group beautifully and a minority group poorly can post an excellent mean while
being clinically unsafe for the minority. These metrics surface that hidden
inequity by computing the *worst-group gap*.

Definition
----------
Given a batch ``prediction``/``target`` of shape ``[B, C, H, W]`` and a
per-sample subgroup label of length ``B``, we compute a per-sample quality
score ``q_i`` (PSNR in dB, or SSIM in ``[0, 1]``), average within each
subgroup ``g`` to obtain ``mean_g``, and report

    disparity = max_g(mean_g) - min_g(mean_g).

For PSNR and SSIM higher quality is better, so a *larger* disparity means the
best-served group is reconstructed much better than the worst-served group —
hence ``higher_is_better = False`` (lower disparity is fairer). A disparity of
``0`` means every subgroup is reconstructed equally well.

Degenerate cases
----------------
* **No ``subgroup`` kwarg** — the whole batch is treated as a single group, so
  there is no between-group gap and the metric returns ``0.0``. This mirrors a
  fairness audit that simply was not given the protected attribute.
* **Fewer than two distinct groups** (all samples share one label) — likewise
  ``0.0``; a worst-group gap is undefined with one group.

A shape mismatch between ``prediction`` and ``target`` raises ``ValueError``;
a ``subgroup`` whose length does not match the batch raises ``ValueError``.

References
----------
[1] M. Hardt, E. Price, and N. Srebro, "Equality of Opportunity in
    Supervised Learning," *NeurIPS*, 2016.
[2] L. Oala et al., "Machine Learning for Health: Algorithm Auditing,"
    npj Digital Medicine, 2021 (subgroup performance gaps in medical imaging).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from mriforge.core.metrics.registry import get_metric, register_metric


def _normalize_labels(subgroup: Any, batch: int) -> list[Any] | None:
    """Coerce a subgroup spec into a length-``batch`` list of hashable labels.

    Accepts a Python sequence (list/tuple) or a 1-D tensor. Returns ``None``
    when no subgroup information is supplied (the single-group case).
    """
    if subgroup is None:
        return None
    if isinstance(subgroup, torch.Tensor):
        if subgroup.ndim != 1:
            raise ValueError(
                f"subgroup tensor must be 1-D of length B, got shape {tuple(subgroup.shape)}"
            )
        labels = [x.item() for x in subgroup]
    elif isinstance(subgroup, Sequence) and not isinstance(subgroup, (str, bytes)):
        labels = list(subgroup)
    else:
        raise ValueError(
            "subgroup must be a 1-D tensor or a sequence of per-sample labels, "
            f"got {type(subgroup).__name__}"
        )
    if len(labels) != batch:
        raise ValueError(f"subgroup has {len(labels)} labels but batch has {batch} samples")
    return labels


def _check_shapes(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError(f"expected a 4-D batch [B, C, H, W], got {tuple(prediction.shape)}")


def _sample_psnr(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """PSNR in dB for a single image ``pred``/``tgt`` (no batch/channel pooling).

    Implements ``20 * log10(data_range / sqrt(mse))`` with a target-derived
    data range, matching the exemplar formula in the task spec. A perfect
    reconstruction (``mse == 0``) is reported as a large finite ceiling
    (100 dB) so disparity arithmetic stays finite.
    """
    mse = torch.mean((pred - tgt) ** 2).item()
    if mse <= 0.0:
        return 100.0
    data_range = (tgt.max() - tgt.min()).item()
    if data_range <= 0.0:
        data_range = 1.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def _worst_group_gap(scores: list[float], labels: list[Any] | None) -> float:
    """Max-minus-min of the per-group mean score; ``0.0`` for < 2 groups."""
    if labels is None:
        return 0.0
    groups: dict[Any, list[float]] = {}
    for score, label in zip(scores, labels, strict=True):
        groups.setdefault(label, []).append(score)
    if len(groups) < 2:
        return 0.0
    means = [sum(v) / len(v) for v in groups.values()]
    return float(max(means) - min(means))


@register_metric("subgroup_psnr_disparity", aliases=["psnr_disparity", "worst_group_psnr_gap"])
class SubgroupPSNRDisparity:
    """Worst-group PSNR gap across subgroups (lower is fairer).

    Returns ``max_g(mean PSNR_g) - min_g(mean PSNR_g)`` in dB. ``0.0`` when no
    ``subgroup`` label is supplied or all samples share one group.
    """

    @property
    def name(self) -> str:
        return "subgroup_psnr_disparity"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        _check_shapes(prediction, target)
        labels = _normalize_labels(kwargs.get("subgroup"), prediction.shape[0])
        scores = [_sample_psnr(prediction[i], target[i]) for i in range(prediction.shape[0])]
        return _worst_group_gap(scores, labels)


@register_metric("subgroup_ssim_gap", aliases=["ssim_disparity", "worst_group_ssim_gap"])
class SubgroupSSIMGap:
    """Worst-group SSIM gap across subgroups (lower is fairer).

    Returns ``max_g(mean SSIM_g) - min_g(mean SSIM_g)``. Per-sample SSIM is
    delegated to the registered ``ssim`` metric. ``0.0`` when no ``subgroup``
    label is supplied or all samples share one group.
    """

    def __init__(self) -> None:
        # Resolve the canonical SSIM metric through the registry so we reuse
        # the SSOT implementation rather than reimplementing the windowing.
        self._ssim = get_metric("ssim")

    @property
    def name(self) -> str:
        return "subgroup_ssim_gap"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        _check_shapes(prediction, target)
        labels = _normalize_labels(kwargs.get("subgroup"), prediction.shape[0])
        scores = [
            float(self._ssim(prediction[i : i + 1], target[i : i + 1]))
            for i in range(prediction.shape[0])
        ]
        return _worst_group_gap(scores, labels)


__all__ = [
    "SubgroupPSNRDisparity",
    "SubgroupSSIMGap",
]
