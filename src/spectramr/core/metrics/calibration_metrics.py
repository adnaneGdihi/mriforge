r"""Calibration metrics beyond ECE (PR-8, frontier-gap audit 2026-05).

The existing :class:`ExpectedCalibrationError`
(``report_step_metrics.py`` lines 499-526) is **binning-biased**: with
equal-width bins, regions of high confidence density share a coarse bin
and the reliability gap is averaged away. This module adds three
complementary, less-biased calibration diagnostics:

* ``brier_score``   — proper scoring rule, no binning at all (#PR-8a)
* ``adaptive_ece``  — ECE with equal-**mass** (quantile) bins (#PR-8b)
* ``classwise_ece`` — mean per-class ECE for multi-class calibration (#PR-8c)

All three follow the existing ECE prediction/target convention exactly:
``preds`` are predicted probabilities / confidences (clipped to
``[0, 1]``), ``target`` are the binary outcome / one-hot / label
indicators. All three are ``higher_is_better = False``.

This is a NEW top-level module under ``core/metrics/`` — it self-registers
via ``@register_metric`` and is auto-discovered by the package
``pkgutil.walk_packages`` scan in ``__init__``; nothing else imports it.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import torch
from torch import Tensor

from spectramr.core.metrics.evaluation_metrics import BaseMetric
from spectramr.core.metrics.registry import register_metric

Image = torch.Tensor


def _to_numpy(t: Tensor) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def _equal_width_ece(confs: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """Equal-width ECE, byte-for-byte semantics of the existing ECE metric.

    Mirrors ``ExpectedCalibrationError.compute_metric`` (report_step_metrics.py
    lines 517-526) so ``classwise_ece`` aggregates the *same* per-class
    quantity the headline metric reports per-class.
    """
    if confs.size == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(bins):
        in_bin = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
        if not in_bin.any():
            continue
        avg_conf = confs[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        ece += (in_bin.sum() / confs.size) * abs(avg_acc - avg_conf)
    return float(ece)


# ── PR-8a Brier score ────────────────────────────────────────────────────────


@register_metric("brier_score", aliases=["brier", "brier_loss"])
class BrierScore(BaseMetric):
    r"""Brier score — mean squared error between predicted probabilities and
    the (binary / one-hot) outcomes.

    .. math::

        \mathrm{BS} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2

    A strictly proper scoring rule with **no binning**, so unlike ECE it is
    free of the equal-width binning bias. Lower is better; ``0`` only for a
    perfectly confident and correct forecast.

    ``preds`` are predicted probabilities/confidences (clipped to
    ``[0, 1]``); ``target`` are the matching binary/one-hot outcomes — the
    same convention as :class:`ExpectedCalibrationError`.
    """

    higher_is_better = False

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        p = _to_numpy(preds).ravel().astype(np.float64).clip(0.0, 1.0)
        y = _to_numpy(target).ravel().astype(np.float64)
        if p.size != y.size or p.size == 0:
            return torch.tensor(float("nan"), device=self.device)
        return torch.tensor(float(np.mean((p - y) ** 2)), device=self.device)


# ── PR-8b Adaptive ECE (equal-mass / quantile bins) ──────────────────────────


@register_metric("adaptive_ece", aliases=["aece", "adaptive_calibration_error"])
class AdaptiveECE(BaseMetric):
    r"""Adaptive ECE — ECE with equal-**mass** bins instead of equal-width.

    Bin edges are the empirical quantiles of the confidence distribution, so
    every bin holds (approximately) the same number of samples. This removes
    the equal-width binning bias of the standard ECE, which lumps a dense
    cluster of confidences into one coarse bin and averages its reliability
    gap away.

    .. math::

        \mathrm{aECE} = \sum_{b} \frac{|B_b|}{N}
            \, \bigl| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \bigr|

    where :math:`\{B_b\}` partition the samples into ``n_bins`` equal-mass
    groups. Lower is better.

    ``preds`` are confidences (clipped to ``[0, 1]``); ``target`` are binary
    correctness indicators — same convention as :class:`ExpectedCalibrationError`.
    """

    higher_is_better = False

    def __init__(self, n_bins: int = 15, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.n_bins = int(n_bins)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        confs = _to_numpy(preds).ravel().astype(np.float64).clip(0, 1)
        correct = _to_numpy(target).ravel().astype(np.float64)
        if confs.size != correct.size or confs.size == 0:
            return torch.tensor(float("nan"), device=self.device)

        # Equal-mass edges via quantiles. Deduplicate to handle confidence
        # clustering (many identical values) gracefully — a degenerate input
        # with a single unique confidence collapses to one bin.
        n_bins = max(1, min(self.n_bins, confs.size))
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(confs, quantiles))
        if edges.size < 2:
            # All confidences identical → a single bin spanning everything.
            gap = abs(correct.mean() - confs.mean())
            return torch.tensor(float(gap), device=self.device)

        # np.digitize assigns each sample to a bin; clip into [0, n_edges-2].
        n_actual = edges.size - 1
        idx = np.clip(np.digitize(confs, edges[1:-1], right=False), 0, n_actual - 1)
        aece = 0.0
        for b in range(n_actual):
            in_bin = idx == b
            if not in_bin.any():
                continue
            avg_conf = confs[in_bin].mean()
            avg_acc = correct[in_bin].mean()
            aece += (in_bin.sum() / confs.size) * abs(avg_acc - avg_conf)
        return torch.tensor(float(aece), device=self.device)


# ── PR-8c Classwise ECE (multi-class calibration) ─────────────────────────────


@register_metric("classwise_ece", aliases=["cwece", "class_wise_ece"])
class ClasswiseECE(BaseMetric):
    r"""Classwise ECE — mean over classes of the per-class (one-vs-rest) ECE.

    Standard top-1 ECE only inspects the predicted (argmax) class. Classwise
    ECE evaluates the calibration of *every* class probability column against
    its one-vs-rest binary outcome, then averages, exposing miscalibration in
    the non-top classes (Kull et al., 2019).

    .. math::

        \mathrm{cwECE} = \frac{1}{K} \sum_{k=1}^{K}
            \mathrm{ECE}\bigl(p_{\cdot,k},\; \mathbb{1}[y = k]\bigr)

    Lower is better.

    Forward signature: ``preds`` is a ``(N, K)`` matrix of per-class
    probabilities; ``target`` is either ``(N,)`` integer labels or a
    ``(N, K)`` one-hot matrix. A single-column / single-class input is handled
    gracefully (averages over the one present class).
    """

    higher_is_better = False

    #: ``preds`` is a (N, K) probability matrix while ``target`` is (N,) labels — a
    #: DELIBERATE shape disagreement, so this metric opts out of the BaseMetric shape
    #: guard (which exists to stop *image* metrics silently broadcasting a wider
    #: prediction against a narrower target).
    REQUIRES_MATCHING_SHAPES: bool = False

    def __init__(self, n_bins: int = 15, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.n_bins = int(n_bins)

    def compute_metric(self, preds: Image, target: Image, **_: Any) -> Tensor:
        probs = _to_numpy(preds).astype(np.float64)
        tgt = _to_numpy(target)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        n, k = probs.shape
        probs = probs.clip(0.0, 1.0)

        # Derive a one-hot outcome matrix from either labels or one-hot input.
        if tgt.ndim >= 2 and tgt.shape[-1] == k:
            onehot = tgt.reshape(-1, k).astype(np.float64)
        else:
            labels = tgt.ravel().astype(int)
            if labels.size != n:
                return torch.tensor(float("nan"), device=self.device)
            onehot = np.zeros((n, k), dtype=np.float64)
            valid = (labels >= 0) & (labels < k)
            onehot[np.arange(n)[valid], labels[valid]] = 1.0

        per_class = [_equal_width_ece(probs[:, c], onehot[:, c], self.n_bins) for c in range(k)]
        per_class = [v for v in per_class if not np.isnan(v)]
        if not per_class:
            return torch.tensor(float("nan"), device=self.device)
        return torch.tensor(float(np.mean(per_class)), device=self.device)
