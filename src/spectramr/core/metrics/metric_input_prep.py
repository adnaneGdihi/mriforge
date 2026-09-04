"""Canonical input-finiteness guard for perceptual image metrics.

Perceptual metrics (LPIPS, FID, ...) feed their inputs through a frozen VGG /
Inception backbone after range-normalising to a fixed interval and expanding to
three channels. The historical per-metric normalisers used a silent
``torch.clamp(x, lo, hi)``: ``clamp`` leaves NaN/Inf **unchanged**, so a model
that diverged and emitted non-finite predictions had its NaN passed straight
through. The failure then surfaced deep inside torchmetrics as a misleading
*"Expected ... normalized tensors ... values in range [nan, nan] ... when all
values are expected to be in the [-1, 1] range"* error — pointing the reader at
a range/shape problem when the real cause was an unstable model (the 2026-06-24
``exp_hm_06`` / ``exp_hm_10`` Hyper-Mamba crash).

This module is the single source of truth for the *finiteness* half of the
perceptual-metric input contract. It raises a precise, actionable error at the
metric boundary instead of silently forwarding NaN — CLAUDE.md pitfall #9 (no
silent fallbacks) and #10 (a non-finite metric must fail loudly, with a message
that names the real fix).
"""

from __future__ import annotations

import torch

__all__ = ["assert_finite_metric_input"]


def assert_finite_metric_input(metric_name: str, **named_tensors: torch.Tensor | None) -> None:
    """Raise ``ValueError`` if any named perceptual-metric input is non-finite.

    Args:
        metric_name: Metric identifier surfaced in the error (e.g. ``"lpips"``),
            so the message names *which* metric received the bad tensor.
        **named_tensors: Tensors to check, keyed by their role (``preds`` /
            ``target`` / ...). ``None`` values are skipped (an absent optional
            reference is not an error).

    Raises:
        ValueError: If any provided tensor contains NaN or Inf. The message
            names the metric and the offending field and reports the NaN/Inf
            counts, and attributes the failure to model instability rather than
            to the metric — fix the pipeline that produced the predictions,
            never silence the metric (pitfall #9).

    Notes:
        The cheap ``torch.isfinite(t).all()`` reduction runs per tensor; the
        per-element NaN/Inf counts (which force a host sync) are computed only
        on the error path. This lives outside the training loop (validation /
        periodic metric computation), so the guard's reduction does not violate
        the no-GPU-sync-in-the-hot-loop rule.
    """
    for field, tensor in named_tensors.items():
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            n_nan = int(torch.isnan(tensor).sum().item())
            n_inf = int(torch.isinf(tensor).sum().item())
            raise ValueError(
                f"Metric {metric_name!r} received non-finite '{field}' "
                f"({n_nan} NaN, {n_inf} Inf of {tensor.numel()} elements). "
                f"The model produced unstable (NaN/Inf) output; fix model / "
                f"training stability rather than the metric — silently "
                f"normalising NaN would hide the divergence (CLAUDE.md "
                f"pitfall #9)."
            )
