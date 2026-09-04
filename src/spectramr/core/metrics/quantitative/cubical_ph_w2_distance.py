"""``cubical_ph_w2_distance``: the persistent-homology Wasserstein-2 distance as a validation metric.

The ``geomamba_ulf`` cohort's claim is topological (the cubical PH-W2 loss
"sharpens fine cortical topology"), yet every arm selected and reported on
``val_psnr`` alone: no registered metric measured topology, so the claim
carried no number (cohort review 2026-09-03). This metric scores a prediction
against its target with the SAME sublevel-set cubical persistence and W2
matching the loss uses --
:class:`~spectramr.models.losses.cubical_ph_w2_loss.CubicalPHWassersteinLoss`
is the one owner of the diagram routine and the matching; this module only
calls it without a gradient and reduces over the batch. Lower is better; ``0``
at ``pred == target``. Needs the ``[topology]`` extra (gudhi + POT): the
constructor raises the loss's own ``ImportError`` when it is absent, never a
silent zero.

The direction is a plain class attribute, not a property. The direction
resolver (:func:`~spectramr.core.metrics.metric_directions.metric_higher_is_better`)
reads a bool off the class without constructing the metric, and constructing
this one needs the extra: a property would leave ``val_cubical_ph_w2_distance``
unresolvable -- and checkpoint selection on it refused -- on every box without
gudhi.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spectramr.core.metrics.registry import register_metric

__all__ = ["CubicalPHW2DistanceMetric"]


def _batched_magnitude(x: Tensor) -> Tensor:
    """Detached real ``(B, C, *spatial)`` view: complex -> magnitude, ``(H, W)`` -> ``(1, 1, H, W)``."""
    out = x.detach()
    if out.is_complex():
        out = out.abs()
    while out.ndim < 4:
        out = out.unsqueeze(0)
    return out


@register_metric("cubical_ph_w2_distance", aliases=["ph_w2_distance", "topology_w2"])
class CubicalPHW2DistanceMetric:
    """Mean cubical PH-W2 distance between prediction and target (lower is better).

    Args:
        wasserstein_p: Wasserstein order handed to the loss (``2`` is the cohort's;
            the loss refuses ``< 1``).
        **kwargs: absorbed (``device=`` from the metrics computer).
    """

    name: str = "cubical_ph_w2_distance"
    higher_is_better: bool = False

    def __init__(self, wasserstein_p: int = 2, **kwargs: Any) -> None:
        from spectramr.models.losses.cubical_ph_w2_loss import CubicalPHWassersteinLoss

        # The loss's constructor is where the optional-dependency contract lives
        # (ImportError with the install hint); a metric must not re-state it.
        self._loss = CubicalPHWassersteinLoss(wasserstein_p=wasserstein_p)

    def __call__(self, prediction: Tensor, target: Tensor, **kwargs: Any) -> float:
        """Score one batch; ``mask`` (foreground, broadcastable to the prediction) is honoured.

        Every other kwarg the metrics computer hands over (``device``, ``domain``,
        ``data_range``) is absorbed: the distance is taken in the tensors' own units.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"cubical_ph_w2_distance needs matching shapes; got {tuple(prediction.shape)} "
                f"and {tuple(target.shape)}."
            )
        pred = _batched_magnitude(prediction).float()
        tgt = _batched_magnitude(target).float()
        mask = kwargs.get("mask")
        if mask is not None:
            # The loss demands a mask of exactly the prediction's shape; expand a
            # (B, 1, ...) foreground mask across channels, and let a shape that
            # cannot broadcast raise here rather than inside the loss.
            mask = _batched_magnitude(torch.as_tensor(mask, device=pred.device)).expand(pred.shape)
        with torch.no_grad():
            value = self._loss(pred, tgt, mask=mask)
        return float(value)
