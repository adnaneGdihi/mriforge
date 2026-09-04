"""Region-restricted reconstruction metrics.

These compute a standard reconstruction error but only inside a supplied
spatial region of interest (ROI). The first member is ``banding_region_mse``,
the primary metric for the bSSFP banding-mitigation arm (vf_10): the science is
whether the combiner nulls the dark bands, so the error must be measured *in the
banding region*, not over the whole FOV where the bands are a tiny fraction of
the pixels.

The ROI is supplied at call time via a ``region_mask`` (or ``banding_mask``)
kwarg. Without a mask the metric **raises** ``MetricNotApplicableError``: there
is no honest number to return. It previously fell back to a full-FOV MSE behind
a log warning, which reported the wrong physical quantity under a name that
promises a region-restricted one — the full FOV is precisely the average the
banding ROI exists to see through, and a warning nobody reads is not consent
(pitfall #9). Callers that want a full-FOV MSE should ask for ``mse``.
"""

from __future__ import annotations

import torch

from spectramr.core.metrics.outcome import MetricNotApplicableError, NotApplicableReason
from spectramr.core.metrics.registry import register_metric


def _magnitude(x: torch.Tensor) -> torch.Tensor:
    """Magnitude for complex tensors; pass-through for real tensors."""
    return x.abs() if torch.is_complex(x) else x


@register_metric(name="banding_region_mse", aliases=["banding_mse"])
class BandingRegionMSEMetric:
    r"""MSE restricted to a banding / artefact ROI.

    .. math:: \mathrm{MSE}_M = \frac{\sum_i M_i\,|\hat{x}_i - x_i|^2}{\sum_i M_i}

    where :math:`M` is the ROI mask. Lower is better.
    """

    def __init__(self, eps: float = 1e-8, **_: object) -> None:
        self.eps = eps

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        region_mask: torch.Tensor | None = None,
        banding_mask: torch.Tensor | None = None,
        **_: object,
    ) -> float:
        pred = _magnitude(prediction)
        tgt = _magnitude(target)
        if pred.shape != tgt.shape:
            raise ValueError(
                f"banding_region_mse: prediction {tuple(pred.shape)} and target "
                f"{tuple(tgt.shape)} must have the same shape."
            )
        sq = (pred - tgt) ** 2

        mask = region_mask if region_mask is not None else banding_mask
        if mask is None:
            raise MetricNotApplicableError(
                "banding_region_mse",
                NotApplicableReason.MISSING_REQUIRED_ROI_MASK,
                "no `region_mask`/`banding_mask` supplied. This metric is defined "
                "only inside a banding ROI; a full-FOV MSE is a different quantity "
                "and returning one here would misreport it under this name. Pass the "
                "ROI via the metrics-computer kwargs, or ask for `mse` instead.",
            )

        mask = mask.to(device=sq.device, dtype=sq.dtype)
        # Broadcast a [H, W] or [B, 1, H, W] mask against [B, C, H, W].
        while mask.dim() < sq.dim():
            mask = mask.unsqueeze(0)
        num = (sq * mask).sum()
        den = mask.sum().clamp_min(self.eps)
        return float((num / den).item())

    @property
    def name(self) -> str:
        return "banding_region_mse"

    @property
    def higher_is_better(self) -> bool:
        return False


__all__ = ["BandingRegionMSEMetric"]
