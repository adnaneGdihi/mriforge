"""Closed-form perceptual full-reference IQA metrics (piq-backed).

The 2026 MRI image-quality-assessment survey
(*A Study of Why We Need to Reassess Full Reference Image Quality Assessment
with Medical Images*, J. Imaging Inform. Med. 2025) evaluates a broad
full-reference panel — PSNR, SSIM, MS-SSIM, IW-SSIM, VIF, GMSD, MS-GMSD, FSIM,
VSI, MDSI, HaarPSI, LPIPS, DISTS, PieAPP, DSS. This codebase already had PSNR,
SSIM, MS-SSIM, VIF, GMSD, FSIM, LPIPS and DISTS, but was missing several widely
used closed-form perceptual metrics that correlate strongly with radiologist
opinion. We add them here by wrapping the validated implementations from
``piq`` (PyTorch Image Quality) rather than reimplementing — the same pattern
the codebase already uses for ``torchmetrics`` (FID/KID) and ``lpips``.

Metrics added:

* ``haarpsi`` — Haar wavelet Perceptual Similarity Index (Reisenhofer et al.,
  2018). Higher is better.
* ``mdsi`` — Mean Deviation Similarity Index (Nafchi et al., 2016). Lower is
  better.
* ``vsi`` — Visual Saliency-Induced Index (Zhang et al., 2014). Higher is
  better.
* ``dss`` — DCT Subband Similarity (Balanov et al., 2015). Higher is better.
* ``ms_gmsd`` — Multi-Scale Gradient Magnitude Similarity Deviation (Zhang et
  al., 2017). Lower is better.

``piq`` is an optional dependency: if it is not installed the metric raises at
construction so the registry consumer (e.g. the sim2rank engine) skips it
gracefully, exactly like the radiomic metrics under a missing ``pyradiomics``.
"""

from __future__ import annotations

import torch

from spectramr.core.metrics.evaluation_metrics import BaseMetric
from spectramr.core.metrics.registry import register_metric
from spectramr.core.types import Image, Tensor

try:  # optional dependency
    import piq

    PIQ_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without piq
    piq = None
    PIQ_AVAILABLE = False


def _prepare_pair(preds: Image, target: Image) -> tuple[Tensor, Tensor]:
    """Coerce a (pred, target) pair to ``piq``'s contract: real, 1-channel,
    ``[0, 1]`` with a shared scale so relative degradation is preserved."""
    p = preds.abs() if preds.is_complex() else preds.float()
    t = target.abs() if target.is_complex() else target.float()
    # piq accepts 1- or 3-channel; collapse anything else to magnitude.
    if p.shape[1] not in (1, 3):
        p = p.mean(dim=1, keepdim=True)
    if t.shape[1] not in (1, 3):
        t = t.mean(dim=1, keepdim=True)
    # Shared scale (the reference's peak) → data_range = 1.0.
    scale = t.amax().clamp_min(1e-8)
    return (p / scale).clamp(0, 1), (t / scale).clamp(0, 1)


class _PIQMetric(BaseMetric):
    """Base wrapper: dispatch to a ``piq`` *full-reference* functional metric."""

    _PIQ_FN: str = ""
    # Some piq metrics (IW-SSIM, like MS-SSIM) require a minimum spatial size;
    # small MRI patches are bilinearly upsampled to satisfy it.
    _MIN_SIZE: int = 0

    def __init__(self, device: str | torch.device = "cpu") -> None:
        if not PIQ_AVAILABLE:
            raise ImportError(
                f"{type(self).__name__} requires the optional 'piq' package (pip install piq)."
            )
        super().__init__(device)

    def _maybe_upsample(self, p: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        import torch.nn.functional as F

        if self._MIN_SIZE and (p.shape[-2] < self._MIN_SIZE or p.shape[-1] < self._MIN_SIZE):
            size = (self._MIN_SIZE, self._MIN_SIZE)
            p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return p, t

    def compute_metric(self, preds: Image, target: Image, **kwargs) -> Tensor:
        import warnings

        p, t = _prepare_pair(preds.to(self.device), target.to(self.device))
        p, t = self._maybe_upsample(p, t)
        fn = getattr(piq, self._PIQ_FN)
        with warnings.catch_warnings():
            # MDSI/VSI emit a benign "converted to RGB" UserWarning on grayscale
            # MRI input — silence it so it doesn't pollute cluster smoke logs.
            warnings.simplefilter("ignore", UserWarning)
            return fn(p, t, data_range=1.0)


class _PIQMetricNR(_PIQMetric):
    """Base wrapper for piq *no-reference* metrics (single image; target ignored)."""

    def compute_metric(self, preds: Image, target: Image, **kwargs) -> Tensor:
        import warnings

        p, _ = _prepare_pair(preds.to(self.device), preds.to(self.device))
        p, _ = self._maybe_upsample(p, p)
        fn = getattr(piq, self._PIQ_FN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return fn(p)


@register_metric("haarpsi", aliases=["HaarPSI"])
class HaarPSIMetric(_PIQMetric):
    """Haar wavelet Perceptual Similarity Index (higher is better)."""

    _PIQ_FN = "haarpsi"


@register_metric("mdsi", aliases=["MDSI"])
class MDSIMetric(_PIQMetric):
    """Mean Deviation Similarity Index (lower is better)."""

    _PIQ_FN = "mdsi"


@register_metric("vsi", aliases=["VSI"])
class VSIMetric(_PIQMetric):
    """Visual Saliency-Induced Index (higher is better)."""

    _PIQ_FN = "vsi"


@register_metric("dss", aliases=["DSS"])
class DSSMetric(_PIQMetric):
    """DCT Subband Similarity (higher is better)."""

    _PIQ_FN = "dss"


@register_metric("ms_gmsd", aliases=["MS-GMSD", "multi_scale_gmsd"])
class MSGMSDMetric(_PIQMetric):
    """Multi-Scale Gradient Magnitude Similarity Deviation (lower is better)."""

    _PIQ_FN = "multi_scale_gmsd"


@register_metric("iw_ssim", aliases=["IW-SSIM", "information_weighted_ssim"])
class IWSSIMMetric(_PIQMetric):
    """Information-content-Weighted SSIM (Wang & Li, 2011; higher is better)."""

    _PIQ_FN = "information_weighted_ssim"
    _MIN_SIZE = 161  # piq requires at least 161x161


@register_metric("srsim", aliases=["SR-SIM", "spectral_residual_sim"])
class SRSIMMetric(_PIQMetric):
    """Spectral Residual based Similarity (Zhang & Li, 2012; higher is better)."""

    _PIQ_FN = "srsim"


@register_metric("vif_p", aliases=["VIFp", "vif_pixel"])
class VIFpMetric(_PIQMetric):
    """Pixel-domain Visual Information Fidelity (Sheikh & Bovik; higher is better)."""

    _PIQ_FN = "vif_p"


@register_metric("total_variation", aliases=["TotalVariation", "tv"], requires_reference=False)
class TotalVariationMetric(_PIQMetricNR):
    """Total Variation (no-reference; rises with noise / fine texture)."""

    _PIQ_FN = "total_variation"
