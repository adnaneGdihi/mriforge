r"""Synthetic-to-real fidelity bound (A-8.5 *SRF-Bound*).

When a method is trained or validated on **synthetic** data (BrainWeb / XCAT /
Bloch-simulated k-space) the central question is *how far is the synthetic
distribution from real acquisitions?* — because a model that excels on synthetic
data may have learned simulator artefacts that do not transfer. SRF-Bound
quantifies that gap as a **Wasserstein-1 distance in a low-dimensional, training-
free feature space** between a set of synthetic images and a set of real images.
Lower ⇒ the synthetic distribution is more faithful to real.

**Feature space (training-free, MRI-meaningful).** Each image is reduced to its
**radial power-spectrum band log-energies** (``n_bands`` bins of the 2-D power
spectrum by spatial frequency). This is the natural axis on which simulators
differ from real MRI — noise colour, resolution roll-off and texture all live in
the radial spectrum — and it needs no learned feature extractor (so the metric is
reproducible and asset-free, unlike a deep-feature FID). The power spectrum uses
``torch.fft`` on a **real** magnitude image (a spectral *feature*, not a complex
k-space round-trip — exempt from the ``fft2c`` SSOT rule, per ``physics.md``).

**Distance.** The **mean per-band 1-D Wasserstein-1** between the two sets'
band-energy samples. Per band, the 1-D W1 is computed on a common quantile grid
(an accurate **approximation** of :math:`\int_0^1|F_a^{-1}(q)-F_b^{-1}(q)|\,dq`
whose accuracy is set by ``n_quantiles``; handles unequal sample sizes). Averaging
the per-band *marginal* W1s is a **tractable proxy** for the joint feature-space W1
— it aggregates the per-band discrepancies and **ignores cross-band coupling**, so
it is a fidelity *estimate*, deliberately not the exact joint optimal transport
(which would need a full :math:`d`-dimensional OT solve). An empty radial band
(only on tiny images with many bands — never for :math:`\ge 32`-px images with
``n_bands <= 8``) contributes a 0 log-energy and hence 0 to the average.

Reads the two image sets from ``kwargs``: ``synthetic=`` + ``real=`` (each a
``[N, C, H, W]`` tensor or a list of images), or ``prediction`` as the synthetic
set plus ``real_reference=``. With no real reference set it returns
``float('nan')`` — there is no synthetic-to-real gap to measure without real data
(a fabricated 0 would falsely certify perfect fidelity; CLAUDE.md #9 / #20).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import register_metric


def _stack_images(obj: Any) -> torch.Tensor | None:
    """Coerce a ``[N,C,H,W]`` tensor or a list of images to ``[N,1,H,W]`` magnitude."""
    from spectramr.core.metrics.nr_texture import _as_magnitude_2d

    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        mags = [_as_magnitude_2d(x) for x in obj]
        mags = [m for m in mags if m is not None]
        if not mags:
            return None
        return torch.cat(mags, dim=0)
    if isinstance(obj, torch.Tensor):
        return _as_magnitude_2d(obj)
    return None


def radial_band_energies(images: torch.Tensor, n_bands: int = 8) -> torch.Tensor:
    """Per-image radial power-spectrum band log-energies ``[N, n_bands]``.

    ``images`` is ``[N, 1, H, W]`` real magnitude. Bins the centred 2-D power
    spectrum into ``n_bands`` equal-width radial-frequency bands and returns
    ``log1p`` of the per-band energy (mean over the band's bins).
    """
    n, _c, h, w = images.shape
    spec = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
    psd = spec.real**2 + spec.imag**2  # [N, 1, H, W]
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=images.device)).abs()[:, None]
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=images.device)).abs()[None, :]
    radius = torch.sqrt(fy**2 + fx**2)  # [H, W], in [0, ~0.707]
    rmax = float(radius.max()) + 1e-8
    edges = torch.linspace(0.0, rmax, n_bands + 1, device=images.device)
    out = images.new_zeros(n, n_bands)
    flat_psd = psd.reshape(n, -1)
    flat_r = radius.reshape(-1)
    for b in range(n_bands):
        hi = edges[b + 1]
        if b == n_bands - 1:
            # Inclusive right edge on the LAST band: edges[-1] == radius.max() after
            # linspace quantization, so a strict `<` would drop the max-frequency
            # pixel(s) and silently lose the highest-frequency energy.
            band = (flat_r >= edges[b]) & (flat_r <= hi)
        else:
            band = (flat_r >= edges[b]) & (flat_r < hi)
        if not bool(band.any()):
            # Empty band (only on tiny images with many bands; never for >=32px,
            # n_bands<=8). Leaves a 0 log-energy -> contributes 0 to the per-band W1.
            continue
        out[:, b] = flat_psd[:, band].mean(dim=1)
    return torch.log1p(out)


def w1_1d(a: torch.Tensor, b: torch.Tensor, n_quantiles: int = 256) -> float:
    """Approximate 1-D Wasserstein-1 between two empirical samples ``a``, ``b``.

    :math:`W_1 = \\int_0^1 |F_a^{-1}(q) - F_b^{-1}(q)|\\,dq`, evaluated as a
    quantile-grid Riemann sum (``n_quantiles`` points; handles unequal sample
    sizes; accuracy improves with ``n_quantiles``). Symmetric and ``0`` iff the
    empirical CDFs match.
    """
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    q = torch.linspace(0.0, 1.0, n_quantiles, device=a.device)
    qa = torch.quantile(a, q)
    qb = torch.quantile(b, q)
    return float((qa - qb).abs().mean())


@register_metric(
    "srf_bound",
    aliases=["synthetic_real_fidelity_bound", "sim2real_w1", "srf_w1"],
    requires_reference=False,
    direction="lower",
)
class SyntheticToRealFidelityBound:
    r"""Synthetic-to-real fidelity bound — mean per-band feature-space W1. Lower is better.

    The mean per-radial-band 1-D Wasserstein-1 between a synthetic image set and a
    real image set in radial-spectrum feature space — a tractable lower bound on
    the joint feature-space W1. Reads ``synthetic=``/``real=`` (or ``prediction``
    + ``real_reference=``) from kwargs; returns ``nan`` without a real reference.

    Args:
        n_bands: Number of radial power-spectrum bands (the feature dimension).
        n_quantiles: Quantile-grid resolution for the per-band 1-D W1.
    """

    def __init__(self, n_bands: int = 8, n_quantiles: int = 256) -> None:
        if n_bands < 1:
            raise ValueError(f"n_bands must be >= 1, got {n_bands}")
        self.n_bands = int(n_bands)
        self.n_quantiles = int(n_quantiles)

    @property
    def name(self) -> str:
        return "srf_bound"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        # ``context`` is accepted for the NR-metric contract but unused (this metric
        # reads its two image sets from kwargs, not from MetricContext fields).
        del context
        synthetic = _stack_images(kwargs.get("synthetic", prediction))
        real = _stack_images(kwargs.get("real", kwargs.get("real_reference")))
        if synthetic is None or real is None:
            return float("nan")
        feats_s = radial_band_energies(synthetic, self.n_bands)
        feats_r = radial_band_energies(real, self.n_bands)
        per_band = [
            w1_1d(feats_s[:, b], feats_r[:, b], self.n_quantiles) for b in range(self.n_bands)
        ]
        finite = [v for v in per_band if math.isfinite(v)]
        if not finite:
            return float("nan")
        return float(sum(finite) / len(finite))


# ``core/metrics/__init__`` walks every submodule on import (pkgutil), so the
# ``@register_metric`` above fires without editing ``__init__.py``.

__all__ = [
    "SyntheticToRealFidelityBound",
    "radial_band_energies",
    "w1_1d",
]
