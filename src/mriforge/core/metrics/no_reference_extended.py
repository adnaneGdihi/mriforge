"""Extended no-reference (blind) artifact-detection metrics.

Classical computer-vision and MRI image-quality measures that flag specific
degradations **without a clean reference** — complementing the learned blind
metrics already present (BRISQUE, NIQE). Every metric collapses complex /
multi-channel input to a single real grayscale channel, so they apply uniformly
across magnitude images, complex reconstructions and multi-coil stacks.

Metrics (registry key — what it detects):

* ``brenner_focus`` — Brenner (1971) gradient focus measure; blur lowers it.
* ``immerkaer_noise`` — Immerkaer (1996) fast noise-variance estimate; a
  Laplacian-like mask insensitive to structure. Higher = noisier.
* ``mlv`` — Maximum Local Variation (Bahrami & Kot, 2014); blur lowers it.
* ``intensity_entropy`` — Shannon entropy of the intensity histogram.
* ``blockiness`` — blocking / aliasing-grid artifact energy (Wang et al., 2000).
* ``high_freq_energy_ratio`` — fraction of spectral energy above a radial
  cutoff; a frequency-domain sharpness measure.
* ``gibbs_ringing`` — oscillation (Laplacian) energy in a band adjacent to
  strong edges; flags Fourier-truncation ringing.

All are no-reference: ``target`` is accepted for interface compatibility but
ignored.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mriforge.core.metrics.evaluation_metrics import BaseMetric
from mriforge.core.metrics.registry import register_metric
from mriforge.core.types import Image, Tensor


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    """Collapse complex / multi-channel input to a real grayscale ``(B, 1, H, W)``."""
    if x.is_complex():
        x = x.abs()
    x = x.float()
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    return x


@register_metric("brenner_focus", aliases=["brenner", "BrennerFocus"], requires_reference=False)
class BrennerFocus(BaseMetric):
    """Brenner gradient focus measure (no-reference; higher = sharper)."""

    def compute_metric(self, preds: Image, target: Image, **kwargs) -> Tensor:
        x = _to_gray(preds.to(self.device))
        dx = x[..., :, 2:] - x[..., :, :-2]
        dy = x[..., 2:, :] - x[..., :-2, :]
        energy = (x**2).sum() + 1e-8
        return ((dx**2).sum() + (dy**2).sum()) / energy


@register_metric(
    "immerkaer_noise",
    aliases=["noise_sigma", "ImmerkaerNoise"],
    requires_reference=False,
)
class ImmerkaerNoise(BaseMetric):
    """Immerkaer (1996) fast noise standard-deviation estimate (no-reference).

    Convolves with a Laplacian-like mask that is (nearly) insensitive to image
    structure, isolating noise: :math:`\\sigma = \\sqrt{\\pi/2}\\,
    \\frac{1}{6(W-2)(H-2)}\\sum |I * N|`. Higher = noisier.
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs) -> Tensor:
        x = _to_gray(preds.to(self.device))
        mask = torch.tensor(
            [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
            device=x.device,
        ).view(1, 1, 3, 3)
        conv = F.conv2d(x, mask)  # "valid" → (H-2, W-2)
        h, w = x.shape[-2], x.shape[-1]
        denom = 6.0 * max(1, (w - 2)) * max(1, (h - 2))
        return math.sqrt(math.pi / 2.0) * conv.abs().sum() / denom


@register_metric("mlv", aliases=["maximum_local_variation", "MLV"], requires_reference=False)
class MaximumLocalVariation(BaseMetric):
    """Maximum Local Variation (Bahrami & Kot, 2014; higher = sharper).

    Per pixel, the maximum absolute difference to its 8 neighbours; the metric
    is the spread (std) of that map — sharp images have a heavier MLV tail.
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs) -> Tensor:
        x = _to_gray(preds.to(self.device))
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
        h, w = x.shape[-2], x.shape[-1]
        diffs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = xp[..., 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
                diffs.append((x - shifted).abs())
        mlv = torch.stack(diffs, dim=0).amax(dim=0)
        return mlv.std()


@register_metric(
    "intensity_entropy",
    aliases=["IntensityEntropy", "shannon_entropy"],
    requires_reference=False,
)
class IntensityEntropy(BaseMetric):
    """Shannon entropy of the (min-max normalized) intensity histogram."""

    def compute_metric(self, preds: Image, target: Image, bins: int = 256, **kwargs) -> Tensor:
        x = _to_gray(preds.to(self.device))
        mn, mx = x.amin(), x.amax()
        xn = (x - mn) / (mx - mn + 1e-8)
        hist = torch.histc(xn.reshape(-1), bins=bins, min=0.0, max=1.0)
        p = hist / (hist.sum() + 1e-8)
        p = p[p > 0]
        return -(p * torch.log2(p + 1e-12)).sum()


@register_metric(
    "blockiness",
    aliases=["Blockiness", "blocking_artifact"],
    requires_reference=False,
)
class Blockiness(BaseMetric):
    """Blocking / aliasing-grid artifact ratio (no-reference; higher = more grid).

    Ratio of mean gradient magnitude *on* an ``block``-periodic grid to that
    *off* the grid (Wang et al., 2000). ~1 for natural images, large when hard
    block/aliasing edges dominate.
    """

    def compute_metric(self, preds: Image, target: Image, block: int = 8, **kwargs) -> Tensor:
        x = _to_gray(preds.to(self.device))
        dh = (x[..., :, 1:] - x[..., :, :-1]).abs()
        dv = (x[..., 1:, :] - x[..., :-1, :]).abs()
        col = torch.arange(dh.shape[-1], device=x.device)
        row = torch.arange(dv.shape[-2], device=x.device)
        bcol = (col + 1) % block == 0
        brow = (row + 1) % block == 0
        if bcol.sum() == 0 or brow.sum() == 0 or (~bcol).sum() == 0 or (~brow).sum() == 0:
            return torch.tensor(1.0, device=x.device)
        on = (dh[..., :, bcol].mean() + dv[..., brow, :].mean()) / 2.0
        off = (dh[..., :, ~bcol].mean() + dv[..., ~brow, :].mean()) / 2.0 + 1e-8
        return on / off


@register_metric(
    "high_freq_energy_ratio",
    aliases=["HighFreqEnergyRatio", "hf_energy_ratio"],
    requires_reference=False,
)
class HighFreqEnergyRatio(BaseMetric):
    """Fraction of spectral energy above a radial cutoff (no-reference).

    A frequency-domain sharpness measure: blur attenuates high frequencies and
    lowers the ratio. This is a real-valued spectral *feature* (not a complex
    MRI k-space round-trip), so a plain FFT is appropriate here — the
    ``fft2c`` rule targets k-space correctness, not feature spectra.
    """

    def compute_metric(self, preds: Image, target: Image, cutoff: float = 0.25, **kwargs) -> Tensor:
        x = _to_gray(preds.to(self.device))
        spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        power = spec.real**2 + spec.imag**2
        h, w = x.shape[-2], x.shape[-1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        hf = torch.sqrt(xx**2 + yy**2) > cutoff
        total = power.sum() + 1e-8
        return power[..., hf].sum() / total


@register_metric("gibbs_ringing", aliases=["GibbsRinging", "ringing"], requires_reference=False)
class GibbsRinging(BaseMetric):
    """Gibbs / truncation ringing measure (no-reference; higher = more ringing).

    Oscillation (Laplacian) energy in a band adjacent to — but not on — strong
    edges, normalized by edge strength. Fourier truncation produces ringing
    lobes beside edges that raise this ratio.
    """

    def compute_metric(self, preds: Image, target: Image, **kwargs) -> Tensor:
        x = _to_gray(preds.to(self.device))
        sx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3)
        sy = torch.tensor([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).view(1, 1, 3, 3)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-12)
        lap_k = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device).view(1, 1, 3, 3)
        lap = F.conv2d(x, lap_k, padding=1).abs()

        thr = torch.quantile(grad.reshape(-1), 0.9)
        edges = (grad >= thr).float()
        dilated = F.max_pool2d(edges, kernel_size=5, stride=1, padding=2)
        band = (dilated > 0) & (edges == 0)
        if band.sum() == 0 or edges.sum() == 0:
            return torch.tensor(0.0, device=x.device)
        return lap[band].mean() / (grad[edges > 0].mean() + 1e-8)
