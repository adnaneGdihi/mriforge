"""Residual descriptors for CDSCR.

The Cross-Domain Spectral Coherence ranker needs an ``energy-removed``
description of where a residual lives. Three complementary bases:

1. **Radial k-space spectral density** ``rho_bar(k)`` — separates aliasing
   (low k) from white noise (high k).
2. **Haar wavelet pyramid energies** — separates focal lesion-like residual
   from diffuse blur. Multi-scale, multi-orientation.
3. **First-order scattering coefficients** — Morlet-like oriented filters in
   Fourier domain. Captures motion-ghost stripe orientation that wavelets
   cannot resolve cleanly.

All three return *normalized* coefficients so the resulting feature vector
``zeta`` is invariant to ``||r||_2``.

The implementations use only ``torch.fft`` and stride-2 averaging — no
PyWavelets / kymatio dependency, which keeps this module portable to the
CPU-only test environment configured by ``conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mriforge.infrastructure.physics.fft_ops import fft2c


@dataclass
class DescriptorConfig:
    n_radial_bins: int = 16
    wavelet_levels: int = 3
    scattering_orientations: int = 6
    scattering_scales: int = 3


def radial_spectral_profile(residual: torch.Tensor, n_bins: int = 16) -> torch.Tensor:
    """Radial profile of |F(r)|^2 / ||r||^2.

    Args:
        residual: real or complex tensor [..., H, W]. If real, treated as a
            magnitude image; FFT is taken over the last two dims.
        n_bins: number of equi-width radial bins.

    Returns:
        Tensor of shape [n_bins], normalized to sum 1 over bins (energy-
        invariant).
    """
    if residual.numel() == 0:
        return torch.zeros(n_bins, dtype=torch.float64)

    if torch.is_complex(residual):
        spec = fft2c(residual.unsqueeze(0)).squeeze(0)
    else:
        # Treat as magnitude; centered FFT keeps DC at the array center.
        spec = fft2c(residual.to(torch.complex64).unsqueeze(0)).squeeze(0)

    power = (spec.real**2 + spec.imag**2).to(torch.float64)
    *_, h, w = power.shape
    cy, cx = h // 2, w // 2
    # Build the radial grid and accumulators on the SAME device as ``power``.
    # ``residual`` may arrive on CUDA (the meta-eval precomputes on GPU); the
    # previous CPU-default ``arange``/``zeros`` made ``scatter_add_`` mix a
    # CPU ``self``/index with a CUDA ``src`` and raised a device mismatch.
    device = power.device
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=device) - cy,
        torch.arange(w, dtype=torch.float64, device=device) - cx,
        indexing="ij",
    )
    radius = torch.sqrt(xx**2 + yy**2)
    r_max = float(radius.max().item()) + 1e-9
    bin_idx = torch.clamp((radius / r_max * n_bins).long(), 0, n_bins - 1)

    # Sum over channels and any leading batch dims by collapsing.
    while power.ndim > 2:
        power = power.sum(dim=0)

    flat_power = power.flatten()
    flat_idx = bin_idx.flatten()

    bin_sum = torch.zeros(n_bins, dtype=torch.float64, device=device)
    bin_count = torch.zeros(n_bins, dtype=torch.float64, device=device)
    bin_sum.scatter_add_(0, flat_idx, flat_power)
    bin_count.scatter_add_(0, flat_idx, torch.ones_like(flat_power))

    rho = bin_sum / bin_count.clamp(min=1.0)
    total = rho.sum()
    if total < 1e-30:
        return torch.zeros_like(rho)
    return rho / total


def haar_pyramid_energies(residual: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """Energy of the LH/HL/HH sub-bands at each level of a Haar pyramid.

    Returns 3*levels coefficients, each normalized by the residual's total
    energy. Operates on the magnitude image when given complex input.
    """
    if torch.is_complex(residual):
        x = residual.abs().to(torch.float64)
    else:
        x = residual.to(torch.float64)

    while x.ndim > 2:
        # Reduce to a 2-D image (sum over channel/batch).
        x = x.sum(dim=0)

    total_energy = (x**2).sum()
    if total_energy < 1e-30:
        return torch.zeros(3 * levels, dtype=torch.float64)

    out = []
    a = x
    for _ in range(levels):
        h, w = a.shape
        h2, w2 = (h // 2) * 2, (w // 2) * 2  # crop odd dims
        a = a[:h2, :w2]
        # Haar lifting via 2x2 blocks.
        ll = (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) * 0.5
        lh = (a[0::2, 0::2] + a[0::2, 1::2] - a[1::2, 0::2] - a[1::2, 1::2]) * 0.5
        hl = (a[0::2, 0::2] - a[0::2, 1::2] + a[1::2, 0::2] - a[1::2, 1::2]) * 0.5
        hh = (a[0::2, 0::2] - a[0::2, 1::2] - a[1::2, 0::2] + a[1::2, 1::2]) * 0.5
        out.extend(
            [
                (lh**2).sum() / total_energy,
                (hl**2).sum() / total_energy,
                (hh**2).sum() / total_energy,
            ]
        )
        a = ll
        if a.shape[0] < 2 or a.shape[1] < 2:
            break

    while len(out) < 3 * levels:
        out.append(torch.tensor(0.0, dtype=torch.float64))

    return torch.stack(out)


def scattering_orientation_energies(
    residual: torch.Tensor, n_orientations: int = 6, n_scales: int = 3
) -> torch.Tensor:
    """First-order scattering-style oriented energies in the Fourier domain.

    For each (scale, orientation) we apply a Morlet-like band-pass filter
    centered at frequency ``2^{-j} * (cos theta, sin theta)`` and integrate
    the squared magnitude. Order: outer loop = scales, inner loop =
    orientations.
    """
    if torch.is_complex(residual):
        x = residual
    else:
        x = residual.to(torch.complex64)

    while x.ndim > 2:
        x = x.sum(dim=0)

    spec = fft2c(x.unsqueeze(0)).squeeze(0)
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(
        (torch.arange(h, dtype=torch.float64) - cy) / max(cy, 1),
        (torch.arange(w, dtype=torch.float64) - cx) / max(cx, 1),
        indexing="ij",
    )
    # Polar coordinates in normalized frequency.
    rho = torch.sqrt(xx**2 + yy**2) + 1e-9
    phi = torch.atan2(yy, xx)

    total_energy = (spec.real**2 + spec.imag**2).sum().to(torch.float64)
    if total_energy < 1e-30:
        return torch.zeros(n_scales * n_orientations, dtype=torch.float64)

    coeffs = []
    for j in range(n_scales):
        center_rho = 2.0 ** (-(j + 1))  # center frequency
        radial_bw = max(center_rho * 0.5, 1.0 / max(h, w))
        radial = torch.exp(-((rho - center_rho) ** 2) / (2 * radial_bw**2))
        for o in range(n_orientations):
            theta = o * torch.pi / n_orientations
            # Cosine angular window centered at theta with bandwidth pi/n.
            angular = torch.cos(torch.clamp((phi - theta), -torch.pi, torch.pi)) ** 2
            mask = (radial * angular).to(torch.float64)
            band_energy = ((spec.real**2 + spec.imag**2).to(torch.float64) * mask).sum()
            coeffs.append(band_energy / total_energy)

    return torch.stack(coeffs)


def residual_descriptor(
    residual: torch.Tensor, config: DescriptorConfig | None = None
) -> torch.Tensor:
    """Compose a single ``zeta`` vector from radial + wavelet + scattering."""
    cfg = config or DescriptorConfig()
    # These descriptors are float64 image statistics composed with
    # ``torch.cat`` and later stacked across samples (see cdscr.py). The
    # sub-functions create grids/accumulators internally, so a CUDA residual
    # would yield parts on mixed devices (some ops default new tensors to
    # CPU) and break ``cat``/``stack``. Normalise to CPU once here — float64
    # is CPU-native and the tensors are tiny, so there is no perf cost.
    if residual.is_cuda:
        residual = residual.detach().to("cpu")
    parts = [
        radial_spectral_profile(residual, cfg.n_radial_bins),
        haar_pyramid_energies(residual, cfg.wavelet_levels),
        scattering_orientation_energies(
            residual, cfg.scattering_orientations, cfg.scattering_scales
        ),
    ]
    return torch.cat(parts, dim=0)
