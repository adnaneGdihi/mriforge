"""Physics + statistical EDA quantities. Reads complex k-space via the fft_ops SSOT."""

from __future__ import annotations

import numpy as np
import torch


def _central_slice(ksp: torch.Tensor) -> torch.Tensor:
    """Coerce k-space to ``[C, H, W]`` (single coil → C=1)."""
    if ksp.ndim == 2:
        return ksp.unsqueeze(0)
    while ksp.ndim > 3:
        ksp = ksp[ksp.shape[0] // 2]
    return ksp


def _flat_mag(images: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [(im.abs() if im.is_complex() else im).detach().cpu().numpy().ravel() for im in images]
    )


def radial_energy(ksp: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radially-averaged ``|k|^2`` vs spatial-frequency radius (coil-summed).

    Returns ``(radii, mean_energy, cumulative_fraction)``.
    """
    k = _central_slice(ksp)
    power = (k.abs() ** 2).sum(0).detach().cpu().numpy()  # [H,W]
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    nbins = int(r.max()) + 1
    radial = np.zeros(nbins)
    counts = np.zeros(nbins)
    np.add.at(radial, r.ravel(), power.ravel())
    np.add.at(counts, r.ravel(), 1)
    mean = radial / np.maximum(counts, 1)
    cumulative = np.cumsum(radial) / max(radial.sum(), 1e-12)
    return np.arange(nbins), mean, cumulative


def detect_sampling(ksp: torch.Tensor) -> dict:
    """Detect zero-filled phase-encode lines → mask / acceleration / ACS / fully-sampled."""
    k = _central_slice(ksp)
    line_energy = (k.abs() ** 2).sum(dim=(0, 2)).detach().cpu().numpy()  # per phase-encode row
    sampled = line_energy > (line_energy.max() * 1e-6)
    n = len(sampled)
    n_sampled = int(sampled.sum())
    acc = n / max(n_sampled, 1)
    center = n // 2
    acs = 0
    i = center
    while 0 <= i < n and sampled[i]:
        acs += 1
        i += 1
    return {
        "mask": sampled.astype(np.uint8),
        "acceleration": float(acc),
        "acs_size": int(acs),
        "fully_sampled": bool(n_sampled >= n - 1),
    }


def _bounded_kspace(ksp: torch.Tensor, max_coils: int = 8, max_fov: int = 128) -> torch.Tensor:
    """Cap coils and center-crop the FOV of a ``[C, H, W]`` k-space slice.

    ESPIRiT's per-pixel Gram matrix and eigendecomposition allocate ``O(H·W·C²)`` memory,
    which OOM-**kills** the EDA process (an uncatchable SIGKILL that aborts the whole suite)
    on a large multicoil slice. Coil sensitivities are smooth / low-frequency, so a
    center-cropped (low-resolution) estimate over a subset of coils is fully representative
    for visualization. The crop is symmetric about the centered DC, preserving the low
    frequencies that carry the sensitivity structure.
    """
    k = _central_slice(ksp)
    if k.shape[0] > max_coils:
        k = k[:max_coils]
    _c, h, w = k.shape
    if h > max_fov:
        top = (h - max_fov) // 2
        k = k[:, top : top + max_fov, :]
    if w > max_fov:
        left = (w - max_fov) // 2
        k = k[:, :, left : left + max_fov]
    return k


def coil_maps(ksp: torch.Tensor, max_coils: int = 8, max_fov: int = 128):
    """ESPIRiT coil sensitivities for a multicoil k-space slice; ``None`` if single-coil/failed.

    The input is bounded (:func:`_bounded_kspace`) so the ESPIRiT eigendecomposition cannot
    OOM-kill the EDA process on a large multicoil slice (e.g. ``ocmr``).
    """
    k = _bounded_kspace(ksp, max_coils=max_coils, max_fov=max_fov)
    if k.shape[0] < 2:
        return None
    try:
        from spectramr.infrastructure.physics.coil_sensitivity import estimate_csm_espirit

        return estimate_csm_espirit(k.unsqueeze(0), num_coils=k.shape[0]).squeeze(0)
    except Exception:
        return None


def snr_estimate(image: torch.Tensor) -> dict:
    img = (image.abs() if image.is_complex() else image).detach().cpu().numpy()
    corner = np.concatenate([img[:8, :8].ravel(), img[-8:, -8:].ravel()])
    fg = img[img > np.percentile(img, 75)]
    bg_std = float(corner.std() + 1e-8)
    return {"fg_mean": float(fg.mean()), "bg_std": bg_std, "snr": float(fg.mean() / bg_std)}


def noise_floor(image: torch.Tensor) -> dict:
    img = (image.abs() if image.is_complex() else image).detach().cpu().numpy()
    corner = np.concatenate([img[:8, :8].ravel(), img[-8:, -8:].ravel()])
    return {"bg_values": corner, "sigma": float(corner.std()), "mean": float(corner.mean())}


def phase_stats(complex_image: torch.Tensor) -> dict:
    ph = torch.angle(complex_image).detach().cpu().numpy()
    gy, gx = np.gradient(ph)
    return {"phase": ph, "grad_mag": np.sqrt(gy**2 + gx**2)}


def normalized_distributions(images: list[torch.Tensor]) -> dict[str, np.ndarray]:
    """Foreground voxel intensities under four normalization schemes."""
    vals = _flat_mag(images)
    fg = vals[vals > np.percentile(vals, 50)]
    if fg.size == 0:
        fg = vals
    p1, p99 = np.percentile(fg, [1, 99])
    return {
        "raw": fg,
        "max": fg / max(fg.max(), 1e-8),
        "pclip": np.clip((fg - p1) / max(p99 - p1, 1e-8), 0, 1),
        "zscore": (fg - fg.mean()) / max(fg.std(), 1e-8),
    }


def percentiles(images: list[torch.Tensor]) -> dict:
    vals = _flat_mag(images)
    return {f"p{q}": float(np.percentile(vals, q)) for q in (0.1, 1, 50, 99, 99.9, 100)}
