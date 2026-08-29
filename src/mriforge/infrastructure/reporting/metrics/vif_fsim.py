r"""Visual Information Fidelity (VIF) and Feature Similarity Index (FSIM).

VIF — H. R. Sheikh, A. C. Bovik, "Image information and visual
quality," IEEE TIP, 15(2), 2006. Tier-1 radiologist correlate per
`TODO/report_step` Part A.

FSIM — L. Zhang *et al.*, "FSIM: A feature similarity index for image
quality assessment," IEEE TIP, 20(8), 2011. Top performer on motion /
noise / undersampling MRI.

Both implementations are pixel-space approximations: the full GSM /
phase-congruency formulations are ~200 lines apiece, so we use the
canonical `sewar` library when available and fall back to compact
NumPy approximations otherwise. Higher is better for both.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def _try_sewar() -> object | None:
    try:
        import sewar  # type: ignore

        return sewar
    except Exception:
        return None


def visual_information_fidelity(
    pred: np.ndarray, target: np.ndarray, *, data_range: float = 1.0
) -> float:
    """VIF (Sheikh & Bovik 2006). Range typically [0, 1]; higher = better."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    sewar = _try_sewar()
    if sewar is not None:
        return float(sewar.vifp(target, pred))  # type: ignore[attr-defined]

    # Pyramid-free approximation: per-block GSM information ratio.
    p = pred.astype(np.float64) / max(data_range, 1e-12)
    t = target.astype(np.float64) / max(data_range, 1e-12)
    sigma_n2 = 2.0 / (255.0**2)
    info_ref = 0.0
    info_dist = 0.0
    for sigma in (1.0, 2.0, 4.0, 8.0):
        mu_t = ndimage.gaussian_filter(t, sigma)
        mu_p = ndimage.gaussian_filter(p, sigma)
        var_t = ndimage.gaussian_filter(t * t, sigma) - mu_t * mu_t
        var_p = ndimage.gaussian_filter(p * p, sigma) - mu_p * mu_p
        cov = ndimage.gaussian_filter(t * p, sigma) - mu_t * mu_p
        g = cov / var_t.clip(min=1e-10)
        sv_sq = (var_p - g * cov).clip(min=0)
        info_ref += float(np.log2(1 + var_t / sigma_n2).sum())
        info_dist += float(np.log2(1 + (g**2) * var_t / (sv_sq + sigma_n2)).sum())
    if info_ref < 1e-12:
        return 0.0
    return info_dist / info_ref


# ----- FSIM -----


def _phase_congruency(image: np.ndarray) -> np.ndarray:
    """Coarse phase-congruency proxy via a magnitude-of-bandpass-energies map."""
    pc = np.zeros_like(image, dtype=np.float64)
    for sigma in (0.5, 1.0, 2.0, 4.0):
        log_g = ndimage.gaussian_laplace(image.astype(np.float64), sigma=sigma)
        pc = pc + np.abs(log_g)
    return pc / 4.0


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8.0
    gx = signal.convolve2d(image, sobel_x, mode="same", boundary="symm")
    gy = signal.convolve2d(image, sobel_x.T, mode="same", boundary="symm")
    return np.sqrt(gx * gx + gy * gy)


def feature_similarity_index(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 1.0,
    t1: float = 0.85,
    t2: float = 160.0,
) -> float:
    """FSIM (Zhang et al. 2011). Higher is better."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    sewar = _try_sewar()
    if sewar is not None:
        return float(sewar.fsim(target, pred))  # type: ignore[attr-defined]

    p = pred.astype(np.float64) / max(data_range, 1e-12) * 255.0
    t = target.astype(np.float64) / max(data_range, 1e-12) * 255.0
    pc_p = _phase_congruency(p)
    pc_t = _phase_congruency(t)
    g_p = _gradient_magnitude(p)
    g_t = _gradient_magnitude(t)
    s_pc = (2 * pc_p * pc_t + t1) / (pc_p**2 + pc_t**2 + t1)
    s_g = (2 * g_p * g_t + t2) / (g_p**2 + g_t**2 + t2)
    s = s_pc * s_g
    pc_max = np.maximum(pc_p, pc_t)
    return float((s * pc_max).sum() / pc_max.sum().clip(min=1e-12))
