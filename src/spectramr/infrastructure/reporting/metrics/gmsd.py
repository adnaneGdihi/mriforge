r"""Gradient-Magnitude Similarity Deviation (GMSD).

W. Xue, L. Zhang, X. Mou, A. C. Bovik, "Gradient magnitude similarity
deviation: A highly efficient perceptual image quality index," IEEE
TIP, 23(2), 2014. Per `TODO/report_step` Part A — high SROCC and very
low cost.

Lower is better.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

_PREWITT_X = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
_PREWITT_Y = _PREWITT_X.T
_C = 170.0


def _grad_mag(x: np.ndarray) -> np.ndarray:
    gx = signal.convolve2d(x, _PREWITT_X, mode="same", boundary="symm")
    gy = signal.convolve2d(x, _PREWITT_Y, mode="same", boundary="symm")
    return np.sqrt(gx * gx + gy * gy)


def gradient_magnitude_similarity_deviation(
    pred: np.ndarray, target: np.ndarray, *, data_range: float = 1.0
) -> float:
    r"""GMSD between two 2-D images.

    Args:
        pred, target: Real-valued 2-D arrays of the same shape, in
            roughly the same intensity range.
        data_range: Peak-to-peak value used internally to normalise C
            (default 1.0). For 8-bit-equivalent data pass 255.0.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    p = pred.astype(np.float64) / max(data_range, 1e-12) * 255.0
    t = target.astype(np.float64) / max(data_range, 1e-12) * 255.0
    g_p = _grad_mag(p)
    g_t = _grad_mag(t)
    gms = (2.0 * g_p * g_t + _C) / (g_p**2 + g_t**2 + _C)
    return float(np.std(gms))
