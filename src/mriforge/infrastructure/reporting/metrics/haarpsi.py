r"""HaarPSI — Haar wavelet-based Perceptual Similarity Index.

R. Reisenhofer, S. Bosse, G. Kutyniok, T. Wiegand, "A Haar wavelet-
based perceptual similarity index for image quality assessment,"
Signal Process.: Image Commun., 61, 2018.

Per `TODO/report_step` Part A — among the strongest in modern MRI IQA
benchmarks. Higher is better.

This is a faithful but compact NumPy implementation: two-scale Haar
decomposition, per-scale similarity weighted by the local magnitude
response.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _haar_filter_bank(scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal/vertical Haar wavelet filters at a given scale."""
    f = 2**scale
    upper = np.ones((f, f))
    upper[f // 2 :, :] = -1
    h_filter = upper / (f * f)
    v_filter = h_filter.T
    return h_filter, v_filter


def haar_psi(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 1.0,
    c: float = 30.0,
    alpha: float = 4.2,
) -> float:
    r"""HaarPSI between two 2-D images. Higher is better."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    p = pred.astype(np.float64) / max(data_range, 1e-12) * 255.0
    t = target.astype(np.float64) / max(data_range, 1e-12) * 255.0

    sims: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for scale in (1, 2):
        for filt in _haar_filter_bank(scale):
            cp = signal.convolve2d(p, filt, mode="same", boundary="symm")
            ct = signal.convolve2d(t, filt, mode="same", boundary="symm")
            num = 2.0 * np.abs(cp) * np.abs(ct) + c
            den = cp * cp + ct * ct + c
            sims.append((num / den).clip(0, 1))
            weights.append(np.maximum(np.abs(cp), np.abs(ct)))
    sim_stack = np.stack(sims, axis=-1)
    w_stack = np.stack(weights, axis=-1)
    # per-pixel logit-mean across scales
    logit = lambda x: np.log(x.clip(1e-12, 1 - 1e-12) / (1 - x.clip(1e-12, 1 - 1e-12)))  # noqa: E731
    sigmoid = lambda y: 1.0 / (1.0 + np.exp(-y))  # noqa: E731
    weighted = (w_stack * logit(sim_stack)).sum(axis=-1) / w_stack.sum(axis=-1).clip(1e-12)
    score = sigmoid(alpha * weighted)
    return float(score.mean())
