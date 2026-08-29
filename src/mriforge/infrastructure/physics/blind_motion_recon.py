r"""Operator-estimating blind-motion reconstruction (A-4.1 *OER*).

Rigid motion during a segmented acquisition corrupts the reconstruction: each
k-space **segment** is acquired at a different motion state, so the segments are
mutually inconsistent and the image ghosts. OER **jointly estimates the motion
operator and the image** ("operator-estimating") by alternating:

* **image update** (given the per-segment translations :math:`\{t_s\}`):
  demodulate each segment by :math:`e^{+i2\pi k\cdot t_s}` (the inverse Fourier
  shift) and recombine — the motion-consistent estimate.
* **motion update** (given the image): choose each :math:`t_s` to **minimise an
  image-sharpness metric** of the recombined reconstruction (gradient entropy —
  the classic *autofocus* criterion), by an integer grid search.

**Why autofocus, not data consistency.** With disjoint segments, data-consistency
alone is *non-identifiable*: each segment perfectly explains its own measurements
at any translation by adjusting the image (the A-3.2 JIFI field-image confound).
Motion is recovered instead from the *image*: a correct demodulation removes the
ghosting and concentrates the image gradients (low entropy), while a wrong one
leaves ghosts (spread gradients, high entropy). A **global** translation is a
reference-frame ambiguity (Fourier-shift theorem), so segment 0 is fixed as the
reference and the rest are recovered *relative* to it.

A self-contained reconstruction primitive (small synthetic problems for its
oracles), not a training paradigm; it reuses the autofocus principle of the
existing ``gradient_entropy_loss`` (vf_35).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c


def _kspace_shift_phasor(h: int, w: int, ty: float, tx: float, device: Any) -> Tensor:
    r"""Fourier-shift phase ramp :math:`e^{-i2\pi(k_x t_x + k_y t_y)}`, ``[H, W]`` complex.

    ``fftshift`` puts the frequency axis in the CENTERED layout that ``fft2c``
    emits, so the ramp is indexed the same way the k-space it multiplies is. An
    unshifted ``fftfreq`` ramp against centered k-space applies the wrong
    frequency at every index — a scramble, not a translation.
    """
    ky = torch.fft.fftshift(torch.fft.fftfreq(h, device=device))[:, None]
    kx = torch.fft.fftshift(torch.fft.fftfreq(w, device=device))[None, :]
    phase = -2.0 * torch.pi * (kx * tx + ky * ty)
    return torch.polar(torch.ones_like(phase), phase)


def gradient_entropy(image: Tensor, eps: float = 1e-12) -> float:
    r"""Autofocus sharpness metric: Shannon entropy of the normalised gradient magnitude.

    Low for a sharp (motion-free) image — gradients concentrate on true edges;
    high for a ghosted image — motion spreads gradient energy. Computed on the
    image magnitude.
    """
    mag = image.abs() if image.is_complex() else image
    gx = mag[..., :, 1:] - mag[..., :, :-1]
    gy = mag[..., 1:, :] - mag[..., :-1, :]
    g = torch.cat([gx.reshape(-1).abs(), gy.reshape(-1).abs()])
    total = float(g.sum()) + eps
    p = g / total
    return float(-(p * torch.log(p + eps)).sum())


def simulate_segmented_motion(
    image: Tensor, segment_masks: list[Tensor], translations: list[tuple[float, float]]
) -> Tensor:
    r"""Build motion-corrupted k-space: each segment carries its own rigid translation."""
    if len(segment_masks) != len(translations):
        raise ValueError("segment_masks and translations must have equal length")
    h, w = image.shape[-2:]
    x_k = fft2c(image.to(torch.complex64))
    y = torch.zeros_like(x_k)
    for m, (ty, tx) in zip(segment_masks, translations, strict=True):
        phasor = _kspace_shift_phasor(h, w, ty, tx, image.device)
        y = y + m.to(x_k.dtype) * (x_k * phasor)
    return y


def _recombine(
    corrupted: Tensor, masks: list[Tensor], trans: list[tuple[int, int]], phasors: dict
) -> Tensor:
    """Demodulate each segment by its current shift and recombine -> k-space estimate."""
    x_k = torch.zeros_like(corrupted)
    for s, m in enumerate(masks):
        x_k = x_k + m * (corrupted * phasors[trans[s]].conj())
    return x_k


def estimate_blind_motion(
    corrupted_kspace: Tensor,
    segment_masks: list[Tensor],
    *,
    search_radius: int = 4,
    n_outer: int = 4,
) -> dict[str, Any]:
    r"""Autofocus blind-motion estimate (OER): recover ``{t_s}`` + the image.

    Segment 0 is the reference (``t_0=(0,0)``); the rest are estimated relative to
    it by an integer grid search over ``[-search_radius, search_radius]^2`` that
    minimises the recombined image's gradient entropy (autofocus).

    Returns ``{translations, image, kspace, entropy_history}`` — entropy_history is
    the autofocus objective per outer iteration (non-increasing as motion is
    resolved).
    """
    h, w = corrupted_kspace.shape[-2:]
    dev = corrupted_kspace.device
    n_seg = len(segment_masks)
    masks = [m.to(corrupted_kspace.dtype) for m in segment_masks]
    offsets = list(range(-search_radius, search_radius + 1))
    phasors = {
        (ty, tx): _kspace_shift_phasor(h, w, ty, tx, dev) for ty in offsets for tx in offsets
    }
    trans: list[tuple[int, int]] = [(0, 0)] * n_seg
    entropy_history: list[float] = []

    for _ in range(n_outer):
        for s in range(1, n_seg):  # segment 0 fixed as the reference frame
            best, best_t = None, trans[s]
            for ty in offsets:
                for tx in offsets:
                    trans[s] = (ty, tx)
                    img = ifft2c(_recombine(corrupted_kspace, masks, trans, phasors))
                    e = gradient_entropy(img)
                    if best is None or e < best:
                        best, best_t = e, (ty, tx)
            trans[s] = best_t
        x_k = _recombine(corrupted_kspace, masks, trans, phasors)
        entropy_history.append(gradient_entropy(ifft2c(x_k)))

    x_k = _recombine(corrupted_kspace, masks, trans, phasors)
    return {
        "translations": trans,
        "image": ifft2c(x_k),
        "kspace": x_k,
        "entropy_history": entropy_history,
    }


__all__ = [
    "estimate_blind_motion",
    "gradient_entropy",
    "simulate_segmented_motion",
]
