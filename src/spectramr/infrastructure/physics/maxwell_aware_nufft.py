r"""Maxwell-aware concomitant NUFFT forward model (A-3.4 *MA-NUFFT*).

At low field the imaging gradients produce **concomitant (Maxwell) fields** whose
lowest-order term scales as :math:`f_c \propto G^2/(2 B_0)` (Hz) — negligible at
3 T, large at 0.1 T. During a non-Cartesian readout this imprints a spatially
varying phase :math:`\phi_c(\mathbf r)=2\pi f_c(\mathbf r)\,t` that a plain NUFFT
ignores, biasing the reconstruction. MA-NUFFT **embeds** that concomitant phase
into the non-uniform forward model:

.. math::

    y(\mathbf k) \;=\; \sum_{\mathbf r} x(\mathbf r)\,
        e^{i\phi_c(\mathbf r)}\,e^{-i 2\pi \mathbf k\cdot\mathbf r},

i.e. apply :math:`e^{i\phi_c}` (from
:class:`~spectramr.infrastructure.physics.concomitant_phase_operator.ConcomitantPhaseOperator`
/ :func:`~spectramr.infrastructure.physics.maxwell_concomitant.maxwell_concomitant_field_hz`)
*before* the non-uniform transform. As :math:`B_0\to\infty` (or zero gradient)
:math:`\phi_c\to 0` and MA-NUFFT reduces to a plain NUFFT.

The non-uniform transform here is an **exact direct NDFT** (the operator is
correct by construction and dependency-light — no ``torchkbnufft`` needed for the
forward / its oracles); for production the same concomitant pre-modulation drops in
front of the gridded NUFFT in ``nufft_ops``. This single-readout-time snapshot is
the standard first-order concomitant model; a time-segmented variant (the
concomitant phasor evaluated per readout-time segment) is the rigorous extension
documented for a follow-up.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def maxwell_concomitant_phase(
    spatial_shape: tuple[int, int],
    gradient_amp_mt_per_m: float,
    field_strength_t: float,
    *,
    voxel_size_mm: float = 1.0,
    readout_time_s: float = 1e-3,
    gamma_hz_per_t: float = 42.577e6,
) -> Tensor:
    r"""First-order concomitant phase map :math:`\phi_c(\mathbf r)=2\pi f_c t` ``[H, W]`` (radians).

    Lowest-order Maxwell term for an in-plane gradient :math:`G`:
    :math:`f_c(\mathbf r)=\gamma\,G^2 (x^2+y^2)/(2 B_0)` (Hz), with ``x, y`` the
    voxel offsets from iso-centre (metres). Scales as :math:`1/B_0` and
    :math:`G^2`; ``0`` for zero gradient.
    """
    if field_strength_t <= 0:
        raise ValueError(f"field_strength_t must be > 0, got {field_strength_t}")
    h, w = spatial_shape
    vs_m = voxel_size_mm * 1e-3
    g_t_per_m = gradient_amp_mt_per_m * 1e-3  # mT/m -> T/m
    yy = (torch.arange(h, dtype=torch.float64) - (h - 1) / 2.0) * vs_m
    xx = (torch.arange(w, dtype=torch.float64) - (w - 1) / 2.0) * vs_m
    r2 = yy[:, None] ** 2 + xx[None, :] ** 2  # [H, W], metres^2
    f_c = gamma_hz_per_t * (g_t_per_m**2) * r2 / (2.0 * field_strength_t)  # Hz
    return (2.0 * math.pi * f_c * readout_time_s).to(torch.float32)


def _ndft_basis(ktraj: Tensor, h: int, w: int) -> Tensor:
    r"""Exact NDFT basis ``[S, H, W]`` :math:`e^{-i2\pi(k_x w/W + k_y h/H)}`.

    ``ktraj`` is ``[S, 2]`` in cycles-per-FOV (``k_x, k_y``); integer ``ktraj`` on
    the grid reproduces the standard DFT.
    """
    ww = torch.arange(w, dtype=torch.float64, device=ktraj.device)
    hh = torch.arange(h, dtype=torch.float64, device=ktraj.device)
    kx = ktraj[:, 0].double()[:, None, None]
    ky = ktraj[:, 1].double()[:, None, None]
    phase = -2.0 * math.pi * (kx * ww[None, None, :] / w + ky * hh[None, :, None] / h)
    return torch.polar(torch.ones_like(phase), phase)  # [S, H, W] complex128


def maxwell_aware_nufft_forward(
    image: Tensor, ktraj: Tensor, concomitant_phase: Tensor | None = None
) -> Tensor:
    r"""Maxwell-aware NUFFT forward: :math:`y(\mathbf k)=\mathrm{NDFT}(x\,e^{i\phi_c})`.

    Args:
        image: ``[B, 1, H, W]`` (or ``[B, H, W]``), real or complex.
        ktraj: ``[S, 2]`` non-uniform k-space samples (cycles-per-FOV, ``k_x, k_y``).
        concomitant_phase: ``[H, W]`` (or broadcastable) phase in radians, or
            ``None`` for a plain NUFFT.

    Returns:
        complex ``[B, S]`` non-Cartesian samples.
    """
    x = image
    if x.dim() == 3:
        x = x[:, None]
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"image must be [B,1,H,W] or [B,H,W], got {tuple(image.shape)}")
    x = x[:, 0].to(torch.complex128)
    if concomitant_phase is not None:
        phasor = torch.polar(torch.ones_like(concomitant_phase), concomitant_phase).to(
            torch.complex128
        )
        x = x * phasor
    h, w = x.shape[-2:]
    basis = _ndft_basis(ktraj, h, w)  # [S, H, W]
    return torch.einsum("bhw,shw->bs", x, basis).to(torch.complex64)


class MaxwellAwareNUFFT:
    """Maxwell-aware concomitant NUFFT forward operator (A-3.4 MA-NUFFT).

    Holds the concomitant phase map (from the Maxwell field, scaling as
    :math:`G^2/B_0`) and applies it before the non-uniform transform. ``None`` /
    high-field phase reduces to a plain NUFFT.
    """

    def __init__(self, concomitant_phase: Tensor | None = None) -> None:
        self.concomitant_phase = concomitant_phase

    def forward(self, image: Tensor, ktraj: Tensor) -> Tensor:
        return maxwell_aware_nufft_forward(image, ktraj, self.concomitant_phase)

    def metadata(self) -> dict[str, Any]:
        """Provenance: whether the concomitant correction is active + its scale."""
        cp = self.concomitant_phase
        return {
            "concomitant_active": cp is not None,
            "max_concomitant_phase_rad": float(cp.abs().max()) if cp is not None else 0.0,
        }


__all__ = [
    "MaxwellAwareNUFFT",
    "maxwell_aware_nufft_forward",
    "maxwell_concomitant_phase",
]
