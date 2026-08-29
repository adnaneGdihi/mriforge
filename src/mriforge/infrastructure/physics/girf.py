"""GIRF (Gradient Impulse Response Function) preprocessor.

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-9.

The GIRF characterises the linear-time-invariant transfer function
from requested gradient waveform ``G_req(t)`` to actually-played
``G_act(t)``.  Per axis it is a 1-D convolution; the corrected
trajectory is

    ``k_act(t) = γ ∫ G_req(τ) ⋆ h(t − τ) dτ``

where ``h`` is the GIRF impulse response (per axis).  This module
provides:

- :func:`apply_girf` — time-domain per-axis FIR filter (implemented
  with ``F.conv1d``, i.e. the cross-correlation convention — PyTorch
  conv ops do not flip the kernel) that applies a learned ``h`` to a
  (B, 3, T) requested waveform. For the delta-initialised (symmetric)
  kernel this matches the LTI convolution above exactly; a learned
  asymmetric ``h`` is applied under the cross-correlation sign
  convention (flip the kernel if true convolution is required).
- :func:`waveform_to_trajectory` — running cumulative sum
  approximation of ``γ ∫ G dτ`` so callers can chain
  ``waveform → corrected_waveform → corrected_trajectory``.
- :class:`GIRFKernel` — learnable per-axis FIR filter initialised
  as a delta so the untrained filter is a no-op.

The strategy module
:mod:`mriforge.infrastructure.training.strategies.girf_aware_strategy`
uses these primitives to pre-correct trajectories before the NUFFT
forward operator.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA_RAD_PER_S_PER_T = 2 * math.pi * 42.577_478_92e6


def apply_girf(waveform: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply per-axis GIRF FIR filter to a gradient waveform.

    waveform: (B, A, T) where A is number of axes (typically 3).
    kernel:   (A, 1, K) FIR coefficients (one per axis).
    Returns:  (B, A, T) — same shape as input.
    """
    if waveform.dim() != 3:
        raise ValueError(f"waveform must be (B, A, T), got {tuple(waveform.shape)}")
    B, A, T = waveform.shape
    if kernel.shape[0] != A:
        raise ValueError(f"kernel axis count {kernel.shape[0]} != waveform axes {A}")
    K = kernel.shape[-1]
    pad = (K - 1) // 2
    return F.conv1d(waveform, kernel, padding=pad, groups=A)


def waveform_to_trajectory(
    waveform: torch.Tensor,
    dt: float = 1e-5,
) -> torch.Tensor:
    """Discrete cumulative integration: ``k(t) = γ Σ G(τ) Δτ``.

    waveform: (B, A, T) gradient amplitudes (T/m).
    Returns:  (B, A, T) trajectory (rad/m, then user-side scale).
    """
    return GAMMA_RAD_PER_S_PER_T * dt * torch.cumsum(waveform, dim=-1)


class GIRFKernel(nn.Module):
    """Learnable per-axis GIRF FIR initialised as delta."""

    def __init__(self, n_axes: int = 3, kernel_size: int = 21) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = torch.zeros(n_axes, 1, kernel_size)
        kernel[:, 0, kernel_size // 2] = 1.0
        self.kernel = nn.Parameter(kernel)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return apply_girf(waveform, self.kernel)

    def delta_distance(self) -> torch.Tensor:
        delta = torch.zeros_like(self.kernel)
        delta[:, 0, self.kernel.shape[-1] // 2] = 1.0
        return (self.kernel - delta).pow(2).mean()


__all__ = [
    "GAMMA_RAD_PER_S_PER_T",
    "GIRFKernel",
    "apply_girf",
    "waveform_to_trajectory",
]
