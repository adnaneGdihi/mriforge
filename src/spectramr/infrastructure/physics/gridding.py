"""Density compensation + gridding helpers for non-Cartesian k-space.

This module complements ``nufft_ops.py`` / ``nufft_utils.py`` (which wrap
``torchkbnufft``) with the missing density-compensation function (DCF)
estimators and a thin gridding kernel used by analytical reconstructors.

Two DCF estimators are provided (both autograd-friendly):

- ``RadialDCF`` — analytic |kr| weighting for radial trajectories.
- ``IterativeDCF`` — Pipe-Menon iterative DCF for arbitrary trajectories.

A simple ``KaiserBesselKernel`` apodization helper is exposed for use in
analytic gridding inside iterative solvers.

References:

- Pipe, J.G. and Menon, P. (1999). "Sampling density compensation in
  MRI: rationale and an iterative numerical solution." MRM 41:179-186.
- Beatty, P.J. et al. (2005). "Rapid gridding reconstruction with a
  minimal oversampling ratio." TMI 24:799-808.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RadialDCF(nn.Module):
    """Analytic density-compensation weights for radial trajectories.

    For a uniform polar trajectory the DCF is proportional to ``|k_r|``;
    the central sample receives a special weight (1 / (2 * num_spokes)).

    Args:
        num_spokes: number of radial spokes acquired.
        samples_per_spoke: samples per spoke (used only for normalisation).
    """

    def __init__(self, num_spokes: int, samples_per_spoke: int) -> None:
        super().__init__()
        if num_spokes < 1:
            raise ValueError(f"num_spokes must be >= 1, got {num_spokes}")
        if samples_per_spoke < 2:
            raise ValueError(f"samples_per_spoke must be >= 2, got {samples_per_spoke}")
        self.num_spokes = int(num_spokes)
        self.samples_per_spoke = int(samples_per_spoke)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """Compute per-sample weights.

        Args:
            traj: trajectory tensor of shape ``[..., 2]`` with normalised
                k-space coordinates in ``[-pi, pi]``.

        Returns:
            Per-sample weights with the same leading shape as ``traj`` minus
            the trailing dim.
        """
        if traj.shape[-1] != 2:
            raise ValueError(f"traj must end in 2 (kx, ky), got {tuple(traj.shape)}")
        kr = torch.linalg.vector_norm(traj, dim=-1)
        # Central sample handling.
        central_w = 1.0 / (2.0 * self.num_spokes)
        weights = torch.where(kr > 1e-6, kr, torch.full_like(kr, central_w))
        # Normalise so weights sum to ~ samples_per_spoke for stable scaling.
        return weights / weights.mean().clamp(min=1e-12)


class IterativeDCF(nn.Module):
    """Pipe-Menon iterative DCF for an arbitrary trajectory.

    Iteratively re-weights the trajectory so the convolution with the
    (Kaiser-Bessel) gridding kernel sums to unity at every sample.

    Args:
        kernel_width: gridding kernel width (samples).
        beta: Kaiser-Bessel shape parameter.
        num_iters: fixed-point iteration count.
        eps: stabiliser used in the kernel-weight division.
    """

    def __init__(
        self,
        kernel_width: float = 4.0,
        beta: float = 13.9,
        num_iters: int = 25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if kernel_width <= 0 or beta <= 0:
            raise ValueError(f"kernel_width and beta must be > 0, got {kernel_width}, {beta}")
        self.kernel_width = float(kernel_width)
        self.beta = float(beta)
        self.num_iters = int(num_iters)
        self.eps = float(eps)

    def _kb_kernel(self, distance: torch.Tensor) -> torch.Tensor:
        """Kaiser-Bessel kernel evaluated at sample distance."""
        # arg = beta * sqrt(1 - (2 d / W)^2) inside the support.
        ratio = (2.0 * distance / self.kernel_width).clamp(max=1.0)
        arg = self.beta * torch.sqrt((1.0 - ratio.pow(2)).clamp(min=0.0))
        # Modified Bessel I_0; torch.special.i0 keeps autograd intact.
        kb = torch.special.i0(arg)
        # Zero outside the support.
        return torch.where(2.0 * distance <= self.kernel_width, kb, torch.zeros_like(kb))

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """Compute DCF weights.

        Args:
            traj: trajectory ``[N, 2]`` (flattened across batch / spokes).

        Returns:
            Weights of shape ``[N]``.
        """
        if traj.dim() != 2 or traj.shape[-1] != 2:
            raise ValueError(f"traj must be [N, 2], got {tuple(traj.shape)}")
        n = traj.shape[0]
        weights = torch.ones(n, device=traj.device, dtype=traj.dtype)
        # Pairwise distances — O(n^2). Caller is responsible for batching.
        # For typical 2D trajectories with n<=1e4 this fits comfortably.
        diff = traj.unsqueeze(0) - traj.unsqueeze(1)  # [N, N, 2]
        dist = torch.linalg.vector_norm(diff, dim=-1)  # [N, N]
        kernel = self._kb_kernel(dist)  # [N, N]
        for _ in range(self.num_iters):
            denom = kernel @ weights  # [N]
            weights = weights / denom.clamp(min=self.eps)
        return weights / weights.mean().clamp(min=1e-12)


class KaiserBesselKernel(nn.Module):
    """Standalone Kaiser-Bessel apodization kernel for analytical gridding.

    Args:
        kernel_width: width in samples.
        beta: shape parameter.
    """

    def __init__(self, kernel_width: float = 4.0, beta: float = 13.9) -> None:
        super().__init__()
        if kernel_width <= 0 or beta <= 0:
            raise ValueError(f"kernel_width and beta must be > 0, got {kernel_width}, {beta}")
        self.kernel_width = float(kernel_width)
        self.beta = float(beta)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        ratio = (2.0 * distance / self.kernel_width).clamp(max=1.0)
        arg = self.beta * torch.sqrt((1.0 - ratio.pow(2)).clamp(min=0.0))
        kb = torch.special.i0(arg)
        return torch.where(2.0 * distance <= self.kernel_width, kb, torch.zeros_like(kb))

    @property
    def normalization(self) -> float:
        """Closed-form L1 normaliser of the 1D KB kernel."""
        return float(self.kernel_width * math.pi / self.beta)


__all__ = [
    "IterativeDCF",
    "KaiserBesselKernel",
    "RadialDCF",
]
