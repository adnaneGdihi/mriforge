"""Spectral (FNO-truncated) branch for the CS-MNO architecture.

This module implements the global low-frequency branch of CS-MNO.
The forward pass is the standard FNO truncation:

1. ``rfftn`` to the real-valued frequency domain.
2. Multiply the lowest ``modes`` coefficients per axis by a learnt
   complex weight tensor.
3. ``irfftn`` back to the spatial domain.

This branch inherits FNO's universal-approximation guarantee on the
global structure of the target operator (``K^{-s/d}`` term in
Theorem B of ``TODO/Mamba_NO.md``).

The 2-D variant is a thin wrapper around the existing
``fno_block.SpectralConv2d`` so we don't duplicate the well-tested
Xavier initialisation. The 3-D variant is new because the existing
``fno_block`` only ships a 2-D version.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .fno_block import SpectralConv2d

__all__ = ["SpectralBranch1d", "SpectralBranch2d", "SpectralBranch3d"]


class SpectralBranch1d(nn.Module):
    """1-D FNO branch — `K`-mode truncation with phase-preserving rfft.

    Mirrors :class:`SpectralBranch2d` for sequence-like inputs. Used by
    CS-MNO when ``spatial_dims=1`` (e.g. 1-D PDE benchmarks like
    Burgers, where the input is a single spatial sequence per sample).
    The Mamba block in CS-MNO is already 1-D natively; this branch
    just supplies the global low-frequency (FNO) component.
    """

    def __init__(self, dim: int, modes: int) -> None:
        super().__init__()
        self.dim = dim
        self.modes = modes
        scale = (2.0 / (dim + dim)) ** 0.5
        # Single learnt weight tensor over the lowest `modes` rfft
        # coefficients. Layout matches the 2-D weights1: [in, out, k].
        self.weights = nn.Parameter(scale * torch.view_as_complex(torch.randn(dim, dim, modes, 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, C, L]``; output has the same shape."""
        L = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        m = min(self.modes, x_ft.shape[-1])
        # Apply learnt complex weight to the low-frequency coefficients.
        out_ft[..., :m] = torch.einsum("bil,iol->bol", x_ft[..., :m], self.weights[..., :m])
        return torch.fft.irfft(out_ft, n=L, dim=-1)


class SpectralBranch2d(nn.Module):
    """2-D FNO branch — `K`-mode truncation with phase-preserving rfft2."""

    def __init__(self, dim: int, modes: int) -> None:
        super().__init__()
        self.dim = dim
        self.modes = modes
        self.spectral = SpectralConv2d(dim, dim, modes, modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, C, H, W]``; output has the same shape."""
        return self.spectral(x)


class SpectralBranch3d(nn.Module):
    """3-D FNO branch — analogous to ``SpectralConv2d`` for volumetric data.

    Uses ``torch.fft.rfftn`` over the trailing three axes and applies a
    learnt complex weight to the lowest ``modes`` coefficients per
    axis. The half-spectrum convention of ``rfftn`` means the last
    axis only needs ``modes`` low frequencies, while the first two
    need ``modes`` from each end (low-low, low-high, high-low,
    high-high quadrants — analogous to the 2-D weights1/weights2
    pair).
    """

    def __init__(self, dim: int, modes: int) -> None:
        super().__init__()
        self.dim = dim
        self.modes = modes
        scale = (2.0 / (dim + dim)) ** 0.5
        # Four learnt weight tensors covering the (±, ±, +) frequency
        # quadrants of the rfftn output (last axis is half-spectrum so
        # only +). Mirrors the 2-D convention in `fno_block`.
        shape = (dim, dim, modes, modes, modes)
        self.weights1 = nn.Parameter(scale * torch.view_as_complex(torch.randn(*shape, 2)))
        self.weights2 = nn.Parameter(scale * torch.view_as_complex(torch.randn(*shape, 2)))
        self.weights3 = nn.Parameter(scale * torch.view_as_complex(torch.randn(*shape, 2)))
        self.weights4 = nn.Parameter(scale * torch.view_as_complex(torch.randn(*shape, 2)))

    @staticmethod
    def _compl_mul(input_: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bidhw,iodhw->bodhw", input_, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, C, D, H, W]``; output has the same shape."""
        D, H, W = x.shape[-3:]
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        m = min(self.modes, D, H, x_ft.size(-1))
        out_ft = torch.zeros_like(x_ft)

        # Four corner blocks of the (D, H, W//2+1) spectrum.
        out_ft[..., :m, :m, :m] = self._compl_mul(
            x_ft[..., :m, :m, :m], self.weights1[..., :m, :m, :m]
        )
        out_ft[..., -m:, :m, :m] = self._compl_mul(
            x_ft[..., -m:, :m, :m], self.weights2[..., :m, :m, :m]
        )
        out_ft[..., :m, -m:, :m] = self._compl_mul(
            x_ft[..., :m, -m:, :m], self.weights3[..., :m, :m, :m]
        )
        out_ft[..., -m:, -m:, :m] = self._compl_mul(
            x_ft[..., -m:, -m:, :m], self.weights4[..., :m, :m, :m]
        )

        return torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))
