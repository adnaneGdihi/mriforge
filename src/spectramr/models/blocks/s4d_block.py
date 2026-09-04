"""S4D — Diagonal Structured State Space Model (PR-22 / Part-I D).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-22.

Reference: Gu et al. 2022, "On the Parameterization and Initialization
of Diagonal State Space Models." S4D is a diagonalized variant of S4
that drops the HiPPO structure for a per-channel diagonal A and recovers
most of the performance with much simpler implementation.

This block implements the convolutional-mode S4D:

    y_t = Σ_{k=0}^{L-1} K_k * u_{t-k}
    K_k = (1/L) Σ_n  C_n * exp(k * Δ * λ_n)

where ``λ_n`` is a complex eigenvalue of the diagonal A and ``Δ`` is a
learnable timescale.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class S4DBlock(nn.Module):
    """Diagonal SSM block with FFT-convolutional inference.

    Args:
        d_model: per-channel dimension (kept the same on input + output).
        d_state: number of diagonal SSM states per channel.
        dropout: post-mix dropout probability.
        bidirectional: stack a reversed-time SSM and concatenate.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.h = 2 if bidirectional else 1

        # log Δ: per-channel timescale (Gu 2022 init range).
        log_dt_min, log_dt_max = math.log(0.001), math.log(0.1)
        log_dt = torch.rand(d_model) * (log_dt_max - log_dt_min) + log_dt_min
        self.log_dt = nn.Parameter(log_dt)

        # Diagonal A: complex eigenvalues. Initialise as a HiPPO-N-like
        # half-circle so spectral radius < 1.
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        A_imag = math.pi * torch.arange(d_state).repeat(d_model, 1)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        # Output mixing C and skip D.
        self.C = nn.Parameter(torch.randn(self.h, d_model, d_state, dtype=torch.cfloat) * 0.5**0.5)
        self.D = nn.Parameter(torch.zeros(d_model))

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model * self.h, d_model)

    def _build_kernel(self, L: int, direction: int = 0) -> torch.Tensor:
        """FFT convolution kernel of length L for one direction."""
        dt = self.log_dt.exp()  # (D,)
        A = -self.log_A_real.exp() + 1j * self.A_imag  # (D, N)
        # Discretise: A_bar = exp(Δ * A), B_bar = (A_bar - 1) / A
        dtA = dt.unsqueeze(-1) * A  # (D, N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device).view(1, 1, -1)
        K = K.exp() * self.C[direction].unsqueeze(-1)  # (D, N, L)
        return 2 * K.sum(dim=1).real  # (D, L)

    def _conv(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """FFT-domain depthwise conv of u (B, D, L) with kernel K (D, L)."""
        L = u.shape[-1]
        K_pad = torch.nn.functional.pad(K, (0, L))
        u_pad = torch.nn.functional.pad(u, (0, L))
        K_f = torch.fft.rfft(K_pad)
        u_f = torch.fft.rfft(u_pad, dim=-1)
        y = torch.fft.irfft(u_f * K_f.unsqueeze(0), n=2 * L)[..., :L]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)."""
        u = x.transpose(-1, -2)  # (B, D, L)
        L = u.shape[-1]
        K_fwd = self._build_kernel(L, direction=0)
        y_fwd = self._conv(u, K_fwd)
        if self.bidirectional:
            K_bwd = self._build_kernel(L, direction=1)
            y_bwd = self._conv(u.flip(-1), K_bwd).flip(-1)
            y = torch.cat([y_fwd, y_bwd], dim=1)
            D_eff = torch.cat([self.D, self.D], dim=0)
            u_eff = torch.cat([u, u.flip(-1)], dim=1)
        else:
            y = y_fwd
            D_eff = self.D
            u_eff = u
        y = y + D_eff.unsqueeze(-1) * u_eff
        y = self.dropout(y)
        y = y.transpose(-1, -2)  # (B, L, D*h)
        return self.out_proj(y)


__all__ = ["S4DBlock"]
