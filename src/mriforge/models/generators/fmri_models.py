"""fMRI generators for ``TODO/audit_plan_novel_fmri.md`` fMRI ideas.

Registered models:
    * ``b0_beltrami_estimator``  — fMRI §2: small UNet predicting Δ B₀ from
      a pair of opposite-phase-encode EPI volumes; output is post-processed
      via the Beltrami constraint so the implied warp is quasi-conformal.
    * ``hrf_diffusion_prior``    — fMRI §5: tiny 1-D conditional U-Net that
      denoises HRF time courses with FiLM-style time conditioning.
    * ``tangent_spd_diffusion``  — fMRI §4: MLP tangent-score network for
      Riemannian diffusion on ``Sym⁺(n)`` (operates in the flat tangent
      space, output is symmetrised).

References match the plan; full citations in
``docs/fmri_mrf_directions_2026.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mriforge.models.registry import register_model


@register_model(
    "b0_beltrami_estimator",
    training_mode="reconstruction",
    spatial_dims=(2,),
)
class B0BeltramiEstimator(nn.Module):
    """Small U-Net producing a per-voxel Δ B₀ field (Hz)."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_width: int = 16,
    ) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Conv2d(base_width, 2 * base_width, 2, stride=2)
        self.mid = nn.Sequential(
            nn.Conv2d(2 * base_width, 2 * base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * base_width, 2 * base_width, 3, padding=1),
            nn.GELU(),
        )
        self.up = nn.ConvTranspose2d(2 * base_width, base_width, 2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(2 * base_width, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        e = self.enc1(x)
        d = self.mid(self.down(e))
        u = self.up(d)
        return self.head(torch.cat([u, e], dim=1))


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard half-sin/half-cos sinusoidal positional embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args_ = t.float().unsqueeze(-1) * freqs
    return torch.cat([args_.sin(), args_.cos()], dim=-1)


@register_model(
    "hrf_diffusion_prior",
    training_mode="diffusion",
    spatial_dims=(1,),
)
class HRFDiffusionPrior(nn.Module):
    """1-D U-Net for diffusion denoising of HRF time courses."""

    def __init__(
        self,
        hrf_length: int = 32,
        width: int = 32,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.hrf_length = hrf_length
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.GELU(), nn.Linear(time_dim * 2, width)
        )
        self.lift = nn.Conv1d(1, width, 3, padding=1)
        self.mid = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2),
            nn.GELU(),
        )
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, h: torch.Tensor, t: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(1)  # [B, T] -> [B, 1, T]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = self.time_mlp(_sinusoidal_time_embedding(t, self.time_dim))
        x = self.lift(h)
        x = x + t_emb.view(-1, x.shape[1], 1)
        x = self.mid(x)
        return self.head(x)


@register_model(
    "tangent_spd_diffusion",
    training_mode="diffusion",
    spatial_dims=(2,),
)
class TangentSPDDiffusion(nn.Module):
    """MLP tangent-space score for Riemannian diffusion on Sym⁺(n).

    Operates on the upper-triangular flat representation of an
    ``n × n`` SPD matrix (dim = n·(n+1)/2). Output is reshaped back
    to ``[..., n, n]`` and symmetrised.
    """

    def __init__(
        self,
        n: int = 8,
        hidden: int = 128,
        n_layers: int = 4,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n = n
        self.flat_dim = n * (n + 1) // 2
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        layers: list[nn.Module] = [nn.Linear(self.flat_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, self.flat_dim))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _to_flat(C: torch.Tensor, n: int) -> torch.Tensor:
        idx = torch.triu_indices(n, n, device=C.device)
        return C[..., idx[0], idx[1]]

    @staticmethod
    def _to_sym(flat: torch.Tensor, n: int) -> torch.Tensor:
        idx = torch.triu_indices(n, n, device=flat.device)
        S = flat.new_zeros(*flat.shape[:-1], n, n)
        S[..., idx[0], idx[1]] = flat
        S[..., idx[1], idx[0]] = flat
        return S

    def forward(self, C: torch.Tensor, t: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        flat = self._to_flat(C, self.n)
        h = self.net[0](flat)
        h = h + self.time_mlp(_sinusoidal_time_embedding(t, self.time_dim))
        for layer in self.net[1:]:
            h = layer(h)
        return self._to_sym(h, self.n)


@register_model(
    "learned_epi_forward_operator",
    training_mode="reconstruction",
    spatial_dims=(2,),
)
class LearnedEPIForwardOperator(nn.Module):
    """Learned multiband-EPI forward operator (fMRI §1).

    The plan calls for a *learned* EPI forward operator applied to
    reconstructed BOLD volumes (since raw multiband k-space is
    essentially unavailable). This model is the trainable companion
    to the fixed :class:`MultibandEPIConfig`: a small UNet predicts a
    per-timepoint complex modulation that, combined with the standard
    multiband aliasing and Cartesian undersampling, yields the
    simulated EPI k-space.

    Forward signature: ``(volume[B, T, S, H, W]) → kspace[B, 2, T, S, H, W]``
    (real-imag interleaved). Caller composes this with the static
    aliasing physics in
    :func:`mriforge.infrastructure.physics.epi_simulator.simulated_multiband_epi`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,  # real + imag of the modulation
        base_width: int = 16,
        n_timepoints: int = 1,
    ) -> None:
        super().__init__()
        self.n_timepoints = n_timepoints
        # Per-slice ConvNet; broadcast across time via reshape.
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, 2 * base_width, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * base_width, 2 * base_width, 3, padding=1),
            nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(2 * base_width, base_width, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Accept either [B, C, H, W] (single time/slice) or
        # [B, C, T, H, W] / [B, T, S, H, W] (volumetric).
        if x.dim() == 4:
            return self.dec(self.enc(x))
        if x.dim() == 5:
            B, C_or_T, mid, H, W = x.shape
            # Treat first non-batch dim as channel-equivalent if it's small;
            # otherwise collapse (T, S) into batch.
            x_flat = x.view(B * C_or_T * mid, 1, H, W)
            out = self.dec(self.enc(x_flat))
            return out.view(B, C_or_T, mid, out.shape[1], H, W)
        raise ValueError(f"LearnedEPIForwardOperator expects 4- or 5-D, got {x.shape}")


__all__ = [
    "B0BeltramiEstimator",
    "HRFDiffusionPrior",
    "LearnedEPIForwardOperator",
    "TangentSPDDiffusion",
]
