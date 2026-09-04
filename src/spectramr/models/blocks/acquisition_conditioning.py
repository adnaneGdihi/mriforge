"""Acquisition-parameter conditioning embedding.

Encodes the physical acquisition vector :math:`\\theta_c=(TE, TR, TI, \\alpha, B_0)`
of a contrast :math:`c` into a learnable embedding suitable as a condition for
FiLM modulation, cross-attention, hypernetwork weight generation, or as a
side-channel input to any reconstruction model.

The intent is *not* to learn the contrast lookup table from scratch —
which is what a bare ``nn.Embedding(n_contrasts, d)`` does — but to make the
embedding a *function of the underlying physics*. Two different protocols of
the same nominal contrast (e.g. T2w at TE=80ms vs TE=120ms) thereby map to
adjacent points in latent space without per-protocol re-training, which is the
key to harmonisation across vendors and field strengths.

The encoder applies a sinusoidal positional encoding (Mildenhall et al. 2020 [12])
to each scalar — chosen over raw scaling because :math:`(TE, TR, TI)` span 4+
orders of magnitude — followed by a small MLP that produces an embedding of
configurable dimension. A *categorical* contrast id (0..n-1) can be concatenated
as an additional one-hot when the cohort has discrete contrast labels, giving
a hybrid physics+label encoding.

The output embedding is consumed by:

- ``FiLMConditioner`` (existing block in ``src/infrastructure/physics/fno_epg.py``)
  via ``γ, β = mlp(emb)``.
- ``FiLMMambaBlock`` (existing block in ``src/models/blocks/film_mamba_block.py``)
  by passing as ``contrast_embedding``.
- Hypernetworks generating decoder weights :math:`\\theta_c = H(\\mathbf{e}_c)`
  (Perez et al. 2018 [17] for FiLM equivalence; physics-informed extension here).

References
----------
[1]  E. M. Haacke, R. W. Brown, M. R. Thompson, R. Venkatesan,
     *Magnetic Resonance Imaging: Physical Principles and Sequence Design*,
     2nd ed., Wiley, 2014.
[12] B. Mildenhall *et al.*, "NeRF: Representing scenes as neural radiance
     fields for view synthesis," ECCV 2020.
[17] E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville,
     "FiLM: Visual reasoning with a general conditioning layer,"
     AAAI 2018.

This module is part of the *Pattern C* multi-contrast handling described in
``docs/multi_contrast.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["AcquisitionEmbedding", "sinusoidal_positional_encode"]


def sinusoidal_positional_encode(
    x: torch.Tensor,
    n_freqs: int = 6,
    log_min: float = -3.0,
    log_max: float = 3.0,
) -> torch.Tensor:
    """Encode each scalar in ``x`` with ``n_freqs`` log-spaced sin/cos pairs.

    The frequency band spans ``10**log_min`` to ``10**log_max`` so a single
    encoding handles TE values from microseconds (FID) to seconds (IR
    preparations). Output dimension per input scalar is ``2 * n_freqs``.

    Args:
        x: Tensor of shape ``(..., D)`` containing the scalars to encode.
        n_freqs: Number of frequency bands.
        log_min: ``log10`` of the lowest frequency.
        log_max: ``log10`` of the highest frequency.

    Returns:
        Tensor of shape ``(..., D * 2 * n_freqs)``.
    """
    freqs = torch.logspace(log_min, log_max, n_freqs, device=x.device, dtype=x.dtype)
    # x: (..., D) → (..., D, 1) * (n_freqs,) → (..., D, n_freqs)
    angles = x.unsqueeze(-1) * freqs * 2.0 * math.pi
    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    # Flatten the last two dims: (..., D * 2 * n_freqs)
    return encoded.reshape(*x.shape[:-1], -1)


class AcquisitionEmbedding(nn.Module):
    """Map a per-contrast acquisition vector to a learnable embedding.

    The input is a dict of physical scalars per sample:

    - ``TE``  — echo time (ms),
    - ``TR``  — repetition time (ms),
    - ``TI``  — inversion time (ms, optional; -1 sentinel for sequences without IR),
    - ``FA``  — flip angle (degrees),
    - ``B0``  — main field strength (Tesla).

    Optionally also a categorical ``contrast_id`` (long), which is one-hot
    embedded and concatenated. This lets a single model learn both the
    *continuous* dependence on physics (via the sinusoidal encoding) and any
    *residual* per-contrast offset that the closed-form Bloch equations don't
    capture (off-resonance, partial-volume, vendor-specific quirks).

    Args:
        embed_dim: Output embedding dimension.
        n_freqs: Number of sinusoidal frequency bands per scalar.
        n_contrasts: Cohort cardinality for the optional categorical embedding.
            Set to 0 to disable the categorical channel.
        param_keys: Names of physical parameters to encode (in this order).
            Default ``("TE", "TR", "TI", "FA", "B0")``.
        hidden_mult: Hidden-layer width multiplier in the projection MLP.

    Forward signature
    -----------------
    ``forward(params: dict[str, Tensor], contrast_id: Tensor | None) -> Tensor``

    Each value in ``params`` is a tensor of shape ``(B,)``. ``contrast_id`` is
    a long tensor of shape ``(B,)`` or ``None``. The returned embedding has
    shape ``(B, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_freqs: int = 6,
        n_contrasts: int = 0,
        param_keys: tuple[str, ...] = ("TE", "TR", "TI", "FA", "B0"),
        hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if n_freqs < 1:
            raise ValueError(f"n_freqs must be >= 1, got {n_freqs}")
        if n_contrasts < 0:
            raise ValueError(f"n_contrasts must be >= 0, got {n_contrasts}")

        self.embed_dim = embed_dim
        self.n_freqs = n_freqs
        self.n_contrasts = n_contrasts
        self.param_keys = tuple(param_keys)
        self.hidden_mult = hidden_mult

        physics_dim = len(self.param_keys) * 2 * n_freqs
        cat_dim = n_contrasts if n_contrasts > 0 else 0
        in_dim = physics_dim + cat_dim
        hidden = max(embed_dim * hidden_mult, in_dim)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )

        if n_contrasts > 0:
            # Buffer of one-hot rows, gathered by index for compactness.
            eye = torch.eye(n_contrasts)
            self.register_buffer("_eye", eye, persistent=False)

    def forward(
        self,
        params: dict[str, torch.Tensor],
        contrast_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Normalise units before encoding so the sinusoidal frequency band is
        # comparable across scalars: TE/TR/TI in seconds, FA in radians, B0 in T.
        normalised = []
        for key in self.param_keys:
            if key not in params:
                raise KeyError(
                    f"AcquisitionEmbedding expected param '{key}', got keys {list(params)}"
                )
            v = params[key].to(self.proj[0].weight.dtype)
            if key in ("TE", "TR", "TI"):
                v = v / 1000.0
            elif key == "FA":
                v = v * math.pi / 180.0
            normalised.append(v)
        # Stack into (B, D) and run through the encoding.
        stacked = torch.stack(normalised, dim=-1)
        encoded = sinusoidal_positional_encode(stacked, n_freqs=self.n_freqs)

        if self.n_contrasts > 0:
            if contrast_id is None:
                raise ValueError(
                    "AcquisitionEmbedding was constructed with n_contrasts > 0 "
                    "but contrast_id was not supplied at forward time."
                )
            one_hot = self._eye[contrast_id.long()]
            encoded = torch.cat([encoded, one_hot], dim=-1)

        return self.proj(encoded)
