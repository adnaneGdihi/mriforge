r"""Universal INR generator with contrast / thickness / site conditioning.

Implements idea §10c of the multi-contrast brainstorm — a single SIREN
implicit-neural-field generator parameterised by the *full* acquisition
context :math:`(\mathbf{r}, \mathbf{e}_c, \mathbf{e}_t, \mathbf{e}_s)`:

- :math:`\mathbf{r}` — normalised pixel coordinate in :math:`[-1, 1]^2`.
- :math:`\mathbf{e}_c` — contrast embedding from
  :class:`~mriforge.models.blocks.acquisition_conditioning.AcquisitionEmbedding`,
  driven by the per-sample physics vector :math:`(TE, TR, TI, \alpha, B_0)`.
- :math:`\mathbf{e}_t` — slice-thickness scalar (mm), expanded via the same
  sinusoidal positional encoding used inside ``AcquisitionEmbedding``.
- :math:`\mathbf{e}_s` — optional site embedding, indexed by an integer
  ``site_id``; learnable lookup table.

Once trained over a heterogeneous cohort, *super-resolution*,
*harmonisation* across vendors, and *contrast synthesis* all become
forward queries with different ``acquisition_params`` / ``thickness`` /
``site_id`` — no retraining needed.

The design choice is to *concatenate* the conditioning into the SIREN's
input vector rather than inject FiLM modulation between hidden layers.
This is the canonical NeRF-style approach [12] and avoids restructuring
:class:`~mriforge.models.generators.siren_generator.SIRENGenerator`. The
augmented input is ``[coords (2), image_features (C), e_contrast (D_c),
e_thickness (D_t), e_site (D_s)]`` — the first SIRENlayer's
:math:`\sin(\omega_0 W \mathbf{x})` projection redistributes the
embedding across all hidden features.

Channel layout
--------------
- ``in_channels``  : input image channels (typically 1).
- ``out_channels`` : output image channels (typically 1).
- ``contrast_dim`` : output dim of ``AcquisitionEmbedding``.
- ``thickness_freqs`` : number of sinusoidal frequency bands for the
  thickness scalar.
- ``site_dim`` / ``n_sites`` : optional site embedding (set ``n_sites=0``
  to disable).

References
----------
[1]  E. M. Haacke *et al.*, *Magnetic Resonance Imaging: Physical
     Principles and Sequence Design*, Wiley, 2014.
[12] B. Mildenhall *et al.*, "NeRF: Representing scenes as neural
     radiance fields for view synthesis," ECCV 2020.
[13] V. Sitzmann *et al.*, "Implicit neural representations with
     periodic activation functions (SIREN)," NeurIPS 2020.
[17] E. Perez *et al.*, "FiLM: Visual reasoning with a general
     conditioning layer," AAAI 2018.

§6 / §10c of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mriforge.models.blocks.acquisition_conditioning import (
    AcquisitionEmbedding,
    sinusoidal_positional_encode,
)
from mriforge.models.generators.siren_generator import SirenLayer
from mriforge.models.registry import register_model

__all__ = ["SIRENUniversalGenerator"]


@register_model(
    name="siren_universal",
    training_mode="reconstruction",
    supports_contrast_conditioning=True,
)
class SIRENUniversalGenerator(nn.Module):
    r"""SIREN INR conditioned on contrast, thickness, and site.

    Args:
        in_channels: Image channels (set to 0 for pure-coordinate mode).
        out_channels: Output channels.
        hidden_features: SIREN hidden width.
        hidden_layers: Number of SIREN hidden layers.
        first_omega_0: Frequency for the first SIREN layer.
        hidden_omega_0: Frequency for hidden SIREN layers.
        contrast_dim: Output dimension of ``AcquisitionEmbedding``.
        n_contrasts: Cohort cardinality forwarded to ``AcquisitionEmbedding``.
        thickness_freqs: Frequency bands for the slice-thickness encoding.
        site_dim: Site-embedding dimension. Set 0 to disable.
        n_sites: Number of sites. Required if ``site_dim > 0``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        contrast_dim: int = 16,
        n_contrasts: int = 4,
        thickness_freqs: int = 4,
        site_dim: int = 0,
        n_sites: int = 0,
    ) -> None:
        super().__init__()
        if site_dim > 0 and n_sites <= 0:
            raise ValueError("site_dim > 0 requires n_sites >= 1.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.contrast_dim = contrast_dim
        self.thickness_freqs = thickness_freqs
        self.site_dim = site_dim

        self.contrast_embedding = AcquisitionEmbedding(
            embed_dim=contrast_dim,
            n_freqs=6,
            n_contrasts=n_contrasts,
        )

        if site_dim > 0:
            self.site_embedding = nn.Embedding(n_sites, site_dim)
        else:
            self.site_embedding = None

        thickness_dim = 2 * thickness_freqs  # sin + cos pairs
        in_features = 2 + in_channels + contrast_dim + thickness_dim + site_dim

        layers: list[nn.Module] = [
            SirenLayer(in_features, hidden_features, omega_0=first_omega_0, is_first=True)
        ]
        for _ in range(hidden_layers):
            layers.append(SirenLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*layers)

        self.final_linear = nn.Linear(hidden_features, out_channels)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            self.final_linear.weight.uniform_(-bound, bound)

    @staticmethod
    def _make_coords(h: int, w: int, device: torch.device) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        xs = torch.linspace(-1.0, 1.0, w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1)  # (H, W, 2)

    def _encode_thickness(self, thickness: torch.Tensor) -> torch.Tensor:
        # thickness: (B,) in mm. Encode via sinusoidal positional encoding
        # spanning the typical clinical band [0.5, 8.0] mm.
        # log_min/max chosen so the smallest period is ~0.5 mm and the
        # largest is ~16 mm, comfortably covering the band.
        return sinusoidal_positional_encode(
            thickness.unsqueeze(-1),
            n_freqs=self.thickness_freqs,
            log_min=-1.0,
            log_max=1.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        acquisition_params: dict[str, torch.Tensor] | None = None,
        contrast_idx: torch.Tensor | None = None,
        thickness: torch.Tensor | None = None,
        site_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"SIRENUniversalGenerator expects (B, C, H, W) input; got {tuple(x.shape)}."
            )
        B, C, H, W = x.shape
        if self.in_channels != C:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {C}.")

        # ----- conditioning vector (B, D_total) -----
        cond_parts: list[torch.Tensor] = []

        if acquisition_params is None:
            # Synthesise a zero-acquisition embedding (sentinel for inference
            # without metadata). Use TE=0 etc. + contrast_idx 0.
            zero_params = {
                k: torch.zeros(B, device=x.device, dtype=x.dtype)
                for k in ("TE", "TR", "TI", "FA", "B0")
            }
            c_idx = (
                contrast_idx
                if contrast_idx is not None
                else torch.zeros(B, dtype=torch.long, device=x.device)
            )
            cond_parts.append(self.contrast_embedding(zero_params, contrast_id=c_idx))
        else:
            params = {k: v.to(x.device).to(x.dtype) for k, v in acquisition_params.items()}
            c_idx = (
                contrast_idx
                if contrast_idx is not None
                else torch.zeros(B, dtype=torch.long, device=x.device)
            )
            cond_parts.append(self.contrast_embedding(params, contrast_id=c_idx))

        if thickness is None:
            thickness = torch.ones(B, device=x.device, dtype=x.dtype)  # default 1.0 mm
        elif thickness.dim() == 0:
            thickness = thickness.expand(B)
        cond_parts.append(self._encode_thickness(thickness.to(x.dtype).to(x.device)))

        if self.site_embedding is not None:
            if site_id is None:
                raise ValueError(
                    "SIRENUniversalGenerator was constructed with site_dim>0 "
                    "but site_id was not supplied at forward time."
                )
            cond_parts.append(self.site_embedding(site_id.long().to(x.device)))

        cond = torch.cat(cond_parts, dim=-1)  # (B, D_cond)

        # ----- per-pixel input vector -----
        coords = self._make_coords(H, W, x.device).to(x.dtype)  # (H, W, 2)
        coords = coords.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
        # cond expanded to spatial
        cond_spatial = cond.view(B, -1, 1, 1).expand(-1, -1, H, W)  # (B, D_cond, H, W)
        combined = torch.cat([coords, x, cond_spatial], dim=1)  # (B, 2+C+D, H, W)

        # Flatten spatial → run SIREN per pixel
        flat = combined.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        out = self.final_linear(self.net(flat))
        return out.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)
