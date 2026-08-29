"""PMPS Phase-1 — tissue-parameter diffusion model.

Wraps the existing :class:`TissueDiffusionMRIGenerator` UNet backbone in
a physical-range output activation (softplus on :math:`T_1, T_2, M_0`;
hard ordering :math:`T_2 < T_1` via an additive constraint), and accepts
optional (B0, site_id, vendor_id) conditioning via channel-concat.

The model is registered with ``training_mode="tissue_diffusion_pretrain"``
so the strategy factory dispatches the PMPS Phase-1 strategy to
``DiffusionTrainingStrategy`` (thin alias) and the schema audit knows the
model belongs to the new paradigm.

Output channels are fixed at 3 (T1, T2, M0). ``in_channels`` is the
noised-tissue-map input channel count plus any conditioning channels.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mriforge.models.generators.tissue_diffusion_mri import (
    TissueDiffusionMRIGenerator,
)
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@register_model(
    name="tissue_parameter_diffusion",
    training_mode="tissue_diffusion_pretrain",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
    supports_contrast_conditioning=False,
)
class TissueParameterDiffusion(nn.Module, IGenerator):
    """Diffusion model over :math:`(T_1, T_2, M_0)` tissue-parameter maps.

    The forward signature follows the framework's convention:
    ``forward(x, timesteps=None, b0=None, site_id=None, **kwargs)``. The
    conditioning vectors are broadcast to spatial maps and concatenated
    onto ``x`` before entering the backbone.

    Output is a 3-channel tensor with channels in ``[0, 1]`` (normalised
    representation). Down-stream callers map back to physical units using
    the bounds in
    :class:`mriforge.models.losses.pmps_losses.TissueParameterPhysicsConstraint`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        condition_on_b0: bool = True,
        condition_on_site: bool = False,
        n_sites: int = 0,
        site_embed_dim: int = 8,
        base_dim: int = 64,
        num_timesteps: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.condition_on_b0 = condition_on_b0
        self.condition_on_site = condition_on_site
        self.n_sites = n_sites

        n_extra_ch = 0
        if condition_on_b0:
            n_extra_ch += 1
        if condition_on_site:
            if n_sites <= 0:
                raise ValueError("condition_on_site=True requires n_sites > 0.")
            n_extra_ch += site_embed_dim
            self.site_embed = nn.Embedding(n_sites, site_embed_dim)
        else:
            self.site_embed = None

        backbone_in = in_channels + n_extra_ch
        self.backbone = TissueDiffusionMRIGenerator(
            in_channels=backbone_in,
            out_channels=out_channels,
            base_dim=base_dim,
            num_timesteps=num_timesteps,
        )

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "tissue_parameter_diffusion"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self._out_channels, *input_shape[2:])

    def _broadcast_b0(self, x: torch.Tensor, b0: torch.Tensor) -> torch.Tensor:
        # b0: (B,) → (B, 1, H, W)
        b = x.shape[0]
        return b0.view(b, 1, 1, 1).expand(b, 1, x.shape[-2], x.shape[-1]).to(x.dtype)

    def _broadcast_site(self, x: torch.Tensor, site_id: torch.Tensor) -> torch.Tensor:
        assert self.site_embed is not None
        # site_id: (B,) longs → (B, D, H, W)
        emb = self.site_embed(site_id)  # (B, D)
        b, d = emb.shape
        return emb.view(b, d, 1, 1).expand(b, d, x.shape[-2], x.shape[-1]).to(x.dtype)

    def _physical_range_activation(self, raw: torch.Tensor) -> torch.Tensor:
        """Map UNet output → normalised (T1, T2, M0) with T2 < T1 ordering.

        The first two channels are passed through a sigmoid so they lie
        in ``[0, 1]`` (normalised tissue-parameter convention). The
        third channel (M0) is also sigmoid'd. The ordering constraint
        T2 ≤ T1 is enforced via an additive softplus residual: we set
        ``T2 = T2_raw * T1_normalised`` so T2 cannot exceed T1.
        """
        out = torch.sigmoid(raw)
        t1 = out[:, 0:1]
        t2_raw = out[:, 1:2]
        m0 = out[:, 2:3]
        t2 = t2_raw * t1  # enforce T2 <= T1 in normalised space
        return torch.cat([t1, t2, m0], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        b0: torch.Tensor | None = None,
        site_id: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        cond_maps = [x]
        if self.condition_on_b0:
            if b0 is None:
                # Default to 3.0 T (clinical) if not provided.
                b0 = torch.full((x.shape[0],), 3.0, device=x.device, dtype=x.dtype)
            cond_maps.append(self._broadcast_b0(x, b0))
        if self.condition_on_site:
            if site_id is None:
                raise ValueError("condition_on_site=True but site_id was not passed to forward.")
            cond_maps.append(self._broadcast_site(x, site_id))
        h = torch.cat(cond_maps, dim=1)
        raw = self.backbone(h, timesteps=timesteps)
        if raw.shape[1] != self._out_channels:
            raise RuntimeError(
                f"Backbone output channels {raw.shape[1]} != requested "
                f"out_channels {self._out_channels}."
            )
        return self._physical_range_activation(raw)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)
