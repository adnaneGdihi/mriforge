"""1-D Navigator Encoder for IB-VF / InfoNCE training (Phase 2).

The IB-VF strategy treats the DC k-space navigator :math:`m(t) = y(0, t)`
as a sufficient statistic for motion: the central-line of k-space is
the spatial integral of the image, so any rigid motion shifts it
predictably.

This module is the encoder :math:`f_\\xi: m(t) \\mapsto z_m(t)` that
extracts a :math:`d`-dimensional latent per readout. The strategy then
runs InfoNCE between :math:`z_m` and :math:`\\hat\\theta(t)` (the
estimated motion) under an IB regulariser that pushes :math:`z_m`
toward minimal sufficiency.

Input layout: complex 1-D tensor of shape ``(B, 2, T)`` where the
channel dim is real / imag (interleaved). Output: ``(B, d)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@register_model(
    name="nav_encoder_1d",
    training_mode="ib_vf",
    spatial_dims=(1,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=False,
    supports_contrast_conditioning=False,
)
class NavEncoder1D(nn.Module, IGenerator):
    """1-D temporal CNN over the DC k-space navigator m(t) = y(0, t).

    Args:
        in_channels: 2 (real, imag interleaved).
        nav_latent_dim: ``d`` in the IB-VF design.
        depth: number of conv blocks (each stride-2 downsample).
        base_channels: width of the first conv layer.
    """

    def __init__(
        self,
        in_channels: int = 2,
        nav_latent_dim: int = 64,
        depth: int = 4,
        base_channels: int = 32,
        out_channels: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels or nav_latent_dim
        self.nav_latent_dim = nav_latent_dim
        self.depth = depth
        layers: list[nn.Module] = []
        c_in = in_channels
        c_out = base_channels
        for _ in range(depth):
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out))
            layers.append(nn.GELU())
            c_in = c_out
            c_out = min(c_out * 2, 256)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, nav_latent_dim)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "nav_encoder_1d"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.nav_latent_dim)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Encode a navigator sequence.

        Args:
            x: ``(B, 2, T)`` real-imag interleaved navigator, or a
                complex ``(B, T)`` tensor that we'll split.

        Returns:
            Latent ``(B, nav_latent_dim)`` after global average pool +
            linear head + L2-normalisation (so the InfoNCE critic acts
            on unit-norm features).
        """
        if x.is_complex():
            x = torch.stack([x.real, x.imag], dim=1)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # add channel dim
        if x.shape[1] != self._in_channels:
            raise ValueError(
                f"NavEncoder1D expected channel dim {self._in_channels}, "
                f"got shape {tuple(x.shape)}."
            )
        h = self.backbone(x)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        z = self.head(h)
        return F.normalize(z, dim=-1)

    def generate(self, z: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.forward(z, **kwargs)


__all__ = ["NavEncoder1D"]
