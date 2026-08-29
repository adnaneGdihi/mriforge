r"""Ladder VAE (``hierarchical_vae_ladder``).

Hierarchical VAE of Sønderby *et al.* [1] with the *ladder* inference
construction: each level's posterior is the precision-weighted product
of a bottom-up deterministic statistic and a top-down generative prior
(see :class:`mriforge.models.blocks.vae.ladder_block.LadderBlock`).

The generative model factorises as

.. math::

    p_\theta(x, z_{1:L}) = p_\theta(x \mid z_1)
        \prod_{\ell=1}^{L-1} p_\theta(z_\ell \mid z_{\ell+1})\, p(z_L),

and the ELBO is the reconstruction term minus the sum of per-level KLs.

Reference:
    [1] C. K. Sønderby *et al.*, "Ladder variational autoencoders,"
        NeurIPS 2016.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mriforge.models.blocks.vae.ladder_block import LadderBlock
from mriforge.models.registry import register_model


@register_model(
    name="hierarchical_vae_ladder",
    training_mode="vae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class LadderVAE(nn.Module):
    r"""Ladder VAE with precision-weighted hierarchical posteriors.

    Hyperparameters are scalars/strings so the model constructs from a
    YAML ``model_kwargs`` block with no nn.Module arguments. The encoder,
    ladder blocks, and decoder are built internally.

    Args:
        in_channels: Input image channels.
        out_channels: Reconstruction channels.
        num_layers: Number of stochastic ladder levels ``L``.
        latent_dims: Per-level latent widths, top-to-bottom; must be
            length ``num_layers`` and non-increasing. When ``None`` a
            geometric default ``(latent_dim, latent_dim/2, ...)`` is used.
        latent_dim: Bottom (largest) latent width used to derive the
            default ``latent_dims``.
        hidden: Channel width / MLP hidden width of the shared trunk.
        pooled_size: Spatial size the encoder is pooled to before the
            ladder (keeps the flatten size fixed across resolutions).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_layers: int = 3,
        latent_dims: list[int] | tuple[int, ...] | None = None,
        latent_dim: int = 64,
        hidden: int = 128,
        pooled_size: int = 4,
    ) -> None:
        super().__init__()

        if latent_dims is None:
            latent_dims = [max(1, latent_dim // (2**i)) for i in range(num_layers)]
        latent_dims = list(latent_dims)
        if len(latent_dims) != num_layers:
            raise ValueError(f"len(latent_dims)={len(latent_dims)} != num_layers={num_layers}")
        if any(latent_dims[i] < latent_dims[i + 1] for i in range(num_layers - 1)):
            raise ValueError(f"latent_dims must be non-increasing top-to-bottom; got {latent_dims}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.latent_dims = latent_dims
        self.hidden = hidden
        self.pooled_size = pooled_size
        feature_dim = hidden * pooled_size * pooled_size

        # Shared deterministic trunk: image -> (B, hidden, p, p) -> feature.
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, 4, 2, 1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))

        # Ladder levels. latent_dims[0] is the TOP level (smallest above).
        # Index 0 is topmost (above_dim=None); deeper indices condition on
        # the level above them.
        blocks: list[LadderBlock] = []
        for level in range(num_layers):
            above_dim = None if level == 0 else latent_dims[level - 1]
            blocks.append(
                LadderBlock(
                    feature_dim=feature_dim,
                    latent_dim=latent_dims[level],
                    above_dim=above_dim,
                    hidden=hidden,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Decoder: bottom latent -> image.
        bottom_dim = latent_dims[-1]
        self.dec_fc = nn.Linear(bottom_dim, feature_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden // 2, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, out_channels, 3, 1, 1),
        )

    @property
    def name(self) -> str:
        return "hierarchical_vae_ladder"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _features(self, x: Tensor) -> Tensor:
        h = self.trunk(x)
        h = self.pool(h)
        return torch.flatten(h, start_dim=1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input image ``(B, in_channels, H, W)``.

        Returns:
            A dict consumed by ``VAETrainingStrategy``: ``reconstruction``
            (``x_hat``) and ``kl_override`` (the summed, batch-mean per-level
            hierarchical KL — a non-negative scalar). The hierarchical KL
            cannot be expressed as one Gaussian (mu, logvar) pair, hence the
            override channel.
        """
        feature = self._features(x)

        # Bottom-up statistics for every level.
        stats_up = [b.bottom_up(feature) for b in self.blocks]

        # Top-down pass from topmost level (index 0) downward.
        z_above: Tensor | None = None
        kl_total = x.new_zeros(())
        for level, block in enumerate(self.blocks):
            mu_hat, logvar_hat = stats_up[level]
            mu_p, logvar_p = block.top_down(z_above)
            mu_q, logvar_q = block.merge(mu_hat, logvar_hat, mu_p, logvar_p)
            z = LadderBlock.reparameterize(mu_q, logvar_q)
            kl_total = kl_total + block.kl_two_gaussians(mu_q, logvar_q, mu_p, logvar_p).mean()
            z_above = z

        # z_above now holds the bottom latent.
        h = self.dec_fc(z_above)
        h = h.view(-1, self.hidden, self.pooled_size, self.pooled_size)
        x_hat = self.decoder(h)
        if x_hat.shape[-2:] != x.shape[-2:]:
            x_hat = torch.nn.functional.interpolate(
                x_hat, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return {"reconstruction": x_hat, "kl_override": kl_total}


__all__ = ["LadderVAE"]
