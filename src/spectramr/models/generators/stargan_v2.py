"""StarGAN v2 model components adapted to single-channel MRI (Task 8).

StarGAN v2 (Choi et al., "StarGAN v2: Diverse Image Synthesis for Multiple
Domains", CVPR 2020) performs multi-domain image translation. For MRIxFields
Task 3 (any-to-any FIELD translation) the domains are the five discrete field
levels ``{0.1, 1.5, 3, 5, 7} T``. This module provides the three from-scratch
networks:

* :class:`StarGANv2Generator` — an encoder / AdaIN-decoder that translates an
  image ``x`` conditioned on a style vector ``s``. Style is injected through
  Adaptive Instance Normalisation in the bottleneck + decoder residual blocks,
  so the output genuinely depends on ``s`` (an AdaIN that ignores ``s`` would be
  a pitfall-#16 facade). The generator is domain-agnostic: it consumes a style
  vector, not a domain label — the domain lives in whatever produced ``s``.
* :class:`MappingNetwork` — a shared MLP with per-domain output heads mapping a
  latent code ``z`` and a domain label ``y`` to a style vector.
* :class:`StyleEncoder` — a shared convolutional trunk with per-domain heads
  mapping a reference image ``x`` and a domain label ``y`` to a style vector.

Only :class:`StarGANv2Generator` is registered via ``@register_model`` (needed
for YAML ``model_type: stargan_v2_generator``). :class:`MappingNetwork` and
:class:`StyleEncoder` are plain importable classes: their ``forward(*, y)``
signatures are not the generator ``forward`` contract the model factory / audit
ladder expect, so the Task-10 training strategy imports them directly rather
than resolving them through the model registry.

Adaptations from the canonical CVPR-2020 architecture:

* ``img_channels=1`` (magnitude MRI) instead of RGB.
* Modest channel counts (``base_channels=64``, capped at ``256``) so a 64x64
  forward is fast enough for unit tests — this is a baseline, not the
  256-resolution behemoth of the original.
* The style encoder pools with :class:`~torch.nn.AdaptiveAvgPool2d` instead of a
  fixed-kernel collapse, making it resolution-agnostic.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.blocks.normalization import AdaptiveInstanceNorm
from spectramr.models.registry import register_model


def _select_domain(per_domain: Tensor, y: Tensor, num_domains: int) -> Tensor:
    """Select, per sample, the ``y``-th of ``num_domains`` head outputs.

    Args:
        per_domain: Stacked head outputs, shape ``[B, num_domains, style_dim]``.
        y: Long tensor of domain indices, shape ``[B]``.
        num_domains: Number of registered domains (heads).

    Returns:
        Tensor of shape ``[B, style_dim]`` — the selected style per sample.

    Raises:
        ValueError: If any index in ``y`` is outside ``[0, num_domains)``.
            Silent clamping is forbidden (pitfall #9) — an unknown FIELD-level
            label must fail loudly rather than degrade to a default head.
    """
    if y.ndim != 1 or y.shape[0] != per_domain.shape[0]:
        raise ValueError(
            f"Domain label y must be a 1-D tensor of length "
            f"{per_domain.shape[0]} (batch), got shape {tuple(y.shape)}."
        )
    # Bounds check is intentional and loud.  int(y.min()) and int(y.max()) each
    # trigger a host sync (GPU→CPU scalar transfer) on every forward call —
    # i.e. twice per MappingNetwork/StyleEncoder call in the training loop.
    # With only num_domains=5 field levels this is negligible overhead for this
    # baseline; the clarity of the explicit raise (pitfall #9) outweighs the sync
    # cost.  Do NOT replace with IndexError-based control flow to avoid the sync.
    y_min = int(y.min())
    y_max = int(y.max())
    if y_min < 0 or y_max >= num_domains:
        raise ValueError(
            f"Domain index out of range: y has values in [{y_min}, {y_max}] "
            f"but only {num_domains} domains are registered "
            f"(valid range [0, {num_domains - 1}]). Refusing to clamp (pitfall #9)."
        )
    idx = torch.arange(per_domain.shape[0], device=per_domain.device)
    return per_domain[idx, y]


class ResBlk(nn.Module):
    """Pre-activation residual block with instance norm (no style).

    Used by the generator encoder and the style-encoder trunk. Optionally
    halves the spatial resolution via average pooling. Follows the canonical
    StarGAN v2 ``ResBlk`` (scaled residual sum for unit-variance preservation).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        downsample: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.act = nn.LeakyReLU(0.2)
        self.norm1 = nn.InstanceNorm2d(dim_in, affine=True) if normalize else nn.Identity()
        self.norm2 = nn.InstanceNorm2d(dim_in, affine=True) if normalize else nn.Identity()
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x: Tensor) -> Tensor:
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Return the scaled residual sum ``(shortcut + residual) / sqrt(2)``."""
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class AdainResBlk(nn.Module):
    """Residual block whose normalisation is style-modulated via AdaIN.

    Reuses :class:`~spectramr.models.blocks.normalization.AdaptiveInstanceNorm`
    (the canonical AdaIN block) so the style vector ``s`` scales/shifts the
    normalised feature map — this is what makes the generator output depend on
    ``s``. Optionally doubles the spatial resolution via nearest upsampling.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int,
        *,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.act = nn.LeakyReLU(0.2)
        self.norm1 = AdaptiveInstanceNorm(dim_in, style_dim)
        self.norm2 = AdaptiveInstanceNorm(dim_out, style_dim)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: Tensor, s: Tensor) -> Tensor:
        x = self.act(self.norm1(x, s))
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.act(self.norm2(x, s))
        x = self.conv2(x)
        return x

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        """Return the scaled residual sum with style-modulated normalisation."""
        return (self._shortcut(x) + self._residual(x, s)) / math.sqrt(2)


@register_model(
    name="stargan_v2_generator",
    training_mode="stargan_v2",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    requires_paired_data=False,
)
class StarGANv2Generator(nn.Module):
    """StarGAN v2 generator: image + style vector -> translated image.

    Architecture (single-channel, base_channels=64, capped at 256):

    * ``from_rgb``: 1 -> 64-channel stem.
    * Encoder: two downsampling :class:`ResBlk` (instance norm, no style).
    * Bottleneck: two :class:`AdainResBlk` (style-modulated, no resize).
    * Decoder: two upsampling :class:`AdainResBlk` back to the input channels.
    * ``to_rgb``: instance-norm + activation + 1x1 conv to ``img_channels``.

    The spatial shape of the output equals that of the input. The output is a
    function of BOTH ``x`` and the style ``s`` — swapping ``s`` yields a
    different image (verified in the unit tests).
    """

    def __init__(
        self,
        img_channels: int = 1,
        style_dim: int = 64,
        base_channels: int = 64,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        self.img_channels = img_channels
        self.style_dim = style_dim

        c1 = base_channels
        c2 = min(base_channels * 2, max_channels)
        c3 = min(base_channels * 4, max_channels)

        self.from_rgb = nn.Conv2d(img_channels, c1, 3, 1, 1)

        # Encoder: two downsampling residual blocks (no style).
        self.encode = nn.ModuleList(
            [
                ResBlk(c1, c2, downsample=True),
                ResBlk(c2, c3, downsample=True),
            ]
        )

        # Bottleneck (no resize) + decoder (upsampling), all AdaIN-modulated.
        self.decode = nn.ModuleList(
            [
                AdainResBlk(c3, c3, style_dim, upsample=False),
                AdainResBlk(c3, c3, style_dim, upsample=False),
                AdainResBlk(c3, c2, style_dim, upsample=True),
                AdainResBlk(c2, c1, style_dim, upsample=True),
            ]
        )

        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(c1, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, img_channels, 1, 1, 0),
        )

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        """Translate ``x`` under style ``s``.

        Args:
            x: Input image, shape ``[B, img_channels, H, W]``.
            s: Style vector, shape ``[B, style_dim]``.

        Returns:
            Translated image, shape ``[B, img_channels, H, W]`` (same spatial
            size as ``x``).
        """
        h = self.from_rgb(x)
        for block in self.encode:
            h = block(h)
        for block in self.decode:
            h = block(h, s)
        return self.to_rgb(h)


class MappingNetwork(nn.Module):
    """Latent-to-style mapping with per-domain output heads.

    A shared MLP embeds the latent code ``z``; ``num_domains`` unshared heads
    then produce a candidate style each, and the ``y``-th head is selected per
    sample. This is the StarGAN v2 mapping network adapted to a modest hidden
    width.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        style_dim: int = 64,
        num_domains: int = 5,
        hidden_dim: int = 256,
        num_shared_layers: int = 3,
        num_head_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.style_dim = style_dim

        shared: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_shared_layers - 1)):
            shared += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.shared = nn.Sequential(*shared)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            head: list[nn.Module] = []
            for _ in range(max(0, num_head_layers - 1)):
                head += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            head += [nn.Linear(hidden_dim, style_dim)]
            self.unshared.append(nn.Sequential(*head))

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        """Map latent ``z`` + domain label ``y`` to a style vector.

        Args:
            z: Latent code, shape ``[B, latent_dim]``.
            y: Long tensor of domain indices, shape ``[B]``.

        Returns:
            Style vector, shape ``[B, style_dim]``.
        """
        h = self.shared(z)
        per_domain = torch.stack([head(h) for head in self.unshared], dim=1)
        return _select_domain(per_domain, y, self.num_domains)


class StyleEncoder(nn.Module):
    """Reference-image-to-style encoder with per-domain output heads.

    A shared convolutional trunk (downsampling :class:`ResBlk`) followed by
    adaptive average pooling produces a global feature; ``num_domains`` unshared
    linear heads then produce a candidate style each, and the ``y``-th head is
    selected per sample. Pooling makes the encoder resolution-agnostic.
    """

    def __init__(
        self,
        img_channels: int = 1,
        style_dim: int = 64,
        num_domains: int = 5,
        base_channels: int = 64,
        max_channels: int = 256,
        num_downsamples: int = 3,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.style_dim = style_dim

        self.from_rgb = nn.Conv2d(img_channels, base_channels, 3, 1, 1)
        blocks: list[nn.Module] = []
        dim_in = base_channels
        for _ in range(num_downsamples):
            dim_out = min(dim_in * 2, max_channels)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        self.blocks = nn.ModuleList(blocks)

        self.act = nn.LeakyReLU(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.unshared = nn.ModuleList([nn.Linear(dim_in, style_dim) for _ in range(num_domains)])

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Encode reference ``x`` + domain label ``y`` into a style vector.

        Args:
            x: Reference image, shape ``[B, img_channels, H, W]``.
            y: Long tensor of domain indices, shape ``[B]``.

        Returns:
            Style vector, shape ``[B, style_dim]``.
        """
        h = self.from_rgb(x)
        for block in self.blocks:
            h = block(h)
        h = self.act(h)
        h = self.pool(h).flatten(1)
        per_domain = torch.stack([head(h) for head in self.unshared], dim=1)
        return _select_domain(per_domain, y, self.num_domains)
