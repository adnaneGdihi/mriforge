r"""Spatially-Adaptive Feature Modulation (SPADE) Blocks.

Inject marker priors (structural maps, navigators, spatial masks) deep into
the decoder of the U-Net via spatially-adaptive normalisation, replacing
naive input-channel concatenation which is washed out by downsampling.

.. math::

    h_{\text{mod}} = \gamma(m) \odot \text{LayerNorm}(h) + \beta(m)

Where:
    - :math:`h` is the feature map at a given decoder resolution,
    - :math:`m` is the marker prior (resized to the same spatial resolution),
    - :math:`\gamma, \beta` are learned per-pixel scale and bias derived
      from :math:`m`.

Reference:
    Park et al., "Semantic Image Synthesis with Spatially-Adaptive
    Normalization [SPADE]", *CVPR* (2019).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SPADEBlock(nn.Module):
    """Spatially-adaptive feature modulation block.

    Takes a feature map *h* and a marker tensor *m* (at any resolution)
    and produces modulated features via learned γ(m) and β(m).

    Example::

        spade = SPADEBlock(feature_channels=128, marker_channels=1)
        h_mod = spade(h, marker)
    """

    def __init__(
        self,
        feature_channels: int,
        marker_channels: int,
        hidden_channels: int | None = None,
        norm_type: str = "group",
        num_groups: int = 8,
    ) -> None:
        """Initialise the SPADE block.

        Args:
            feature_channels: Number of channels in the feature map *h*.
            marker_channels: Number of channels in the marker tensor *m*.
            hidden_channels: Width of the internal marker encoder.
                Defaults to ``min(128, feature_channels)``.
            norm_type: ``'group'`` (default) or ``'instance'``.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        self.feature_channels = feature_channels
        hid = hidden_channels or min(128, feature_channels)

        # Normalisation layer (applied to feature map *h*)
        if norm_type == "instance":
            self.norm = nn.InstanceNorm2d(feature_channels, affine=False)
        else:
            self.norm = nn.GroupNorm(
                min(num_groups, feature_channels),
                feature_channels,
                affine=False,
            )

        # Lightweight marker encoder → shared features
        self.shared = nn.Sequential(
            nn.Conv2d(marker_channels, hid, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        # Gamma (scale) and Beta (bias) heads
        self.gamma_conv = nn.Conv2d(hid, feature_channels, kernel_size=3, padding=1)
        self.beta_conv = nn.Conv2d(hid, feature_channels, kernel_size=3, padding=1)

        # Initialise gamma ≈ 1, beta ≈ 0 for near-identity at start.
        # Use small random weights so different markers produce different
        # modulations from the beginning (pure-zero kills gradients).
        nn.init.normal_(self.gamma_conv.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.gamma_conv.bias)
        nn.init.normal_(self.beta_conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, h: Tensor, marker: Tensor) -> Tensor:
        """Apply SPADE modulation.

        Args:
            h: Feature map ``(B, C, H, W)``.
            marker: Marker tensor ``(B, M, H', W')``.  Will be resized
                to match *h* spatially if shapes differ.

        Returns:
            Modulated features ``(B, C, H, W)``.
        """
        # Resize marker to feature spatial resolution
        if marker.shape[-2:] != h.shape[-2:]:
            marker = F.interpolate(
                marker,
                size=h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Encode marker
        m_feat = self.shared(marker)

        # Produce γ and β
        gamma = self.gamma_conv(m_feat)  # (B, C, H, W)
        beta = self.beta_conv(m_feat)  # (B, C, H, W)

        # Modulate normalised features
        h_norm = self.norm(h)
        return gamma * h_norm + beta


class SPADEEncoder(nn.Module):
    """Multi-scale marker encoder producing SPADE parameters at each decoder level.

    Creates one :class:`SPADEBlock` per resolution scale, allowing marker
    injection at every decoder level of a U-Net.

    Example::

        encoder = SPADEEncoder(
            marker_channels=1,
            feature_channels_per_level=[256, 128, 64, 32],
        )
        modulated_features = []
        for level, h in enumerate(decoder_features):
            modulated_features.append(encoder(h, marker, level))
    """

    def __init__(
        self,
        marker_channels: int,
        feature_channels_per_level: list[int],
        hidden_channels: int | None = None,
        norm_type: str = "group",
    ) -> None:
        """Initialise multi-scale SPADE encoder.

        Args:
            marker_channels: Channels in the raw marker input.
            feature_channels_per_level: Feature channel count at each decoder
                level (from deepest to shallowest).
            hidden_channels: Internal encoder width (shared across levels).
            norm_type: Normalisation type for each SPADE block.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SPADEBlock(
                    feature_channels=ch,
                    marker_channels=marker_channels,
                    hidden_channels=hidden_channels,
                    norm_type=norm_type,
                )
                for ch in feature_channels_per_level
            ]
        )

    def forward(self, h: Tensor, marker: Tensor, level: int) -> Tensor:
        """Modulate features at a specific decoder level.

        Args:
            h: Feature map at the given *level*.
            marker: Raw marker tensor ``(B, M, H_orig, W_orig)``.
            level: Decoder level index (0 = deepest).

        Returns:
            SPADE-modulated feature map.
        """
        if level >= len(self.blocks):
            raise IndexError(f"SPADEEncoder has {len(self.blocks)} levels, got level={level}")
        return self.blocks[level](h, marker)

    @property
    def num_levels(self) -> int:
        """Number of decoder levels supported."""
        return len(self.blocks)


__all__ = ["SPADEBlock", "SPADEEncoder"]
