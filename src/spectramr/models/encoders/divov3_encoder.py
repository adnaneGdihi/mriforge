"""DIVOv3 Encoder implementation.

This module provides a lightweight yet expressive implementation of the
DIVOv3 encoder architecture used to extract multi-scale embeddings for
encoder/decoder style pipelines.  The implementation intentionally keeps
its computational footprint modest so that it can be re-used across
super-resolution, reconstruction, and generative tasks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from spectramr.models.interfaces import IModel, LatentAccessMixin
from spectramr.shared.utils.metaclass import ModuleABCMeta


@dataclass
class DIVOv3EncoderConfig:
    """Configuration dataclass for :class:`DIVOv3Encoder`.

    Attributes
    ----------
    in_channels:
        Number of channels in the input image/volume.
    embed_dim:
        Size of the latent embedding produced by the encoder head.
    stage_channels:
        Tuple with the output channels for every encoder stage.
    stage_depths:
        Tuple describing the number of residual blocks per stage.
    drop_path_prob:
        Probability for stochastic depth in residual blocks.
    """

    in_channels: int = 1
    embed_dim: int = 256
    stage_channels: tuple[int, ...] = (32, 64, 128, 256)
    stage_depths: tuple[int, ...] = (2, 2, 2, 2)
    drop_path_prob: float = 0.0


class StochasticDepth(nn.Module):
    """Implements DropPath/Stochastic Depth used in residual blocks."""

    def __init__(self, drop_prob: float) -> None:
        """__init__.

        Args:
            drop_prob (float): Description.
        """
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for StochasticDepth.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        binary_mask = random_tensor.floor()
        return x.div(keep_prob) * binary_mask


class DIVOv3ResidualBlock(nn.Module):
    """Depthwise separable residual block with squeeze excitation."""

    def __init__(
        self,
        channels: int,
        drop_path_prob: float = 0.0,
    ) -> None:
        """__init__.

        Args:
            channels (int): Description.
            drop_path_prob (float): Description.
        """
        super().__init__()
        self.conv_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm_dw = nn.BatchNorm2d(channels)
        self.act_dw = nn.GELU()

        self.conv_pw = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm_pw = nn.BatchNorm2d(channels)
        self.act_pw = nn.GELU()

        squeeze_channels = max(channels // 16, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze_channels, 1),
            nn.GELU(),
            nn.Conv2d(squeeze_channels, channels, 1),
            nn.Sigmoid(),
        )

        self.stochastic_depth = StochasticDepth(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DIVOv3ResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        identity = x
        out = self.conv_dw(x)
        out = self.norm_dw(out)
        out = self.act_dw(out)

        out = self.conv_pw(out)
        out = self.norm_pw(out)
        out = self.act_pw(out)
        out = out * self.se(out)

        out = self.stochastic_depth(out)
        return out + identity


class DIVOv3Encoder(LatentAccessMixin, nn.Module, IModel, metaclass=ModuleABCMeta):
    """Concrete implementation of the DIVOv3 encoder.

    The encoder progressively downsamples the input while storing the
    intermediate feature maps.  The final latent embedding is obtained by
    global average pooling followed by a linear projection to ``embed_dim``.
    """

    def __init__(
        self,
        config: DIVOv3EncoderConfig | None = None,
        **kwargs,
    ) -> None:
        """__init__.

        Args:
            config (DIVOv3EncoderConfig | None): Description.
        """
        super().__init__()
        LatentAccessMixin.__init__(self)
        if config is None:
            config = DIVOv3EncoderConfig(**kwargs)
        elif kwargs:
            raise ValueError(
                "DIVOv3Encoder received both a config object and keyword overrides.",
            )

        if len(config.stage_channels) != len(config.stage_depths):
            raise ValueError(
                "stage_channels and stage_depths must have the same length",
            )

        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.stage_channels[0],
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.stage_channels[0]),
            nn.GELU(),
        )

        stages = []
        in_channels = config.stage_channels[0]
        total_blocks = sum(config.stage_depths)
        drop_schedule = torch.linspace(
            0.0,
            config.drop_path_prob,
            total_blocks,
        )
        block_idx = 0
        for stage_channels, depth in zip(
            config.stage_channels,
            config.stage_depths,
            strict=False,
        ):
            # Downsample (except for first stage which already matches)
            if in_channels != stage_channels:
                stages.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            stage_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(stage_channels),
                        nn.GELU(),
                    ),
                )
            blocks = []
            for _ in range(depth):
                blocks.append(
                    DIVOv3ResidualBlock(
                        stage_channels,
                        drop_path_prob=float(drop_schedule[block_idx].item()),
                    ),
                )
                block_idx += 1
            stages.append(nn.Sequential(*blocks))
            in_channels = stage_channels

        self.encoder_stages = nn.ModuleList(stages)
        self.head_norm = nn.LayerNorm(config.stage_channels[-1])
        self.projection = nn.Linear(
            config.stage_channels[-1],
            config.embed_dim,
        )
        self._last_embedding: torch.Tensor | None = None
        self._last_bottleneck: torch.Tensor | None = None
        self._skip_connections: list[torch.Tensor] = []

    def iter_feature_maps(self) -> Iterable[torch.Tensor]:
        """Iterate over the stored skip connections from last forward pass."""
        return tuple(self._skip_connections)

    def get_embedding(self) -> torch.Tensor:
        """Return the latent embedding from the last forward pass."""
        if self._last_embedding is None:
            raise RuntimeError(
                "No embedding available - call forward() first.",
            )
        return self._last_embedding

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            x (torch.Tensor): Description.
            return_embedding (bool): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Description.

        forward method for DIVOv3Encoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        skips: list[torch.Tensor] = []
        out = self.stem(x)
        skips.append(out)
        for stage in self.encoder_stages:
            out = stage(out)
            skips.append(out)

        # The last element in skips is the deepest representation
        latent = skips[-1]
        # Global average pooling followed by projection
        pooled = torch.mean(latent, dim=(-2, -1))
        pooled = self.head_norm(pooled)
        embedding = self.projection(pooled)

        self._last_embedding = embedding
        self._last_bottleneck = latent
        # keep encoder features for decoder usage
        self._skip_connections = skips[:-1]
        self._set_latent_state(
            embedding=embedding,
            feature_maps=self._skip_connections,
            bottleneck=latent,
        )

        if return_embedding:
            return embedding, latent
        return embedding

    @property
    def name(self) -> str:
        """Get the model name."""
        return "divov3"

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def export_for_decoder(self) -> dict[str, torch.Tensor]:
        """Return a dictionary with cached encoder representations."""
        return {
            "embedding": self.get_embedding(),
            "skips": tuple(self._skip_connections),
            "bottleneck": self._last_bottleneck,
        }


# Backward compatibility alias
DinoV3Encoder = DIVOv3Encoder
