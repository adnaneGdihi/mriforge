"""Adapter and conditioning blocks for shape/embedding alignment.

These small, reusable modules help standardize how models adapt
channel counts, spatial sizes, and inject conditioning such as
time embeddings or context vectors. Keep these lightweight and
dependency-free so they can be used across generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class ChannelProjector2D(nn.Module):
    """Project 2D feature map channels via 1x1 conv."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move conv to input device and dtype if needed
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ChannelProjector2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if (
            x.device != next(self.proj.parameters()).device
            or x.dtype != next(self.proj.parameters()).dtype
        ):
            self.proj.to(x.device, x.dtype)
        return self.proj(x)


class ChannelProjector3D(nn.Module):
    """Project 3D feature map channels via 1x1x1 conv."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bias (bool): Description.
        """
        super().__init__()  # pyright: ignore[reportGeneralTypeIssues]
        self.proj = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move conv to input device and dtype if needed
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ChannelProjector3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if (
            x.device != next(self.proj.parameters()).device
            or x.dtype != next(self.proj.parameters()).dtype
        ):
            self.proj.to(x.device, x.dtype)
        return self.proj(x)


class SpatialResizer2D(nn.Module):
    """Resize 2D feature maps to a target spatial size using interpolation."""

    def __init__(
        self,
        size: tuple[int, int],
        mode: str = "bilinear",
        align_corners: bool = False,
    ):
        """__init__.

        Args:
            size (tuple[int, int]): Description.
            mode (str): Description.
            align_corners (bool): Description.
        """
        super().__init__()  # pyright: ignore[reportGeneralTypeIssues]
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SpatialResizer2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return cast(
            torch.Tensor,
            F.interpolate(
                x,
                size=self.size,
                mode=self.mode,
                align_corners=self.align_corners,
            ),
        )


class SpatialResizer3D(nn.Module):
    """Resize 3D feature maps to a target size using
    trilinear interpolation.
    """

    def __init__(
        self,
        size: tuple[int, int, int],
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        """__init__.

        Args:
            size (tuple[int, int, int]): Description.
            mode (str): Description.
            align_corners (bool): Description.
        """
        super().__init__()  # pyright: ignore[reportGeneralTypeIssues]
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SpatialResizer3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return cast(
            torch.Tensor,
            F.interpolate(
                x,
                size=self.size,
                mode=self.mode,
                align_corners=self.align_corners,
            ),
        )


class DecoderAdapter3D(nn.Module):
    """Project latent vectors to coarse 3D grids then resize to target size.

    This adapter is useful when a decoder expects spatial latents but
    upstream components only provide latent vectors. It first uses a linear
    projection to generate a coarse grid and then (optionally) upsamples to
    the desired resolution via :class:`SpatialResizer3D`.

    Args:
        latent_dim: Dimension of incoming latent vector.
        out_channels: Channel count for the generated feature map.
        coarse_resolution: Spatial shape produced directly from the latent
            projection.
        target_resolution: Final desired spatial resolution. If this matches
            ``coarse_resolution`` the resize step is skipped.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        coarse_resolution: tuple[int, int, int],
        target_resolution: tuple[int, int, int],
    ) -> None:
        """__init__.

        Args:
            latent_dim (int): Description.
            out_channels (int): Description.
            coarse_resolution (tuple[int, int, int]): Description.
            target_resolution (tuple[int, int, int]): Description.
        """
        super().__init__()  # pyright: ignore[reportGeneralTypeIssues]
        d0, h0, w0 = coarse_resolution
        self.out_channels = out_channels
        self.coarse_resolution = coarse_resolution
        self.target_resolution = target_resolution
        self.linear = nn.Linear(latent_dim, out_channels * d0 * h0 * w0)
        self.resizer: SpatialResizer3D | None = None
        if coarse_resolution != target_resolution:
            self.resizer = SpatialResizer3D(target_resolution)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DecoderAdapter3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch, _ = x.shape
        coarse = self.linear(x).view(
            batch,
            self.out_channels,
            *self.coarse_resolution,
        )
        if self.resizer is not None:
            coarse = self.resizer(coarse)
        return coarse


@dataclass(frozen=True)
class LatentGridSpec:
    """Specification of a latent grid given a target resolution and strides."""

    target_3d: tuple[int, int, int]
    num_downsamples: int = 4  # by default, 4 layers with stride=2 each

    def compute_latent_grid(self) -> tuple[int, int, int]:
        """compute_latent_grid.

        Returns:
            tuple[int, int, int]: Description.
        """
        d, h, w = self.target_3d
        scale = 2**self.num_downsamples
        # ceil division to ensure decoder can reach/exceed target then resize
        latent_d = max(1, (d + scale - 1) // scale)
        latent_h = max(1, (h + scale - 1) // scale)
        latent_w = max(1, (w + scale - 1) // scale)
        return (latent_d, latent_h, latent_w)


class TimeFiLM2D(nn.Module):
    """FiLM-style time conditioning for 2D feature maps.

    Given a time embedding vector t_emb of shape [B, t_dim], predict
    per-channel affine parameters (gamma, beta) and apply:
        y = gamma * x + beta

    Args:
        feat_channels: number of channels in x
        time_emb_dim: dimension of time embedding vector
        hidden_dim: hidden dimension in conditioning MLP
    """

    def __init__(
        self,
        feat_channels: int,
        time_emb_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        """__init__.

        Args:
            feat_channels (int): Description.
            time_emb_dim (int): Description.
            hidden_dim (int | None): Description.
        """
        super().__init__()
        hidden = hidden_dim or max(feat_channels * 2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_channels * 2),  # gamma and beta
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TimeFiLM2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b, c, *_ = x.shape
        gb = self.mlp(t_emb)  # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)  # [B, C] each
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        return gamma * x + beta


class TimeFiLM3D(nn.Module):
    """FiLM-style time conditioning for 3D feature maps (B, C, D, H, W)."""

    def __init__(
        self,
        feat_channels: int,
        time_emb_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        """__init__.

        Args:
            feat_channels (int): Description.
            time_emb_dim (int): Description.
            hidden_dim (int | None): Description.
        """
        super().__init__()  # pyright: ignore[reportGeneralTypeIssues]
        hidden = hidden_dim or max(feat_channels * 2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_channels * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TimeFiLM3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b, c, *_ = x.shape
        gb = self.mlp(t_emb)  # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)  # [B, C]
        gamma = gamma.view(b, c, 1, 1, 1)
        beta = beta.view(b, c, 1, 1, 1)
        return gamma * x + beta


class ContextProjector3D(nn.Module):
    """Project a context embedding vector to a 3D spatial tensor and
    optionally resize.

    Args:
        context_dim: input context vector dimension [B, context_dim]
        out_channels: desired channel count in the spatial map
    """

    def __init__(self, context_dim: int, out_channels: int):
        """__init__.

        Args:
            context_dim (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.linear = nn.Linear(context_dim, out_channels)

    def forward(
        self,
        context: torch.Tensor,
        spatial_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """forward.

        Args:
            context (torch.Tensor): Description.
            spatial_size (tuple[int, int, int]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ContextProjector3D.

        Executes PyTorch tensor operations.

        Args:
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            spatial_size (tuple[int, int, int]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b = context.shape[0]
        c = self.linear(context)  # [B, out_channels]
        c = c.view(b, -1, 1, 1, 1)
        return c.expand(-1, -1, *spatial_size)


class ContextProjector2D(nn.Module):
    """Project a context embedding vector to a 2D spatial tensor and
    optionally resize.

    Args:
        context_dim: input context vector dimension [B, context_dim]
        out_channels: desired channel count in the spatial map
    """

    def __init__(self, context_dim: int, out_channels: int):
        """__init__.

        Args:
            context_dim (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.linear = nn.Linear(context_dim, out_channels)

    def forward(
        self,
        context: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        """forward.

        Args:
            context (torch.Tensor): Description.
            spatial_size (tuple[int, int]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ContextProjector2D.

        Executes PyTorch tensor operations.

        Args:
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            spatial_size (tuple[int, int]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b = context.shape[0]
        c = self.linear(context)  # [B, out_channels]
        c = c.view(b, -1, 1, 1)
        return c.expand(-1, -1, *spatial_size)


class LatentReshape2D(nn.Module):
    """Reshape [N, L, C] or flat latent to 2D spatial grids with padding/truncation.

    Useful for converting transformer outputs or latent vectors to spatial
    feature maps that Conv2d blocks can process.

    Args:
        latent_dim: Expected latent dimension (L or flattened size)
        out_channels: Desired channel count in output spatial map
        target_resolution: Desired (H, W) output resolution
        mode: How to handle size mismatches ('pad', 'truncate', 'error')
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        target_resolution: tuple[int, int],
        mode: str = "pad",
    ):
        """__init__.

        Args:
            latent_dim (int): Description.
            out_channels (int): Description.
            target_resolution (tuple[int, int]): Description.
            mode (str): Description.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.target_resolution = target_resolution
        self.mode = mode
        self.target_pixels = out_channels * target_resolution[0] * target_resolution[1]

        if mode not in ("pad", "truncate", "error"):
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input formats
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for LatentReshape2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() == 2:  # [N, L] - assume flattened latent
            batch, latent_size = x.shape
            if latent_size != self.latent_dim:
                if self.mode == "error":
                    raise ValueError(f"Latent size {latent_size} != expected {self.latent_dim}")
                elif self.mode == "pad" and latent_size < self.latent_dim:
                    padding = torch.zeros(
                        batch,
                        self.latent_dim - latent_size,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    x = torch.cat([x, padding], dim=1)
                elif self.mode == "truncate" and latent_size > self.latent_dim:
                    x = x[:, : self.latent_dim]
        elif x.dim() == 3:  # [N, L, C] - sequence format
            batch, seq_len, channels = x.shape
            expected_seq = self.latent_dim // channels if channels > 0 else self.latent_dim
            if seq_len != expected_seq:
                if self.mode == "error":
                    raise ValueError(f"Sequence length {seq_len} != expected {expected_seq}")
                elif self.mode == "pad" and seq_len < expected_seq:
                    padding_shape = (batch, expected_seq - seq_len, channels)
                    padding = torch.zeros(*padding_shape, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, padding], dim=1)
                    import warnings

                    warnings.warn(
                        f"Padded latent from {seq_len} to {expected_seq} tokens",
                        UserWarning,
                        stacklevel=2,
                    )
                elif self.mode == "truncate" and seq_len > expected_seq:
                    x = x[:, :expected_seq]
                    import warnings

                    warnings.warn(
                        f"Truncated latent from {seq_len} to {expected_seq} tokens",
                        UserWarning,
                        stacklevel=2,
                    )
            # Flatten to [N, L*C]
            x = x.view(batch, -1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # Reshape to spatial
        batch = x.shape[0]
        h, w = self.target_resolution
        return x.view(batch, self.out_channels, h, w)


class LatentReshape3D(nn.Module):
    """Reshape [N, L, C] or flat latent to 3D spatial grids with padding/truncation.

    Useful for converting transformer outputs or latent vectors to spatial
    feature maps that Conv3d blocks can process.

    Args:
        latent_dim: Expected latent dimension (L or flattened size)
        out_channels: Desired channel count in output spatial map
        target_resolution: Desired (D, H, W) output resolution
        mode: How to handle size mismatches ('pad', 'truncate', 'error')
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        target_resolution: tuple[int, int, int],
        mode: str = "pad",
    ):
        """__init__.

        Args:
            latent_dim (int): Description.
            out_channels (int): Description.
            target_resolution (tuple[int, int, int]): Description.
            mode (str): Description.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.target_resolution = target_resolution
        self.mode = mode
        self.target_pixels = (
            out_channels * target_resolution[0] * target_resolution[1] * target_resolution[2]
        )

        if mode not in ("pad", "truncate", "error"):
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input formats
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for LatentReshape3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() == 2:  # [N, L] - assume flattened latent
            batch, latent_size = x.shape
            if latent_size != self.latent_dim:
                if self.mode == "error":
                    raise ValueError(f"Latent size {latent_size} != expected {self.latent_dim}")
                elif self.mode == "pad" and latent_size < self.latent_dim:
                    padding = torch.zeros(
                        batch,
                        self.latent_dim - latent_size,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    x = torch.cat([x, padding], dim=1)
                elif self.mode == "truncate" and latent_size > self.latent_dim:
                    x = x[:, : self.latent_dim]
        elif x.dim() == 3:  # [N, L, C] - sequence format
            batch, seq_len, channels = x.shape
            expected_seq = self.latent_dim // channels if channels > 0 else self.latent_dim
            if seq_len != expected_seq:
                if self.mode == "error":
                    raise ValueError(f"Sequence length {seq_len} != expected {expected_seq}")
                elif self.mode == "pad" and seq_len < expected_seq:
                    padding_shape = (batch, expected_seq - seq_len, channels)
                    padding = torch.zeros(*padding_shape, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, padding], dim=1)
                    import warnings

                    warnings.warn(
                        f"Padded latent from {seq_len} to {expected_seq} tokens",
                        UserWarning,
                        stacklevel=2,
                    )
                elif self.mode == "truncate" and seq_len > expected_seq:
                    x = x[:, :expected_seq]
                    import warnings

                    warnings.warn(
                        f"Truncated latent from {seq_len} to {expected_seq} tokens",
                        UserWarning,
                        stacklevel=2,
                    )
            # Flatten to [N, L*C]
            x = x.view(batch, -1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # Reshape to spatial
        batch = x.shape[0]
        d, h, w = self.target_resolution
        return x.view(batch, self.out_channels, d, h, w)
