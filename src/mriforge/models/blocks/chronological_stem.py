"""Chronological K-Space Linearizer for Motion-Tracking SSMs.

Converts k-space data into a temporally-ordered 1D sequence for Mamba,
where the sequence index t maps to the physical passage of time during
the MRI scan. This transforms Mamba's causal ODE into a kinematic tracker:
h_t = A h_{t-1} + B x_t tracks continuous patient motion over time.

For Cartesian acquisitions, phase-encode (Ky) lines are sorted by their
acquisition index. For non-Cartesian (radial/spiral), lines are sorted
by absolute shot timestamps from the ISMRMRD/H5 headers.

**Domain Rule:** Never apply spatial curves (Morton/Hilbert) to k-space.
K-space is a temporal frequency-domain signal where chronological order
encodes the physics of motion. Scrambling it spatially destroys the
exact continuity needed for motion estimation.

Reference:
    Cordero-Grande et al. (2018). "Three-dimensional motion corrected
    sensitivity encoding." Magnetic Resonance in Medicine.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class ChronologicalKSpaceStem(nn.Module):
    """Temporally-ordered k-space tokenizer for HyperMamba bridge.

    Sorts k-space readout lines by acquisition time, then projects
    each line into a token embedding. The resulting sequence is
    strictly chronological, enabling Mamba's state to track motion
    as a causal time-series.

    Supports:
    - **Cartesian:** PE lines sorted by index (0, 1, 2, ..., Ky-1)
    - **Non-Cartesian:** Shots sorted by timestamp from scanner headers

    Args:
        in_channels: Number of coil/complex channels (e.g., 2 for real/imag).
        embed_dim: Token embedding dimension.
        num_readout_samples: Expected number of frequency samples per line.
        mode: Sorting mode — ``"cartesian"`` or ``"timestamp"``.
    """

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 128,
        num_readout_samples: int = 256,
        mode: str = "cartesian",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mode = mode

        # Project each readout line [C, Kx] → [D] token
        self.line_proj = nn.Linear(in_channels * num_readout_samples, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        logger.info(
            "[ChronologicalKSpaceStem] Initialized: in_ch=%d, embed=%d, "
            "readout_samples=%d, mode=%s",
            in_channels,
            embed_dim,
            num_readout_samples,
            mode,
        )

    def forward(
        self,
        kspace: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert k-space to chronologically-ordered token sequence.

        Args:
            kspace: K-space data [B, C, Ky, Kx].
            timestamps: Optional acquisition timestamps [B, Ky].
                Required when ``mode="timestamp"``. For Cartesian mode,
                lines are sorted by PE index (identity permutation).

        Returns:
            Chronological token sequence [B, Ky, D].

        forward method for ChronologicalKSpaceStem.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timestamps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, Ky, Kx = kspace.shape

        if self.mode == "timestamp" and timestamps is not None:
            # Sort PE lines by absolute acquisition time
            sorted_idx = torch.argsort(timestamps, dim=1)  # [B, Ky]
            # Expand for gathering across C and Kx dims
            idx_expanded = sorted_idx[:, None, :, None].expand_as(kspace)
            kspace = torch.gather(kspace, dim=2, index=idx_expanded)

        # Flatten each readout line: [B, Ky, C * Kx]
        lines = rearrange(kspace, "b c ky kx -> b ky (c kx)")

        # Project to token space
        tokens = self.line_proj(lines)  # [B, Ky, D]
        tokens = self.norm(tokens)

        return tokens


class ChronologicalKSpaceUnstem(nn.Module):
    """Reverse projection from token sequence back to k-space shape.

    Args:
        embed_dim: Token embedding dimension.
        out_channels: Number of output channels.
        num_readout_samples: Frequency samples per line.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        out_channels: int = 2,
        num_readout_samples: int = 256,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.num_readout_samples = num_readout_samples
        self.line_unproj = nn.Linear(embed_dim, out_channels * num_readout_samples)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Unproject tokens back to k-space layout.

        Args:
            tokens: Token sequence [B, Ky, D].

        Returns:
            K-space [B, C, Ky, Kx].

        forward method for ChronologicalKSpaceUnstem.

        Executes PyTorch tensor operations.

        Args:
            tokens (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        lines = self.line_unproj(tokens)  # [B, Ky, C * Kx]
        return rearrange(
            lines,
            "b ky (c kx) -> b c ky kx",
            c=self.out_channels,
            kx=self.num_readout_samples,
        )
