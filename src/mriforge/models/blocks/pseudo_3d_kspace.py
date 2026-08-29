"""Pseudo-3D K-Space Convolution Block.

This module implements a 3D convolution block that strictly adheres to the
Convolution Theorem for k-space data. It uses a (kernel_depth, 1, 1) kernel
to prevent cross-talk between spatial frequencies, avoiding FOV shading artifacts.
"""

import torch
import torch.nn as nn


class Pseudo3DKSpaceBlock(nn.Module):
    """Pseudo-3D Convolution Block for K-Space.

    Applies 3D convolution across the depth dimension (e.g., slices or coils)
    while strictly maintaining 1x1 spatial kernels to preserve the Convolution Theorem.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_depth: int = 3,
        activation: bool = True,
        norm: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_depth (int): Description.
            activation (bool): Description.
            norm (bool): Description.
        """
        super().__init__()

        # Padding only along the depth dimension
        padding_depth = kernel_depth // 2

        # STRICTLY (kernel_depth, 1, 1) to prevent spatial frequency mixing
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_depth, 1, 1),
            padding=(padding_depth, 0, 0),
            bias=not norm,
        )

        self.norm = nn.InstanceNorm3d(out_channels) if norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if activation else nn.Identity()
        self.use_checkpoint = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, D, H, W] tensor

        Returns:
            [B, C_out, D, H, W] tensor

        forward method for Pseudo3DKSpaceBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if torch.is_complex(x):
            # Process real and imaginary parts separately
            if getattr(self, "use_checkpoint", False) and self.training:
                conv_real = torch.utils.checkpoint.checkpoint(
                    self.conv, x.real, use_reentrant=False
                )
                conv_imag = torch.utils.checkpoint.checkpoint(
                    self.conv, x.imag, use_reentrant=False
                )
            else:
                conv_real = self.conv(x.real)
                conv_imag = self.conv(x.imag)
            real = self.act(self.norm(conv_real))
            imag = self.act(self.norm(conv_imag))
            return torch.complex(real, imag)
        else:
            if getattr(self, "use_checkpoint", False) and self.training:
                conv_out = torch.utils.checkpoint.checkpoint(self.conv, x, use_reentrant=False)
            else:
                conv_out = self.conv(x)
            return self.act(self.norm(conv_out))


class RepetitionFusion(nn.Module):
    """Fuse multiple repetitions via permutation-invariant attention.

    This module learns to combine noise-corrupted repetitions of identical
    anatomy into high-SNR estimates using an attention mechanism over the reps.
    """

    def __init__(self, channels: int, num_reps: int = 18, num_layers: int = 2):
        """__init__.

        Args:
            channels (int): Description.
            num_reps (int): Description.
            num_layers (int): Description.
        """
        super().__init__()

        # We keep num_reps for compatibility but it's not strictly rigid in forward
        self.num_reps = num_reps

        # Stack of 3D conv blocks (kernel (3, 1, 1))
        self.fusion_blocks = nn.ModuleList(
            [
                Pseudo3DKSpaceBlock(channels, channels, kernel_depth=3, activation=True, norm=True)
                for _ in range(num_layers)
            ]
        )

        # Permutation-invariant attention scorer.
        # Uses 1x1x1 convolutions over the channel dimension to score each repetition independently.
        self.rep_attention_scorer = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # [B, C, Reps, 1, 1]
            nn.Conv3d(channels, max(1, channels // 2), kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(max(1, channels // 2), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse reps via 3D processing and attention weighting.

        Args:
            x: [B, C, H, W, Reps] (TorchIO layout maps Reps to Depth at the end)

        Returns:
            fused: [B, C, H, W]

        forward method for RepetitionFusion.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Permute to [B, C, Reps, H, W] for sequence processing
        x_permuted = x.permute(0, 1, 4, 2, 3)

        # Process through shared-weight pseudo-3D blocks
        for block in self.fusion_blocks:
            x_permuted = x_permuted + block(x_permuted)  # Residual connection

        # Score repetitions using magnitude if complex
        score_input = x_permuted.abs() if torch.is_complex(x_permuted) else x_permuted

        # [B, 1, Reps, 1, 1]
        attention_logits = self.rep_attention_scorer(score_input)
        attention_weights = torch.softmax(attention_logits, dim=2)

        # Collapse the repetition dimension identically, regardless of sequence order
        # x_permuted: [B, C, Reps, H, W]
        # attention_weights: [B, 1, Reps, 1, 1]
        x_fused = torch.sum(x_permuted * attention_weights, dim=2)

        # Returns [B, C, H, W]
        # By collapsing Reps, we seamlessly bridge the 5D -> 4D gap for the UNet
        return x_fused
