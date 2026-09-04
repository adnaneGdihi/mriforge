"""Complex-valued Building Blocks.
=============================

Reusable blocks for complex-valued MRI reconstruction networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.conditioning import PhysicsInformedConditioning
from spectramr.models.layers.complex_conv import ComplexConv2d


class ComplexResBlock(nn.Module):
    """Residual Block with Complex Convolutions and Physics Conditioning.

    Structure:
    Input -> Norm -> Act -> ComplexConv -> Cond -> Norm -> Act -> Dropout -> ComplexConv -> Cond + Input
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: Total input channels (Real + Imag). MUST BE EVEN.
            out_channels: Total output channels (Real + Imag). MUST BE EVEN.
            emb_dim: Dimension of physics embedding.
            dropout: Dropout probability.
        """
        super().__init__()

        # ComplexConv2d expects complex channel counts (half of real channels)
        if in_channels % 2 != 0 or out_channels % 2 != 0:
            raise ValueError(
                f"ComplexResBlock requires even channels. Got in={in_channels}, out={out_channels}"
            )

        c_in = in_channels // 2
        c_out = out_channels // 2

        # GroupNorm requires num_channels divisible by num_groups; pick the
        # largest 2^k ≤ 8 that divides each channel count so very thin
        # complex blocks (e.g. c=4 ⇒ groups=4) don't crash at construction.
        def _group_count(c: int) -> int:
            for g in (8, 4, 2, 1):
                if c % g == 0:
                    return g
            return 1

        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = ComplexConv2d(c_in, c_out, 3, padding=1)
        self.cond1 = PhysicsInformedConditioning(emb_dim, in_channels)
        self.act = nn.SiLU()

        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = ComplexConv2d(c_out, c_out, 3, padding=1)
        self.cond2 = PhysicsInformedConditioning(emb_dim, out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            # Shortcut needs to adjust channels
            self.shortcut = ComplexConv2d(c_in, c_out, 1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # [ARCHITECT FIX] Apply FiLM AFTER normalization, BEFORE activation
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ComplexResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.norm1(x)
        h = self.cond1(h, emb)  # AdaGN / FiLM
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.cond2(h, emb)  # AdaGN / FiLM
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if isinstance(self.shortcut, nn.Identity):
            return h + x
        else:
            return h + self.shortcut(x)


class ComplexNAFBlock(nn.Module):
    """NAFNet Block with Complex Convolutions."""

    def __init__(
        self,
        c: int,
        DW_Expand: int = 2,
        drop_out_rate: float = 0.0,
        time_embedding_dim: int | None = None,
    ):
        """__init__.

        Args:
            c (int): Description.
            DW_Expand (int): Description.
            drop_out_rate (float): Description.
            time_embedding_dim (int | None): Description.
        """
        super().__init__()

        # c is total stacked channels (e.g. 64) -> 32 complex channels
        if c % 2 != 0:
            raise ValueError("ComplexNAFBlock requires even channels.")

        c_complex = c // 2
        dw_channel_complex = c_complex * DW_Expand
        dw_channel = c * DW_Expand

        # Time projection
        self.time_proj = None
        if time_embedding_dim is not None:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embedding_dim, c),  # Projects to C (stacked) for addition
            )

        # Depthwise Convolution (Complex)
        # Groups = in_channels for depthwise
        # ComplexConv groups logic: weight is [C_out, C_in/G, K, K].
        # If G=C_in, then weight is [C_out, 1, K, K].

        self.conv1 = ComplexConv2d(c_complex, dw_channel_complex, 1, padding=0)

        self.conv2 = ComplexConv2d(
            dw_channel_complex,
            dw_channel_complex,
            3,
            padding=1,
            groups=dw_channel_complex,  # Depthwise
        )

        self.conv3 = ComplexConv2d(dw_channel_complex // 2, c_complex, 1, padding=0)

        # Simplified Channel Attention (Real-valued on magnitude or stacked features?)
        # Let's use real-valued operations on the stacked representation for mixing
        self.sca_avg = nn.AdaptiveAvgPool2d(1)
        # 1x1 conv on stacked features (Real-valued mixing is fine/good here)
        self.sca_conv = nn.Conv2d(dw_channel, dw_channel, 1)
        self.sca_sig = nn.Sigmoid()

        # SimpleGate activation (Elementwise)
        # Implementation: x1 * x2.
        # For complex stacked: just split and multiply.

        self.conv4 = ComplexConv2d(c_complex, c_complex, 1, padding=0)
        self.conv5 = ComplexConv2d(c_complex // 2, c_complex, 1, padding=0)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # Beta/Gamma for layer scale
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """forward.

        Args:
            inp (torch.Tensor): Description.
            emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ComplexNAFBlock.

        Executes PyTorch tensor operations.

        Args:
            inp (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = inp

        if self.time_proj is not None and emb is not None:
            t = self.time_proj(emb)
            x = x + t[..., None, None]

        # Conv1 (1x1)
        x = self.conv1(x)

        # Conv2 (3x3 DW)
        x = self.conv2(x)

        # SCA
        # We treat x as real stacked features for SCA gating
        attn = self.sca_avg(x)
        attn = self.sca_conv(attn)
        attn = self.sca_sig(attn)
        x = x * attn

        # SimpleGate
        # Split channels: 2*C -> C, C
        # Complex logic: ???
        # NAFNet SimpleGate: x * y
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2

        # Conv3 (1x1)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN Part
        x = self.conv4(y)
        # SimpleGate
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2

        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
