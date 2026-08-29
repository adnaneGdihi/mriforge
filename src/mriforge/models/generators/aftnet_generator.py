"""AFTNet: Artificial Fourier Transform Network.

Implements an efficient transformer-like attention mechanism
using Fourier decomposition instead of vanilla attention.

Key Features:
- O(N log N) complexity via FFT-based attention
- Positional encoding in frequency domain
- Compatible with existing transformer decoders

Reference:
- Zhen et al. "An Attention Free Transformer" (AFT)
- Extensions for MRI reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class AFTBlock(nn.Module):
    """Attention-Free Transformer Block.

    Replaces dot-product attention with element-wise operations
    in frequency domain for efficiency.

    Args:
        dim: Feature dimension
        hidden_dim: Hidden dimension for expansion
        num_heads: Number of attention heads (for compatibility)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_heads: int = 8,
    ):
        """__init__.

        Args:
            dim (int): Description.
            hidden_dim (Optional[int]): Description.
            num_heads (int): Description.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        hidden_dim = hidden_dim or dim * 4

        # Query, Key, Value projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Learnable position bias (in frequency domain)
        self.pos_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Output projection
        self.to_out = nn.Linear(dim, dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """AFT attention mechanism.

        Args:
            x: [B, N, D] input sequence
            context: Optional [B, M, D] context (for cross-attention)

        Returns:
            [B, N, D] output

        forward method for AFTBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, D = x.shape

        # Normalize
        x_norm = self.norm1(x)

        # Projections
        q = self.to_q(x_norm)  # [B, N, D]
        k = self.to_k(context if context is not None else x_norm)
        v = self.to_v(context if context is not None else x_norm)

        # Reshape for multi-head
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # AFT: Instead of QK^T softmax, use element-wise sigmoid gating
        # This is the key insight: replace O(N^2) attention with O(N)

        # Compute in frequency domain for global context
        k_fft = torch.fft.fft(k, dim=2, norm="ortho")
        v_fft = torch.fft.fft(v, dim=2, norm="ortho")

        # Element-wise modulation in frequency space
        kv_fft = k_fft * v_fft  # [B, heads, M, D]

        # Inverse FFT to get context
        kv = torch.fft.ifft(kv_fft, dim=2, norm="ortho").real

        # Sigmoid gating (AFT core)
        gate = torch.sigmoid(q + self.pos_bias.unsqueeze(0))

        # If context length differs from query, we need to interpolate
        M = kv.shape[2]
        if M != N:
            kv = F.interpolate(
                kv.permute(0, 1, 3, 2),  # [B, H, D, M]
                size=N,
                mode="linear",
                align_corners=False,
            ).permute(0, 1, 3, 2)  # [B, H, N, D]

        # Apply gate
        out = gate * kv

        # Reshape and project
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.to_out(out)

        # Residual
        x = x + out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class AFTEncoder2D(nn.Module):
    """2D AFT Encoder for image processing.

    Converts image to sequence, applies AFT blocks, converts back.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            dim (int): Description.
            num_layers (int): Description.
            num_heads (int): Description.
            patch_size (int): Description.
        """
        super().__init__()

        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # AFT blocks
        self.blocks = nn.ModuleList([AFTBlock(dim, num_heads=num_heads) for _ in range(num_layers)])

        # Positional encoding
        self.pos_embed = None  # Dynamically created

    def _get_pos_embed(self, N: int, device: torch.device) -> torch.Tensor:
        """Get positional embedding."""
        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            # Sinusoidal positional encoding
            pos = torch.arange(N, device=device).float()
            dim = self.dim

            pe = torch.zeros(N, dim, device=device)
            position = pos.unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2, device=device).float()
                * (-torch.log(torch.tensor(10000.0)) / dim)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.pos_embed = pe.unsqueeze(0)

        return self.pos_embed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Encode image to sequence.

        Args:
            x: [B, C, H, W] input image

        Returns:
            features: [B, N, D] encoded features
            orig_size: (H_out, W_out) for reconstruction

        forward method for AFTEncoder2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, tuple[int, int]]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Patch embed: [B, D, H/P, W/P]
        x = self.patch_embed(x)
        H_out, W_out = x.shape[2], x.shape[3]

        # Flatten to sequence: [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        N = x.shape[1]

        # Add positional encoding
        x = x + self._get_pos_embed(N, x.device)

        # AFT blocks
        for block in self.blocks:
            x = block(x)

        return x, (H_out, W_out)


@register_model(name="aftnet", training_mode="reconstruction")
class AFTNetGenerator(nn.Module, IGenerator):
    """AFT Network for MRI Reconstruction.

    Attention-free transformer that uses Fourier-based
    attention for efficient global context modeling.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        dim: Feature dimension
        num_layers: Number of AFT layers
        patch_size: Patch size for tokenization
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            dim (int): Description.
            num_layers (int): Description.
            num_heads (int): Description.
            patch_size (int): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.dim = dim

        # Encoder
        self.encoder = AFTEncoder2D(
            in_channels=in_channels,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_size=patch_size,
        )

        # Decoder (upsample back to image)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * patch_size * patch_size),
            nn.GELU(),
        )

        # Final projection
        self.final = nn.Conv2d(dim, out_channels, 1)

        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] input image

        Returns:
            [B, C_out, H, W] output image

        forward method for AFTNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Encode
        features, (H_enc, W_enc) = self.encoder(x)

        # Decode: [B, N, D] -> [B, N, D*P*P]
        decoded = self.decoder(features)

        # Reshape: [B, N, D*P*P] -> [B, D, H, W]
        decoded = decoded.view(B, H_enc, W_enc, self.dim, self.patch_size, self.patch_size)
        decoded = decoded.permute(0, 3, 1, 4, 2, 5).contiguous()
        decoded = decoded.view(B, self.dim, H_enc * self.patch_size, W_enc * self.patch_size)

        # Resize if needed
        if decoded.shape[-2:] != (H, W):
            decoded = F.interpolate(decoded, size=(H, W), mode="bilinear", align_corners=False)

        # Final projection + skip
        out = self.final(decoded) + self.skip(x)

        return out

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "AFTNetGenerator"

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        B, C, H, W = input_shape
        return (B, self.out_channels, H, W)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "AFTBlock",
    "AFTEncoder2D",
    "AFTNetGenerator",
]
