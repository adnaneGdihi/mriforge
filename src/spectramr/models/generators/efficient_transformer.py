"""Efficient Transformer Generator for MRI Super-Resolution.

This module implements memory-efficient transformer architectures with:
- Gradient checkpointing for reduced memory usage
- Low-Rank Adapters (LoRA) for parameter-efficient fine-tuning
- Mixed precision training support
- Linear attention mechanisms for faster computation
- Memory-efficient self-attention with chunked processing
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.lora import LoRAAdapter
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


# Removed duplicate LoRALayer class definition.
# Using spectramr.models.lora.LoRAAdapter instead.


class LinearAttention(nn.Module):
    """Linear attention mechanism for efficient computation."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            qkv_bias (bool): Description.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Feature map for linear attention
        self.feature_map = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for LinearAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply feature map for linear attention
        q = self.feature_map(q.reshape(B * self.num_heads, N, self.head_dim))
        k = self.feature_map(k.reshape(B * self.num_heads, N, self.head_dim))

        q = q.reshape(B, self.num_heads, N, self.head_dim)
        k = k.reshape(B, self.num_heads, N, self.head_dim)
        v = v.reshape(B, self.num_heads, N, self.head_dim)

        # Linear attention computation
        # q: [B, H, N, D], k: [B, H, N, D], v: [B, H, N, D]
        q = q / math.sqrt(self.head_dim)

        # Compute attention with linear complexity
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        attn = torch.einsum("bhnd,bhde->bhne", q, kv)

        # Reshape and project
        attn = attn.reshape(B, N, C)
        x = self.proj(attn)

        return x


class ChunkedSelfAttention(nn.Module):
    """Memory-efficient self-attention with chunked processing."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        chunk_size: int = 1024,
        qkv_bias: bool = False,
    ):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            chunk_size (int): Description.
            qkv_bias (bool): Description.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ChunkedSelfAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Process in chunks to save memory
        output_chunks = []
        for i in range(0, N, self.chunk_size):
            end_idx = min(i + self.chunk_size, N)
            q_chunk = q[:, :, i:end_idx]

            # Compute attention for this chunk
            attn_chunk = torch.einsum("bhnd,bhmd->bhnm", q_chunk, k) * self.scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)

            # Apply attention to values
            out_chunk = torch.einsum("bhnm,bhmd->bhnd", attn_chunk, v)
            output_chunks.append(out_chunk)

        # Concatenate chunks
        x = torch.cat(output_chunks, dim=2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class EfficientTransformerBlock(nn.Module):
    """Memory-efficient transformer block with gradient checkpointing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_lora: bool = True,
        use_linear_attn: bool = False,
        chunk_size: int = 1024,
        qkv_bias: bool = False,
    ):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            use_lora (bool): Description.
            use_linear_attn (bool): Description.
            chunk_size (int): Description.
            qkv_bias (bool): Description.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Choose attention mechanism
        if use_linear_attn:
            self.attn = LinearAttention(dim, num_heads, qkv_bias)
        else:
            self.attn = ChunkedSelfAttention(dim, num_heads, chunk_size, qkv_bias)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

        # LoRA adapters
        self.use_lora = use_lora
        if use_lora:
            self.lora_attn = LoRAAdapter(dim, dim)
            self.lora_mlp = LoRAAdapter(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention with residual connection
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for EfficientTransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        attn_out = self.attn(self.norm1(x))

        if self.use_lora:
            attn_out = attn_out + self.lora_attn(x)

        x = x + attn_out

        # MLP with residual connection
        mlp_out = self.mlp(self.norm2(x))

        if self.use_lora:
            mlp_out = mlp_out + self.lora_mlp(x)

        x = x + mlp_out

        return x


@register_model(name="efficient_transformer", training_mode="reconstruction")
class EfficientTransformerGenerator(nn.Module):
    """Memory-efficient transformer generator for super-resolution."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_lora: bool = True,
        use_linear_attn: bool = False,
        chunk_size: int = 1024,
        patch_size: int = 16,
        upscale_factor: int = 4,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            use_lora (bool): Description.
            use_linear_attn (bool): Description.
            chunk_size (int): Description.
            patch_size (int): Description.
            upscale_factor (int): Description.
        """
        super().__init__()
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Position embedding
        self.num_patches = (256 // patch_size) ** 2  # Assuming 256x256 input
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                EfficientTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_lora=use_lora,
                    use_linear_attn=use_linear_attn,
                    chunk_size=chunk_size,
                )
                for _ in range(depth)
            ],
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Reconstruction head
        self.reconstruction = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim,
                embed_dim // 2,
                kernel_size=upscale_factor,
                stride=upscale_factor,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, out_channels, 3, padding=1),
        )

        # Weight initialization handled by centralized weight_initialization module
        pass

    def initialize_weights(self):
        """Weight initialization handled by centralized weight_initialization module."""
        pass

    def _init_weights(self, m):
        """Weight initialization handled by centralized weight_initialization module."""
        pass

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for EfficientTransformerGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Ensure position embedding matches the actual number of patches
        num_patches = x.shape[1]
        if self.pos_embed.shape[1] != num_patches:
            # Resize position embedding if needed
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim).to(x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks with gradient checkpointing
        for block in self.blocks:
            if self.training:
                # Use gradient checkpointing during training to save memory
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        x = self.norm(x)

        # Reshape for reconstruction
        x = x.transpose(1, 2).view(
            B,
            self.embed_dim,
            H // self.patch_size,
            W // self.patch_size,
        )

        # Reconstruction
        x = self.reconstruction(x)

        return x


class EfficientTransformerGeneratorWrapper(nn.Module, IGenerator):
    """Efficient transformer generator implementing IGenerator interface."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        use_lora: bool = True,
        use_linear_attn: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            use_lora (bool): Description.
            use_linear_attn (bool): Description.
        """
        super().__init__()
        self.generator = EfficientTransformerGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            use_lora=use_lora,
            use_linear_attn=use_linear_attn,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for EfficientTransformerGeneratorWrapper.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.generator(x)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate super-resolved image."""
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        batch_size, channels, height, width = input_shape
        # Assume 4x upsampling
        return (batch_size, channels, height * 4, width * 4)

    @property
    def name(self) -> str:
        """Returns the model name."""
        features = []
        if self.generator.use_lora:
            features.append("lora")
        if self.generator.use_linear_attn:
            features.append("linear_attn")
        feature_str = "_".join(features) if features else "standard"
        return f"EfficientTransformerGenerator_{feature_str}_depth{self.generator.depth}"

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_efficient_transformer_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    embed_dim: int = 768,
    depth: int = 12,
    use_lora: bool = True,
    use_linear_attn: bool = False,
    **kwargs,
) -> EfficientTransformerGeneratorWrapper:
    """Factory function for efficient transformer generator."""
    return EfficientTransformerGeneratorWrapper(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        depth=depth,
        use_lora=use_lora,
        use_linear_attn=use_linear_attn,
        **kwargs,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the efficient transformer generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create generator with memory-efficient settings
    generator = create_efficient_transformer_generator(
        in_channels=1,
        out_channels=1,
        embed_dim=384,  # Smaller for memory efficiency
        depth=6,  # Fewer layers
        use_lora=True,
        use_linear_attn=True,
    ).to(device)

    # Test input
    batch_size = 2
    lr_size = 64
    lr_input = torch.randn(batch_size, 1, lr_size, lr_size).to(device)

    # Generate
    with torch.no_grad():
        output = generator.generate(lr_input)

    logger.debug(f"Input shape: {lr_input.shape}")
    logger.debug(f"Output shape: {output.shape}")
    logger.debug(f"Parameter count: {generator.get_parameter_count():,}")

    # Test memory usage during training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        generator.train()

        # Forward pass with gradient computation
        output = generator(lr_input)
        loss = output.mean()
        loss.backward()

        memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
        logger.debug(".1f")
