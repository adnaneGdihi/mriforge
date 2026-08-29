import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

__all__ = ["KSpaceGPT"]


@register_model(name="kspace_gpt", training_mode="reconstruction", spatial_dims=(2,))
class KSpaceGPT(nn.Module, IGenerator):
    """
    KSpaceGPT: Generative Pre-trained Transformer for k-space (Experiment 102).

    Models k-space as a sequence of tokens (e.g., patches or lines).
    Uses a Transformer to predict missing k-space data or refine it.
    """

    def __init__(
        self,
        in_channels: int = 2,  # Real/Imag
        out_channels: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        img_size: int = 256,
        patch_size: int = 16,
        features: list[int] = None,  # Compatibility with UNet-style configs
        **kwargs,  # Absorb extra config args (causal_mask, max_seq_length, etc.)
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            d_model (int): Description.
            nhead (int): Description.
            num_layers (int): Description.
            dim_feedforward (int): Description.
            img_size (int): Description.
            patch_size (int): Description.
            features (list[int]): Description.
        """
        super().__init__()

        # [COMPATIBILITY FIX] Handle 'features' from config if d_model is default/mismatched
        if features is not None and len(features) > 0:
            # Use largest feature dimension as d_model if not explicitly set to something else
            # (Heuristic: usually the bottleneck dimension)
            if d_model == 256 and features[-1] != 256:
                d_model = features[-1]

        # [COMPATIBILITY FIX] Handle 'embed_dim' alias for 'd_model'
        if "embed_dim" in kwargs:
            d_model = kwargs["embed_dim"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Positional encoding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.ConvTranspose2d(
            d_model, out_channels, kernel_size=patch_size, stride=patch_size
        )

        # Cache for positional embeddings
        self._cached_pos_emb = None
        self._cached_shape = None

    def _get_pos_embedding(self, seq_len: int) -> torch.Tensor:
        """Get positional embedding with caching to avoid graph breaks."""
        # Zero-cost check if shape matches
        if self._cached_shape == seq_len and self._cached_pos_emb is not None:
            return self._cached_pos_emb

        # Resize logic runs ONLY on shape change
        if seq_len <= self.pos_embedding.shape[1]:
            # Slice if smaller
            self._cached_pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            # Interpolate if larger (better for spatial coherence than repeat)
            # pos_embedding is [1, N, D] -> [1, D, N] for interpolation
            pos_emb_perm = self.pos_embedding.transpose(1, 2)

            # Interpolate to new length
            pos_emb_interp = torch.nn.functional.interpolate(
                pos_emb_perm, size=seq_len, mode="linear", align_corners=False
            )

            # Permute back to [1, N, D]
            self._cached_pos_emb = pos_emb_interp.transpose(1, 2)

        self._cached_shape = seq_len
        return self._cached_pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KSpaceGPT.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Embed patches
        x = self.patch_embed(x)  # [B, d_model, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d_model]

        # Add pos embedding (O(1) lookup most of the time)
        pos_emb = self._get_pos_embedding(x.shape[1])
        x = x + pos_emb

        # Transformer
        x = self.transformer(x)

        # Reshape back
        h, w = H // self.patch_size, W // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.d_model, h, w)

        # Output projection
        x = self.output_proj(x)

        return x

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])

    def generate(self, x: torch.Tensor) -> torch.Tensor:
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
        return "kspace_gpt"
