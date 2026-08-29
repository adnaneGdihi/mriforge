"""SiMAE: Subject-identity Separation Latent Masked Autoencoder.
=============================================================

Implementation of SiMAE for disentangled representation learning on MRI.
Uses a ViT-based MAE with Adaptive Layer Normalization (AdaLN) in the decoder
to disentangle Anatomy (Spatial Tokens) from Contrast (CLS Token).

References:
- He et al. "Masked Autoencoders Are Scalable Vision Learners"
- Karras et al. "A Style-Based Generator Architecture..." (AdaIN concept)
- Peebles et al. "Scalable Diffusion Models with Transformers" (AdaLN concept)
"""

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.mae.mae_generator import Attention, DropPath, MAEEncoder
from mriforge.models.registry import register_model


class AdaLN(nn.Module):
    """Adaptive Layer Normalization.

    Regulates the mean and variance of the input x based on a style vector y.
    Used to inject contrast information into the decoder.
    """

    def __init__(self, hidden_dim: int, style_dim: int):
        """__init__.

        Args:
            hidden_dim (int): Description.
            style_dim (int): Description.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.fc = nn.Linear(style_dim, hidden_dim * 2)

        # Initialize to identity
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        style: [B, S_D]

        forward method for AdaLN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Linear projection of style
        h = self.fc(style)  # [B, 2*D]
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        # Reshape for broadcasting
        # [B, 1, D]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        return self.norm(x) * (1 + gamma) + beta


class SiMAETransformerBlock(nn.Module):
    """Transformer Block with AdaLN for style conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        style_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
    ):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            style_dim (int): Description.
            mlp_ratio (float): Description.
            qkv_bias (bool): Description.
            qk_scale (float): Description.
            drop (float): Description.
            attn_drop (float): Description.
            drop_path (float): Description.
            act_layer (nn.Module): Description.
        """
        super().__init__()
        # Norms are handled by AdaLN
        self.ada_ln1 = AdaLN(dim, style_dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ada_ln2 = AdaLN(dim, style_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Pre-Norm with AdaLN
        """forward.

        Args:
            x (torch.Tensor): Description.
            style (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SiMAETransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x + self.drop_path(self.attn(self.ada_ln1(x, style)))
        x = x + self.drop_path(self.mlp(self.ada_ln2(x, style)))
        return x


class SiMAEDecoder(nn.Module):
    """Decoder for SiMAE with AdaLN style injection."""

    def __init__(
        self,
        embed_dim: int,
        style_dim: int,  # Typically same as encoder embed_dim (CLS token)
        num_patches: int,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        """__init__.

        Args:
            embed_dim (int): Description.
            style_dim (int): Description.
            num_patches (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
        )

        self.blocks = nn.ModuleList(
            [
                SiMAETransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    style_dim=style_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        # Note: Final norm might ideally be AdaLN too, but LayerNorm is stable for pixel prediction

    def forward(
        self, x: torch.Tensor, ids_restore: torch.Tensor, style: torch.Tensor
    ) -> torch.Tensor:
        # x: [B, N_visible, D]
        # Embed
        """forward.

        Args:
            x (torch.Tensor): Description.
            ids_restore (torch.Tensor): Description.
            style (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SiMAEDecoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ids_restore (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1,
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # [B, N_patches+1, D]

        # Add Pos Embed
        x = x + self.decoder_pos_embed

        # Apply blocks with style
        for blk in self.blocks:
            x = blk(x, style)

        x = self.norm(x)
        return x


@register_model(name="simae", training_mode="ssl")
class SiMAEGenerator(nn.Module, IGenerator):
    """Subject-identity Masked Autoencoder.

    Encoder: Standard ViT (Masked)
    Decoder: AdaLN-Transformer (Style Conditioned)

    Disentanglement:
    - Encoder 'CLS' token -> Style/Identity (Contrast)
    - Encoder 'Patch' tokens -> Content/Subject (Anatomy)
    """

    @property
    def name(self) -> str:
        """Get model name."""
        return "simae"

    def __init__(
        self,
        img_size: tuple[int, int, int] = (64, 64, 64),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        decoder_depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ):
        """__init__.

        Args:
            img_size (tuple[int, int, int]): Description.
            patch_size (tuple[int, int, int]): Description.
            in_channels (int): Description.
            embed_dim (int): Description.
            encoder_depth (int): Description.
            decoder_depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            mask_ratio (float): Description.
        """
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Encoder (Standard MAE)
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Decoder (SiMAE with AdaLN)
        self.decoder = SiMAEDecoder(
            embed_dim=embed_dim,
            style_dim=embed_dim,  # We use CLS token as style
            num_patches=self.encoder.patch_embed.num_patches,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Prediction head
        self.decoder_pred = nn.Linear(
            embed_dim,
            patch_size[0] * patch_size[1] * patch_size[2] * in_channels,
            bias=True,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """_init_weights.

        Args:
            m (Any): Description.
        Returns:
            Any: Description.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if m.weight is not None:
                nn.init.ones_(m.weight)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random masking to patches."""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        # Create a dummy MAEGenerator to use its methods or reimplement
        # Ideally we refactor MAEGenerator to have static methods or utility class
        # For now, duplicate logic or use the one from MAEGenerator if it was static.
        # It's an instance method there. I'll re-implement for safety.
        """patchify.

        Args:
            imgs (Any): Description.
        Returns:
            Any: Description.
        """
        p = self.patch_size[0]
        h, w, d = self.img_size
        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, w, d))
        x = x.reshape(shape=(x.shape[0], self.in_channels, h // p, p, w // p, p, d // p, p))
        x = torch.einsum("nchpwqdr->nhwpqrdc", x)
        x = x.reshape(
            shape=(
                x.shape[0],
                (h // p) * (w // p) * (d // p),
                self.in_channels * p * p * p,
            )
        )
        return x

    def unpatchify(self, x):
        """unpatchify.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        p = self.patch_size[0]
        h, w, d = self.img_size
        c = self.in_channels
        x = x.reshape(shape=(x.shape[0], h // p, w // p, d // p, p, p, p, c))
        x = torch.einsum("nhwpqrdc->nchpwqdr", x)
        return x.reshape(shape=(x.shape[0], c, h, w, d))

    def forward_encoder(self, x, mask_ratio):
        # 1. Embed
        """forward_encoder.

        Args:
            x (Any): Description.
            mask_ratio (Any): Description.
        Returns:
            Any: Description.
        """
        x = self.encoder.patch_embed(x)

        # 2. Add Pos Embed
        # Skip CLS token in pos_embed as it's not added yet
        pos_embed = self.encoder.encoder.pos_embed[:, 1:, :]
        x = x + pos_embed

        # 3. Mask, but self.random_masking is an instance method, so call it normally
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)

        # 4. Append CLS
        B, N, C = x_masked.shape
        cls_token = self.encoder.encoder.cls_token.expand(B, -1, -1)
        # Add CLS pos embed
        cls_token = cls_token + self.encoder.encoder.pos_embed[:, :1, :]
        x_cat = torch.cat((cls_token, x_masked), dim=1)

        # 5. Trunk
        for blk in self.encoder.encoder.blocks:
            x_cat = blk(x_cat)
        x_cat = self.encoder.encoder.norm(x_cat)

        return x_cat, mask, ids_restore

    def forward(self, x: torch.Tensor, style_reference: torch.Tensor = None) -> torch.Tensor:
        """
        x: Input image (masked)
        style_reference: Reference image for style. If None, uses x (self-recon).

        forward method for SiMAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style_reference (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Encode Content (from x)
        latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)

        # latent: [B, N_vis+1, D]
        # CLS token is at 0
        vocab_cls = latent[:, 0, :]  # [B, D]
        content_tokens = latent

        # 2. Encode Style
        style_vec = vocab_cls  # Default to self-style

        if style_reference is not None:
            # Encode reference image (without masking? or with?)
            # Usually style is from full image
            ref_latent, _, _ = self.forward_encoder(style_reference, mask_ratio=0.0)
            style_vec = ref_latent[:, 0, :]  # CLS token of reference

        # 3. Decode
        # Pass style_vec to decoder
        decoded_params = self.decoder(content_tokens, ids_restore, style_vec)

        # Predict pixels
        pred = self.decoder_pred(decoded_params)

        # Remove CLS
        pred = pred[:, 1:, :]

        return self.unpatchify(pred)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x, **kwargs)

    def get_output_shape(self, input_shape):
        """get_output_shape.

        Args:
            input_shape (Any): Description.
        Returns:
            Any: Description.
        """
        return input_shape

    def get_parameter_count(self):
        """get_parameter_count.

        Returns:
            Any: Description.
        """
        return sum(p.numel() for p in self.parameters())
