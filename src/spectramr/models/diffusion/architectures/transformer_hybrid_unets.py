"""Hybrid U-Net architectures combining CNNs with Transformers (Swin, ViT)."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from spectramr.models.blocks.block_registry import create_block
from spectramr.models.diffusion.diffusion_parts.swin_transformer_parts import (
    PatchMerging,
    SwinTransformerBlock,
)
from spectramr.models.diffusion.diffusion_parts.vit_parts import PatchEmbedding, ViTBlock
from spectramr.models.reconstruction.unet import SinusoidalPositionEmbeddings
from spectramr.models.registry import register_model


@register_model(name="swin_hybrid_unet", training_mode="diffusion")
class SwinHybridUNet(nn.Module):
    """Swin Transformer-based Diffusion U-Net.

    This model combines a U-Net backbone for spatial feature extraction with a
    time-conditional Swin Transformer path for score prediction. The outputs from
    both paths are fused to produce the final result. This hybrid architecture
    leverages the strengths of both CNNs for spatial hierarchies and Transformers
    for global context modeling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        time_dim: int = 256,
        cond_channels: int = 0,
        image_size: int = 256,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: list[int] = None,
        num_heads: list[int] = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        layer_norm_strategy: str = "pre",
        layer_norm_eps: float = 1e-6,
        use_enhanced_layer_norm: bool = True,
        bilinear: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_dim (int): Description.
            cond_channels (int): Description.
            image_size (int): Description.
            patch_size (int): Description.
            embed_dim (int): Description.
            depths (list[int]): Description.
            num_heads (list[int]): Description.
            window_size (int): Description.
            mlp_ratio (float): Description.
            drop_rate (float): Description.
            attn_drop_rate (float): Description.
            drop_path_rate (float): Description.
            layer_norm_strategy (str): Description.
            layer_norm_eps (float): Description.
            use_enhanced_layer_norm (bool): Description.
            bilinear (bool): Description.
        """
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        # --- Standard Parameters ---
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        # Conditioning network (similar to ScoreBasedUNet)
        self.cond_net = None
        if cond_channels > 0:
            self.cond_net = nn.Sequential(
                nn.Conv2d(cond_channels, embed_dim // 4, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, embed_dim),
        )

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        patches_resolution = [image_size // patch_size, image_size // patch_size]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        # Position embedding
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList(
                [
                    SwinTransformerBlock(
                        dim=int(embed_dim * 2**i_layer),
                        input_resolution=(
                            patches_resolution[0] // (2**i_layer),
                            patches_resolution[1] // (2**i_layer),
                        ),
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=0 if (j % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])][j],
                        norm_layer=nn.LayerNorm,
                        layer_norm_strategy=layer_norm_strategy,
                        layer_norm_eps=layer_norm_eps,
                        use_enhanced_layer_norm=use_enhanced_layer_norm,
                    )
                    for j in range(depths[i_layer])
                ],
            )

            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(
                    input_resolution=(
                        patches_resolution[0] // (2**i_layer),
                        patches_resolution[1] // (2**i_layer),
                    ),
                    dim=int(embed_dim * 2**i_layer),
                    norm_layer=nn.LayerNorm,
                )
            else:
                downsample = None

            self.layers.append(
                nn.ModuleDict({"blocks": layer, "downsample": downsample}),
            )

        # Score prediction network
        final_dim = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = nn.LayerNorm(final_dim)
        self.score_net = nn.Sequential(
            nn.Linear(final_dim, final_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(final_dim * 2, in_channels * patch_size**2),
        )

        # U-Net components for spatial refinement
        self.down_conv = create_block("unet_double_conv", in_channels=in_channels, out_channels=64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            create_block("unet_double_conv", in_channels=64, out_channels=128),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            create_block("unet_double_conv", in_channels=128, out_channels=256),
        )

        # Bridge with attention
        self.bridge = nn.Sequential(
            nn.MaxPool2d(2),
            create_block("unet_double_conv", in_channels=256, out_channels=512),
            nn.Conv2d(512, 512, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )

        # Upsampling path
        factor = 2 if bilinear else 1
        self.up1 = nn.ConvTranspose2d(512, 256 // factor, 2, stride=2)
        self.up_conv1 = create_block("unet_double_conv", in_channels=512, out_channels=256)

        self.up2 = nn.ConvTranspose2d(256, 128 // factor, 2, stride=2)
        self.up_conv2 = create_block("unet_double_conv", in_channels=256, out_channels=128)

        self.up3 = nn.ConvTranspose2d(128, 64 // factor, 2, stride=2)
        self.up_conv3 = create_block("unet_double_conv", in_channels=128, out_channels=64)

        # Output layer
        self.outc = create_block("unet_out_conv", in_channels=64, out_channels=out_channels)
        self.output_activation = (
            nn.Tanh()
        )  # Ensure output is in [-1, 1] range for models predicting the image

    def forward(self, x, time, cond=None):
        """Forward pass with score-based diffusion.

        Args:
            x: Input tensor [B, C, H, W]
            time: Time step [B]
            cond: Conditional input tensor [B, C_cond, H, W] (optional)

        Returns:
            Score estimate for denoising

        forward method for SwinHybridUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            time (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b, c, h, w = x.shape

        # Time embedding
        time_emb = self.time_embed(time)

        # Conditioning embedding
        if self.cond_net is not None and cond is not None:
            cond_emb = self.cond_net(cond)
            time_emb = time_emb + cond_emb

        # U-Net encoder path
        x1 = self.down_conv(x)  # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        bridge = self.bridge(x3)  # [B, 512, H/8, W/8]

        # Swin Transformer processing
        if h >= self.patch_size and w >= self.patch_size:
            # Resize input to match expected size if needed
            # NOTE: The MRIDataset typically resizes images to 240x240.
            # This interpolation might be redundant if `image_size` is also 240,
            # adding unnecessary computational overhead.
            if h != self.image_size or w != self.image_size:
                x_swin = F.interpolate(
                    x,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                x_swin = x

            # Patch embedding
            x_swin = self.patch_embed(
                x_swin,
            )  # [B, embed_dim, H/patch_size, W/patch_size]
            x_swin = x_swin.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            x_swin = x_swin + self.absolute_pos_embed
            x_swin = self.pos_drop(x_swin)

            # Add time embedding to each patch
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, x_swin.size(1), -1)
            x_swin = x_swin + time_emb_expanded

            # Process through Swin Transformer layers
            for layer in self.layers:
                for block in layer["blocks"]:
                    x_swin = block(x_swin)

                if layer["downsample"] is not None:
                    x_swin = layer["downsample"](x_swin)

            # Apply final normalization
            x_swin = self.norm(x_swin)

            # Predict scores
            scores = self.score_net(x_swin)  # [B, num_patches, c*patch_size^2]

            # --- STABILITY FIX: Use einops.rearrange for robust reshaping ---
            # Calculate the number of patches at the final resolution
            final_res_divisor = 2 ** (self.num_layers - 1)
            h_patches = self.patches_resolution[0] // final_res_divisor
            w_patches = self.patches_resolution[1] // final_res_divisor

            scores = rearrange(
                scores,
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                h=h_patches,
                w=w_patches,
                p1=self.patch_size,
                p2=self.patch_size,
                c=c,
            )

            # Resize scores from the Swin path back to the original input size
            if scores.shape[-2:] != (h, w):
                scores = F.interpolate(
                    scores,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            # For small images, use only U-Net
            scores = torch.zeros_like(x)

        # U-Net decoder path with skip connections
        x_up = self.up1(bridge)
        x_up = torch.cat([x_up, x3], dim=1)
        x_up = self.up_conv1(x_up)

        x_up = self.up2(x_up)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.up_conv2(x_up)

        x_up = self.up3(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.up_conv3(x_up)

        # Final output
        unet_output = self.outc(x_up)

        # Combine Swin scores with U-Net output
        final_output = scores + unet_output

        # Apply tanh activation to ensure output is in [-1, 1] range
        return self.output_activation(final_output)


@register_model(name="vit_hybrid_unet", training_mode="diffusion")
class ViTHybridUNet(nn.Module):
    """Vision Transformer-based Diffusion U-Net.

    This model combines a U-Net backbone for spatial feature extraction with a
    time-conditional Vision Transformer (ViT) path for score prediction. The outputs from
    both paths are fused to produce the final result. This hybrid architecture
    leverages the strengths of both CNNs for spatial hierarchies and Transformers
    for global context modeling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        time_dim: int = 256,
        cond_channels: int = 0,
        image_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        bilinear: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_dim (int): Description.
            cond_channels (int): Description.
            image_size (int): Description.
            patch_size (int): Description.
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            drop_rate (float): Description.
            bilinear (bool): Description.
        """
        super().__init__()

        # --- Standard Parameters ---
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Conditioning network (similar to ScoreBasedUNet)
        self.cond_net = None
        if cond_channels > 0:
            self.cond_net = nn.Sequential(
                nn.Conv2d(cond_channels, embed_dim // 4, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, embed_dim),
        )

        # Patch and position embedding
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            in_channels,
            embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ViT Blocks
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                )
                for _ in range(depth)
            ],
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Score prediction head
        # Predicts a value for each patch that can be reshaped into an image
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels * patch_size**2),
        )

        # U-Net components for spatial refinement (reusing BlockLayer
        # from registry)
        self.down_conv = create_block("unet_double_conv", in_channels=in_channels, out_channels=64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            create_block("unet_double_conv", in_channels=64, out_channels=128),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            create_block("unet_double_conv", in_channels=128, out_channels=256),
        )

        self.bridge = nn.Sequential(
            nn.MaxPool2d(2),
            create_block("unet_double_conv", in_channels=256, out_channels=512),
            nn.Conv2d(512, 512, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
        )

        factor = 2 if bilinear else 1
        self.up1 = nn.ConvTranspose2d(512, 256 // factor, 2, stride=2)
        self.up_conv1 = create_block("unet_double_conv", in_channels=512, out_channels=256)

        self.up2 = nn.ConvTranspose2d(256, 128 // factor, 2, stride=2)
        self.up_conv2 = create_block("unet_double_conv", in_channels=256, out_channels=128)

        self.up3 = nn.ConvTranspose2d(128, 64 // factor, 2, stride=2)
        self.up_conv3 = create_block("unet_double_conv", in_channels=128, out_channels=64)

        self.outc = create_block("unet_out_conv", in_channels=64, out_channels=out_channels)
        self.output_activation = nn.Tanh()

    def forward(self, x, time, cond=None):
        """Forward pass with score-based diffusion.

        Args:
            x: Input tensor [B, C, H, W]
            time: Time step [B]
            cond: Conditional input tensor [B, C_cond, H, W] (optional)

        Returns:
            Score estimate for denoising

        forward method for ViTHybridUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            time (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure input tensors are on the correct device
        x = x.to(next(self.parameters()).device)
        time = time.to(next(self.parameters()).device)
        if cond is not None:
            cond = cond.to(next(self.parameters()).device)

        b, c, h, w = x.shape

        # Time embedding
        time_emb = self.time_embed(time)

        # Conditioning embedding
        if self.cond_net is not None and cond is not None:
            cond_emb = self.cond_net(cond)
            time_emb = time_emb + cond_emb

        # U-Net encoder path
        x1 = self.down_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        bridge = self.bridge(x3)

        # ViT processing
        if h >= self.patch_size and w >= self.patch_size:
            # Resize input to match expected size if needed.
            # NOTE: The MRIDataset typically resizes images to 240x240.
            # This interpolation might be redundant if `image_size` is also
            # 240.
            x_vit = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

            # Patch embedding
            x_vit = self.patch_embed(x_vit)  # [B, num_patches, embed_dim]

            # Add CLS token and position embedding
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x_vit = torch.cat((cls_tokens, x_vit), dim=1)
            x_vit = x_vit + self.pos_embed
            x_vit = self.pos_drop(x_vit)

            # Add time embedding to each token
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, x_vit.size(1), -1)
            x_vit = x_vit + time_emb_expanded

            # Process through ViT blocks
            for blk in self.blocks:
                x_vit = blk(x_vit)

            x_vit = self.norm(x_vit)

            # Predict scores from patch tokens (excluding CLS token)
            patch_tokens = x_vit[:, 1:, :]
            # [B, num_patches, C * patch_size^2]
            scores = self.score_net(patch_tokens)

            # Reshape back to image format
            scores = rearrange(
                scores,
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                h=self.image_size // self.patch_size,
                w=self.image_size // self.patch_size,
                p1=self.patch_size,
                p2=self.patch_size,
            )

            # Resize back to original size
            if scores.shape[-2:] != (h, w):
                scores = F.interpolate(
                    scores,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            scores = torch.zeros_like(x)

        # U-Net decoder path
        x_up = self.up1(bridge)
        x_up = torch.cat([x_up, x3], dim=1)
        x_up = self.up_conv1(x_up)

        x_up = self.up2(x_up)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.up_conv2(x_up)

        x_up = self.up3(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.up_conv3(x_up)

        unet_output = self.outc(x_up)

        # Combine ViT scores with U-Net output
        final_output = scores + unet_output

        return self.output_activation(final_output)
