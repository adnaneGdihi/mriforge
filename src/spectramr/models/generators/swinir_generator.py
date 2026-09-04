"""SwinIR Generator Implementation
==============================

SOLID-compliant implementation of SwinIR (Swin Transformer for
Image Restoration) generator.
Based on the paper: "SwinIR: Image Restoration Using Swin Transformer"
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.blocks.swin import SwinBlock
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for 4D inputs (B, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, **kwargs):
        super().__init__(normalized_shape, eps, elementwise_affine, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        img_size: int = 224,
        patch_norm: bool = True,
    ):
        """__init__.

        Args:
            dim (int): Description.
            input_resolution (tuple[int, int]): Description.
            depth (int): Description.
            num_heads (int): Description.
            window_size (int): Description.
            mlp_ratio (float): Description.
            qkv_bias (bool): Description.
            qk_scale (Optional[float]): Description.
            drop (float): Description.
            attn_drop (float): Description.
            drop_path (float): Description.
            norm_layer (nn.Module): Description.
            img_size (int): Description.
            patch_norm (bool): Description.
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = nn.ModuleList(
            [
                SwinBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop,
                    attn_dropout=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ],
        )

        self.conv = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for RSTB.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        res = x
        for block in self.residual_group:
            res = block(res)
        return x + self.conv(res)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        """__init__.

        Args:
            img_size (int): Description.
            patch_size (int): Description.
            in_chans (int): Description.
            embed_dim (int): Description.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for PatchEmbed.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        x = self.conv(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Image to Patch Unembedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        """__init__.

        Args:
            img_size (int): Description.
            patch_size (int): Description.
            in_chans (int): Description.
            embed_dim (int): Description.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            x_size (tuple[int, int]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for PatchUnEmbed.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_size (tuple[int, int]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


@register_model(name="swinir", training_mode="reconstruction")
class SwinIRGenerator(IGenerator, nn.Module):
    """SwinIR (Swin Transformer for Image Restoration) generator
    implementation. Based on the paper: "SwinIR: Image Restoration
    Using Swin Transformer"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        upscale: int = 2,
        window_size: int = 8,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            upscale (int): Description.
            window_size (int): Description.
            img_size (int): Description.
            patch_size (int): Description.
            in_chans (int): Description.
            embed_dim (int): Description.
            depths (tuple[int, ...]): Description.
            num_heads (tuple[int, ...]): Description.
            mlp_ratio (float): Description.
            qkv_bias (bool): Description.
            qk_scale (Optional[float]): Description.
            drop_rate (float): Description.
            attn_drop_rate (float): Description.
            drop_path_rate (float): Description.
            norm_layer (nn.Module): Description.
            patch_norm (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale = upscale
        self.window_size = window_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # absolute position embedding
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = nn.ModuleList(
                [
                    RSTB(
                        dim=int(embed_dim * 2**i_layer),
                        input_resolution=(
                            patches_resolution[0] // (2**i_layer),
                            patches_resolution[1] // (2**i_layer),
                        ),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=(
                            dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])][i]
                            if isinstance(dpr, list)
                            else dpr
                        ),
                        norm_layer=norm_layer,
                        img_size=img_size // (2**i_layer),
                        patch_norm=patch_norm,
                    )
                    for i in range(depths[i_layer])
                ],
            )

            if i_layer < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        int(embed_dim * 2**i_layer),
                        int(embed_dim * 2 ** (i_layer + 1)),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    LayerNorm2d(int(embed_dim * 2 ** (i_layer + 1))),
                )
                layer.append(downsample)

            self.layers.append(layer)

        self.norm = norm_layer(int(embed_dim * 2 ** (len(depths) - 1)))

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(
            int(embed_dim * 2 ** (len(depths) - 1)),
            embed_dim,
            3,
            1,
            1,
        )

        # upsample
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * upscale * upscale, 1, 1, 0),
            nn.PixelShuffle(upscale),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Weight initialization handled by centralized weight_initialization module
        """_init_weights.

        Args:
            m (Any): Description.
        Returns:
            Any: Description.
        """
        pass

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "SwinIR"

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Check and adjust image size to be divisible by
        window_size * patch_size.
        """
        _, _, h, w = x.size()
        mod_pad_h = (
            self.window_size * self.patch_size - h % (self.window_size * self.patch_size)
        ) % (self.window_size * self.patch_size)
        mod_pad_w = (
            self.window_size * self.patch_size - w % (self.window_size * self.patch_size)
        ) % (self.window_size * self.patch_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def _get_pos_embed(self, num_patches: int) -> torch.Tensor:
        """Get positional embeddings, resampling if necessary for variable input shapes.

        Args:
            num_patches: Number of patches in the input

        Returns:
            Positional embeddings with shape (1, num_patches, embed_dim)
        """
        if self.absolute_pos_embed.shape[1] == num_patches:
            return self.absolute_pos_embed

        # Resample pos_embed for new patch resolution
        # This handles variable input sizes gracefully
        pos_embed = self.absolute_pos_embed
        if pos_embed.shape[1] != num_patches:
            # Reshape to spatial coordinates for interpolation
            orig_h = int(pos_embed.shape[1] ** 0.5 + 0.5)
            orig_w = int(pos_embed.shape[1] ** 0.5 + 0.5)
            new_h = int(num_patches**0.5 + 0.5)
            new_w = int(num_patches**0.5 + 0.5)

            pos_embed_2d = pos_embed.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)
            pos_embed_2d = F.interpolate(
                pos_embed_2d, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            pos_embed = pos_embed_2d.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)

        return pos_embed

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the SwinIR model.

        forward method for SwinIRGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # shallow feature extraction
        x = self.patch_embed(x)  # B, H*W, C
        pos_embed = self._get_pos_embed(x.shape[1])
        x = x + pos_embed
        x = self.pos_drop(x)

        # deep feature extraction
        for i_layer, layer in enumerate(self.layers):
            has_downsample = i_layer < len(self.layers) - 1
            rstb_blocks = layer[:-1] if has_downsample else layer

            for block in rstb_blocks:
                x = block(x)

            if has_downsample:
                # Convert to spatial for downsampling
                B, HW, C = x.shape
                res_h = self.patches_resolution[0] // (2**i_layer)
                res_w = self.patches_resolution[1] // (2**i_layer)
                x = x.view(B, res_h, res_w, C)
                x = x.permute(0, 3, 1, 2)  # B, C, H, W
                x = layer[-1](x)  # downsample
                # Convert back to patch sequence
                x = x.flatten(2).transpose(1, 2)  # B, H*W, C

        x = self.norm(x)  # B, H*W, C
        res_h_last = self.patches_resolution[0] // (2 ** (len(self.layers) - 1))
        res_w_last = self.patches_resolution[1] // (2 ** (len(self.layers) - 1))

        x = self.conv_after_body(
            x.view(
                x.shape[0],
                res_h_last,
                res_w_last,
                -1,
            ).permute(0, 3, 1, 2),
        )

        # upsample
        x = self.conv_before_upsample(x)
        x = self.upsample(x)

        return x[:, :, : H * self.upscale, : W * self.upscale]

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples using the generator interface.

        This delegates directly to ``forward`` and mirrors other generator
        implementations in the codebase.
        """
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the output shape for a given input shape.

        The method accepts both (N, C, H, W) and (C, H, W) style tuples and
        returns the output shape including the batch dimension if present.
        """
        if len(input_shape) == 4:
            n, c, h, w = input_shape
            return (n, self.out_channels, h * self.upscale, w * self.upscale)
        if len(input_shape) == 3:
            c, h, w = input_shape
            return (self.out_channels, h * self.upscale, w * self.upscale)
        raise ValueError(
            "Unsupported input_shape for SwinIRGenerator: expected (N,C,H,W) or (C,H,W)"
        )

    def get_parameter_count(self) -> int:
        """Return the total number of parameters for the model."""
        return sum(p.numel() for p in self.parameters())
