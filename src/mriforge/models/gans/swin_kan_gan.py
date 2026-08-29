import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from mriforge.models.blocks.swin import WindowAttention
from mriforge.models.layers.kan.kan_convs.kans.layers import FastKANLayer
from mriforge.models.registry import register_model


def window_partition(x, window_size):
    """window_partition.

    Args:
        x (Any): Description.
        window_size (Any): Description.
    Returns:
        Any: Description.
    """
    B, H, W, C = x.shape

    # Ensure H and W are divisible by window_size by padding if necessary
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            (0, 0, 0, pad_w, 0, pad_h),
        )  # pad (left, right, top, bottom, front, back)
        H_padded, W_padded = H + pad_h, W + pad_w
    else:
        H_padded, W_padded = H, W

    x = x.view(
        B,
        H_padded // window_size,
        window_size,
        W_padded // window_size,
        window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (H, W, pad_h, pad_w)


def window_reverse(windows, window_size, H, W, padding_info=None):
    """window_reverse.

    Args:
        windows (Any): Description.
        window_size (Any): Description.
        H (Any): Description.
        W (Any): Description.
        padding_info (Any): Description.
    Returns:
        Any: Description.
    """
    if padding_info is not None:
        H_orig, W_orig, pad_h, pad_w = padding_info
    else:
        H_orig, W_orig, pad_h, pad_w = H, W, 0, 0

    # Use padded dimensions for reconstruction
    H_padded = H_orig + pad_h
    W_padded = W_orig + pad_w

    B = int(windows.shape[0] / (H_padded * W_padded / window_size / window_size))
    x = windows.view(
        B,
        H_padded // window_size,
        W_padded // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)

    # Remove padding if it was added
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H_orig, :W_orig, :]

    return x


class FeedForwardKAN(nn.Module):
    """FeedForwardKAN class."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        """__init__.

        Args:
            dim (Any): Description.
            hidden_dim (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            FastKANLayer(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            FastKANLayer(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for FeedForwardKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


class SwinTransformerKANBlock(nn.Module):
    """SwinTransformerKANBlock class."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        """__init__.

        Args:
            dim (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            shift_size (Any): Description.
            mlp_ratio (Any): Description.
            qkv_bias (Any): Description.
            qk_scale (Any): Description.
            drop (Any): Description.
            attn_drop (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForwardKAN(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)

    def forward(self, x, H, W):
        """forward.

        Args:
            x (Any): Description.
            H (Any): Description.
            W (Any): Description.
        Returns:
            Any: Description.

        forward method for SwinTransformerKANBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            H (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            W (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        x_windows, x_padding_info = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)

        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows, mask_padding_info = window_partition(
                img_mask,
                self.window_size,
            )
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
                attn_mask == 0,
                0.0,
            )
        else:
            attn_mask = None

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, x_padding_info)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """PatchMerging class."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        """__init__.

        Args:
            dim (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """forward.

        Args:
            x (Any): Description.
            H (Any): Description.
            W (Any): Description.
        Returns:
            Any: Description.

        forward method for PatchMerging.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            H (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            W (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, C = x.shape
        x = x.reshape(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.reshape(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """BasicLayer class."""

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):
        """__init__.

        Args:
            dim (Any): Description.
            depth (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            mlp_ratio (Any): Description.
            qkv_bias (Any): Description.
            qk_scale (Any): Description.
            drop (Any): Description.
            attn_drop (Any): Description.
            norm_layer (Any): Description.
            downsample (Any): Description.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerKANBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ],
        )
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """forward.

        Args:
            x (Any): Description.
            H (Any): Description.
            W (Any): Description.
        Returns:
            Any: Description.

        forward method for BasicLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            H (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            W (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_down, Wh, Ww, x_down, Wh, Ww
        return x, H, W, x, H, W


@register_model(name="swin_kan_generator", training_mode="gan")
class SwinKANGenerator(nn.Module):
    """SwinKANGenerator class."""

    def __init__(
        self,
        img_size=240,
        patch_size=4,
        in_chans=1,
        num_classes=1,
        embed_dim=96,
        depths=None,
        num_heads=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        """__init__.

        Args:
            img_size (Any): Description.
            patch_size (Any): Description.
            in_chans (Any): Description.
            num_classes (Any): Description.
            embed_dim (Any): Description.
            depths (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            mlp_ratio (Any): Description.
            qkv_bias (Any): Description.
            qk_scale (Any): Description.
            drop_rate (Any): Description.
            attn_drop_rate (Any): Description.
            norm_layer (Any): Description.
        """
        if num_heads is None:
            num_heads = [3, 6, 12]
        if depths is None:
            depths = [2, 2, 2]
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
            norm_layer(embed_dim),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            self.layers.append(
                BasicLayer(
                    dim=int(embed_dim * 2**i),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i < len(depths) - 1) else None,
                ),
            )

        final_dim = int(embed_dim * 2 ** (len(depths) - 1))
        self.norm = norm_layer(final_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(final_dim, embed_dim * 4, 4, 4),
            nn.InstanceNorm2d(embed_dim * 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim * 4, embed_dim, 4, 4),
            nn.InstanceNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, in_chans, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.final_upsample = lambda x: F.interpolate(
            x,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for SwinKANGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = x.shape[2] // 4, x.shape[3] // 4
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W, _, _, _ = layer(x, H, W)

        x = self.norm(x)
        B, L, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = self.decoder(x)
        x = self.final_upsample(x)
        return x


def get_generator(
    in_channels=1,
    out_channels=1,
    img_size=240,
    patch_size=4,
    embed_dim=96,
    depths=None,
    num_heads=None,
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    norm_layer=nn.LayerNorm,
    **kwargs,
):
    """get_generator.

    Args:
        in_channels (Any): Description.
        out_channels (Any): Description.
        img_size (Any): Description.
        patch_size (Any): Description.
        embed_dim (Any): Description.
        depths (Any): Description.
        num_heads (Any): Description.
        window_size (Any): Description.
        mlp_ratio (Any): Description.
        qkv_bias (Any): Description.
        qk_scale (Any): Description.
        drop_rate (Any): Description.
        attn_drop_rate (Any): Description.
        norm_layer (Any): Description.
    Returns:
        Any: Description.
    """
    return SwinKANGenerator(
        in_chans=in_channels,
        num_classes=out_channels,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        norm_layer=norm_layer,
        **kwargs,
    )
